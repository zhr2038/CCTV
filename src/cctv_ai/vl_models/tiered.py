from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Mapping, Sequence

from ..model_config import VisionLanguageConfig
from .base import (
    StructuredVisualEvent,
    VisionAnalysisContext,
    VisionLanguageModel,
)
from .factory import create_vision_language_model
from .qwen3_vl import _checkpoint_complete


MILESTONE_ACTIVITIES = {
    "rolling": "first_roll",
    "sitting": "first_sit",
    "crawling": "first_crawl",
    "standing": "first_stand",
    "speaking": "first_words",
}


@dataclass(frozen=True)
class TieredAnalysis:
    primary: StructuredVisualEvent
    milestone: str | None
    status: str
    confirmation: StructuredVisualEvent | None = None

    @property
    def final_event(self) -> StructuredVisualEvent:
        return self.confirmation or self.primary

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.to_dict(),
            "milestone": self.milestone,
            "status": self.status,
            "confirmation": (
                self.confirmation.to_dict()
                if self.confirmation is not None
                else None
            ),
        }


class TieredVisionAnalyzer:
    """Run normal frames on 8B and only milestone candidates on 32B."""

    def __init__(
        self,
        primary_config: VisionLanguageConfig,
        high_value_config: VisionLanguageConfig,
        *,
        factory: Callable[
            [VisionLanguageConfig],
            VisionLanguageModel,
        ] = create_vision_language_model,
    ) -> None:
        self.primary_config = primary_config
        self.high_value_config = high_value_config
        self._factory = factory
        self._primary: VisionLanguageModel | None = None
        self._high_value: VisionLanguageModel | None = None

    def _primary_model(self) -> VisionLanguageModel:
        if self._primary is None:
            self._primary = self._factory(self.primary_config)
        return self._primary

    def _high_value_model(self) -> VisionLanguageModel:
        if self._high_value is None:
            if not _checkpoint_complete(
                Path(self.high_value_config.model_path)
            ):
                raise FileNotFoundError(
                    "High-value Qwen3-VL checkpoint is not installed: "
                    f"{self.high_value_config.model_path}"
                )
            # A 32B 4-bit checkpoint fits the 32 GB RTX 5090 by itself, but
            # retaining the 8B model at the same time leaves too little room
            # for multimodal activations. Release 8B before escalating.
            if self._primary is not None:
                self._primary.close()
                self._primary = None
            self._high_value = self._factory(self.high_value_config)
        return self._high_value

    def _candidate(
        self,
        event: StructuredVisualEvent,
        context: VisionAnalysisContext | Mapping[str, Any] | None,
    ) -> tuple[str | None, str | None]:
        milestone = MILESTONE_ACTIVITIES.get(event.activity)
        if milestone is None:
            return None, None
        if (
            event.uncertain
            or event.confidence
            < self.high_value_config.candidate_confidence
        ):
            return milestone, "candidate_below_threshold"
        if event.activity == "speaking":
            raw = context if isinstance(context, Mapping) else {}
            if not bool(raw.get("audio_transcript_confirmed", False)):
                # Lip movement cannot establish what word was spoken.
                return milestone, "pending_audio_evidence"
        return milestone, None

    def _confirmation_prompt(
        self,
        primary: StructuredVisualEvent,
        milestone: str,
        prompt: str,
    ) -> str:
        return (
            "这是高价值成长里程碑的二次复核。8B 候选如下：\n"
            f"{primary.to_json()}\n"
            f"候选里程碑：{milestone}\n"
            "独立检查画面，不要因为候选结论而默认同意。只有动作完整、"
            "主体清楚且直接可见时，才返回相同 activity；否则返回 "
            "unclear 并令 uncertain=true。第一次说话还必须有已确认的"
            "同步音频转写，不能根据嘴部动作猜词。\n"
            f"原始分析任务：{prompt}"
        )

    def analyze_image(
        self,
        image: Any,
        prompt: str,
        context: VisionAnalysisContext | Mapping[str, Any] | None = None,
    ) -> TieredAnalysis:
        primary = self._primary_model().analyze_image(
            image,
            prompt,
            context,
        )
        milestone, blocked_status = self._candidate(primary, context)
        if milestone is None:
            return TieredAnalysis(primary, None, "ordinary_event")
        if blocked_status is not None:
            return TieredAnalysis(primary, milestone, blocked_status)
        if not _checkpoint_complete(
            Path(self.high_value_config.model_path)
        ):
            return TieredAnalysis(primary, milestone, "pending_32b_model")
        confirmation = self._high_value_model().analyze_image(
            image,
            self._confirmation_prompt(primary, milestone, prompt),
            context,
        )
        confirmed = (
            not confirmation.uncertain
            and confirmation.activity == primary.activity
            and confirmation.confidence
            >= self.high_value_config.candidate_confidence
        )
        return TieredAnalysis(
            primary,
            milestone,
            "confirmed" if confirmed else "rejected_by_32b",
            confirmation,
        )

    def analyze_video_frames(
        self,
        frames: Sequence[Any],
        prompt: str,
        context: VisionAnalysisContext | Mapping[str, Any] | None = None,
    ) -> TieredAnalysis:
        primary = self._primary_model().analyze_video_frames(
            frames,
            prompt,
            context,
        )
        milestone, blocked_status = self._candidate(primary, context)
        if milestone is None:
            return TieredAnalysis(primary, None, "ordinary_event")
        if blocked_status is not None:
            return TieredAnalysis(primary, milestone, blocked_status)
        if not _checkpoint_complete(
            Path(self.high_value_config.model_path)
        ):
            return TieredAnalysis(primary, milestone, "pending_32b_model")
        confirmation = self._high_value_model().analyze_video_frames(
            frames,
            self._confirmation_prompt(primary, milestone, prompt),
            context,
        )
        confirmed = (
            not confirmation.uncertain
            and confirmation.activity == primary.activity
            and confirmation.confidence
            >= self.high_value_config.candidate_confidence
        )
        return TieredAnalysis(
            primary,
            milestone,
            "confirmed" if confirmed else "rejected_by_32b",
            confirmation,
        )

    def close(self) -> None:
        if self._primary is not None:
            self._primary.close()
        if self._high_value is not None:
            self._high_value.close()


def save_confirmed_milestone(
    connection: sqlite3.Connection,
    *,
    activity_event_id: int,
    analysis: TieredAnalysis,
) -> bool:
    """Attach confirmation to an existing event without changing the schema."""
    if not analysis.confirmed or analysis.milestone is None:
        return False
    row = connection.execute(
        """
        SELECT details_json
        FROM activity_events
        WHERE id = ?
        """,
        (activity_event_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Unknown activity event: {activity_event_id}")
    try:
        details = json.loads(row["details_json"] or "{}")
    except (TypeError, ValueError):
        details = {}
    details["growth_milestone"] = {
        "kind": analysis.milestone,
        "status": "confirmed",
        "analysis": analysis.to_dict(),
        "confirmed_at": int(time.time()),
    }
    connection.execute(
        """
        UPDATE activity_events
        SET details_json = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            json.dumps(
                details,
                ensure_ascii=False,
                sort_keys=True,
            ),
            int(time.time()),
            activity_event_id,
        ),
    )
    return True
