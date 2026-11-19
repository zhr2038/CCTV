"""Loopback-only review server with MP4 byte ranges; no directory listing."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
from pathlib import PurePosixPath
import json
import re
from urllib.parse import unquote, urlsplit


def resolve_asset(root, catalog, url_path):
    relative = unquote(urlsplit(url_path).path).lstrip('/') or 'index.html'
    if '\\' in relative or '\x00' in relative or any(':' in p or p == '..' for p in PurePosixPath(relative).parts):
        return None
    parts = PurePosixPath(relative).parts
    selected = Path(root).resolve()
    if parts and parts[0] == 'batches':
        if len(parts) < 3 or parts[1] not in catalog:
            return None
        selected = Path(catalog[parts[1]]).resolve()
        relative = '/'.join(parts[2:])
    path = (selected / relative).resolve()
    return path if path.is_relative_to(selected) and path.is_file() else None


def serve(batch, port, catalog_path=None):
    root = Path(batch).resolve()
    if not (root / "index.html").is_file():
        raise FileNotFoundError("Build a review page first")
    catalog = json.loads(Path(catalog_path).read_text(encoding='utf-8')) if catalog_path else {}
    for name, folder in catalog.items():
        if not re.fullmatch(r'[a-z0-9_-]+', name) or not (Path(folder)/'index.html').is_file():
            raise ValueError('Invalid review catalog entry')

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.respond(False)

        def do_GET(self):
            self.respond(True)

        def respond(self, body):
            path = resolve_asset(root, catalog, self.path)
            if path is None:
                self.send_error(404)
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            request = self.headers.get("Range")
            if request:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", request)
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = min(int(last), end) if last else end
                else:
                    start = max(0, size - int(last))
                if not 0 <= start <= end < size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            self.send_response(206 if request else 200)
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or path.suffix in {".json", ".js"}:
                content_type += "; charset=utf-8"
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if request:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if body:
                try:
                    with path.open("rb") as handle:
                        handle.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = handle.read(min(65536, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Review: http://127.0.0.1:{port}/index.html (loopback only)", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
