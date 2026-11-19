FROM python@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 SQLITE_TMPDIR=/state/tmp
WORKDIR /app
COPY backup.py /app/backup.py
COPY dashboard.py index.html app.js /app/
CMD ["python", "/app/backup.py", "daemon"]
