import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from app.config import setting

router = APIRouter(prefix="/logs", tags=["Logs"])

_LOG_DIR = Path(setting.log_dir).resolve()


def _safe_log_path(filename: str) -> Path:
    """Resolve path and reject anything outside the log directory."""
    path = (_LOG_DIR / filename).resolve()
    if not path.is_relative_to(_LOG_DIR):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Log file not found.")
    return path


@router.get("", response_class=PlainTextResponse, include_in_schema=False)
async def list_logs_html():
    """Redirect bare /logs to the log viewer page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="logs/view")


@router.get("/view", include_in_schema=False)
async def log_viewer_page():
    from fastapi.responses import HTMLResponse
    files = []
    if _LOG_DIR.exists():
        for p in sorted(_LOG_DIR.rglob("*.log")):
            rel = p.relative_to(_LOG_DIR)
            size_kb = round(p.stat().st_size / 1024, 1)
            files.append((str(rel), size_kb))

    rows = "".join(
        f'<tr>'
        f'<td class="text-start">{name}</td>'
        f'<td>{size} KB</td>'
        f'<td>'
        f'<a href="read/{name}" class="btn btn-sm btn-outline-light me-1">View</a>'
        f'<a href="download/{name}" class="btn btn-sm btn-outline-secondary">Download</a>'
        f'</td>'
        f'</tr>'
        for name, size in files
    ) or '<tr><td colspan="3" class="text-muted">No log files found.</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
  <meta charset="UTF-8"/>
  <title>MIND – Logs</title>
  <link href="/mind/static/bootstrap/css/bootstrap.min.css" rel="stylesheet"/>
</head>
<body class="bg-black text-white p-4">
  <h2 class="mb-4">Log Files <small class="text-muted fs-6">{str(_LOG_DIR)}</small></h2>
  <table class="table table-dark table-hover table-borderless align-middle">
    <thead><tr><th class="text-start">File</th><th>Size</th><th>Actions</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <a href="view" class="btn btn-outline-light btn-sm mt-2">&#8635; Refresh</a>
</body>
</html>"""
    return HTMLResponse(html)


@router.get("/read/{filename:path}", response_class=PlainTextResponse, include_in_schema=False)
async def read_log(filename: str):
    """Return the last 2000 lines of a log file as plain text."""
    path = _safe_log_path(filename)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    tail = "".join(lines[-2000:])
    return PlainTextResponse(tail, media_type="text/plain; charset=utf-8")


@router.get("/download/{filename:path}", include_in_schema=False)
async def download_log(filename: str):
    path = _safe_log_path(filename)
    return FileResponse(path, filename=path.name, media_type="text/plain")
