"""Launch TriSense Studio. Local .runtime is optional and ignored by Git."""
import argparse
from pathlib import Path
import sys
import threading
import webbrowser

ROOT = Path(__file__).resolve().parent
if (ROOT / ".runtime").exists():
    sys.path.insert(0, str(ROOT / ".runtime"))


def main():
    parser = argparse.ArgumentParser(description="TriSense Studio local application")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        import uvicorn
        from trisense.api import app
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Run: python -m pip install -r requirements.txt")
    url = f"http://127.0.0.1:{args.port}"
    print(f"TriSense Studio: {url}\nPress Ctrl+C to stop. Features stay on this computer.")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
