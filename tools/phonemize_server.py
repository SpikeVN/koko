#!/usr/bin/env python3
"""sea-g2p phonemize endpoint — the ~15-line wrapper koko's Phonemizer expects.

POST http://127.0.0.1:8788/phonemize  {"text": "..."}
 -> {"phonemes": "..."}

sea_g2p needs Python >= 3.10 (Rust abi3 wheel), so on the Jetson this runs on
the phonemize host; on the laptop it runs anywhere (single process).

Usage: python tools/phonemize_server.py [--host 0.0.0.0] [--port 8788] [--lang vi]
"""
import argparse
import json
import sys

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sea_g2p import SEAPipeline   # pip install sea-g2p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--lang", default="vi", choices=("vi", "th", "id"))
    args = ap.parse_args()

    pipeline = SEAPipeline(lang=args.lang)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                n = int(self.headers.get("content-length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                text = str(req.get("text", ""))
                phonemes = pipeline.run(text, punc_norm=True)
                body = json.dumps({"phonemes": phonemes}).encode("utf-8")
            except Exception as e:  # noqa: BLE001 - report, don't crash the server
                body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode("utf-8")
                self.send_response(500)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *a):  # chatty; drop
            pass

    print(f"phonemize server (sea-g2p, lang={args.lang}) on {args.host}:{args.port}",
          file=sys.stderr)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
