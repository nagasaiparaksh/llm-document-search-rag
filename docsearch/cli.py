"""Command line interface.

    python -m docsearch.cli ingest path/to/file.pdf      # or a folder
    python -m docsearch.cli ask "What is the notice period?"
    python -m docsearch.cli watch data/inbox             # auto-index anything dropped in
    python -m docsearch.cli list
    python -m docsearch.cli serve --port 8000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .services import build_services


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docsearch")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest"); p.add_argument("path")
    p = sub.add_parser("ask"); p.add_argument("question"); p.add_argument("-k", type=int)
    p = sub.add_parser("watch"); p.add_argument("folder", nargs="?"); p.add_argument("--interval", type=float, default=5)
    sub.add_parser("list")
    p = sub.add_parser("serve"); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("docsearch.api:app_factory", factory=True, host=args.host, port=args.port)
        return 0

    svc = build_services()
    if args.cmd == "ingest":
        target = Path(args.path)
        results = svc.ingestor.ingest_folder(target) if target.is_dir() else [svc.ingestor.ingest_file(target)]
        for r in results:
            print(f"{r.file:<40} {r.status:<10} chunks={r.chunks} {r.error}")
    elif args.cmd == "ask":
        ans = svc.answerer.ask(args.question, args.k)
        print(f"\n{ans.answer}\n")
        for s in ans.sources:
            mark = "*" if s.cited else " "
            print(f" {mark}[{s.n}] {s.file} p.{s.page}  (similarity {s.score})")
    elif args.cmd == "list":
        for d in svc.ingestor.documents():
            print(f"{d['file']:<40} {d['chunks']} chunks")
        print(f"total chunks: {svc.index.count()}")
    elif args.cmd == "watch":
        folder = args.folder or svc.settings.inbox_dir
        print(f"Watching {folder} (Ctrl+C to stop)")
        try:
            svc.ingestor.watch(folder, args.interval)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
