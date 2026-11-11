"""Command-line interface.

Usage: python3 -m inventory.cli COMMAND ...
Commands: add SKU NAME | receive SKU QTY | ship SKU QTY | report
Errors print "error: <message>" to stderr and return 1; the database is left unchanged.
"""
import argparse
import sys

from inventory import persist, report
from inventory.store import InsufficientStock

DEFAULT_DB = "inventory.json"


def build_parser():
    parser = argparse.ArgumentParser(prog="inventory")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("add")
    p.add_argument("sku")
    p.add_argument("name")
    p = sub.add_parser("receive")
    p.add_argument("sku")
    p.add_argument("qty", type=int)
    p = sub.add_parser("ship")
    p.add_argument("sku")
    p.add_argument("qty", type=int)
    sub.add_parser("report")
    return parser


def main(argv, db_path):
    args = build_parser().parse_args(argv)
    try:
        store = persist.load(db_path)
        if args.command == "report":
            print(report.render(store))
            return 0
        if args.command == "add":
            store.add_item(args.sku, args.name)
        elif args.command == "receive":
            store.receive(args.sku, args.qty)
        elif args.command == "ship":
            store.ship(args.sku, args.qty)
        persist.save(store, db_path)
    except (KeyError, ValueError, InsufficientStock) as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], DEFAULT_DB))
