"""Small entry point; experiments stay explicit modules."""
import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="ai4ct", description="AI4CT research workspace")
    parser.add_argument("command", nargs="?", choices=("doctor", "export-metrics"))
    # Delegate subcommand help to the relevant parser.
    args = parser.parse_args(argv[:1])
    if args.command == "doctor":
        from .doctor import main as doctor_main
        return doctor_main(argv[1:])
    if args.command == "export-metrics":
        from .features.export_metrics import main as export_main
        return export_main(argv[1:])
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
