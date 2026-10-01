"""Primary FedSOAR command-line entry point."""

from fedhydra.cli import main as _legacy_main


def main(argv: list[str] | None = None) -> int:
    return _legacy_main(argv, prog="fedsoar")


if __name__ == "__main__":
    raise SystemExit(main())
