"""Migrate the previous local SQLite case/job store to PostgreSQL."""

import argparse

from backend.config import DATA_DIR
from backend.repository import initialize_database, migrate_from_sqlite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sqlite-path",
        default=str(DATA_DIR / "deeptrace.sqlite3"),
        help="Path to the old backend SQLite database.",
    )
    args = parser.parse_args()
    initialize_database()
    counts = migrate_from_sqlite(args.sqlite_path)
    print(f"Copied {counts['cases']} cases and {counts['jobs']} jobs into PostgreSQL.")


if __name__ == "__main__":
    main()
