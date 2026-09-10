import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.config import settings
from backend.persistence.mysql_connection import engine
from sqlalchemy import text


def main():
    print("Using DB settings:")
    print("  host:", settings.DB_HOST)
    print("  port:", settings.DB_PORT)
    print("  name:", settings.DB_NAME)
    print("  user:", settings.DB_USERNAME)
    print("  ssl_ca:", settings.DB_SSL_CA)

    # check for ca file
    possible_paths = [
        Path(settings.DB_SSL_CA),
        Path(__file__).resolve().parent.parent / settings.DB_SSL_CA,
        Path(__file__).resolve().parent / settings.DB_SSL_CA,
    ]
    found = None
    for p in possible_paths:
        if p.exists():
            found = p
            break

    if found:
        print("Found CA file at:", found)
    else:
        print("CA file not found in project paths; SSL may still work if DB does not require CA file")

    print("Attempting DB connection (this may take a few seconds)...")
    try:
        with engine.connect() as conn:
            # simple test query
            res = conn.execute(text("SELECT 1")).scalar()
            print("Query result:", res)
            print("Connection successful.")
    except Exception as e:
        print("Connection failed:", type(e).__name__, str(e))


if __name__ == "__main__":
    main()
