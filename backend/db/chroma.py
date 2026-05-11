import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import settings
from backend.db.rebuild import rebuild_index


def main() -> None:
    dataset_path = Path(settings.RAG_DATASET_PATH)
    persist_dir = Path(settings.RAG_PERSIST_DIR)

    print(f"Dataset path: {dataset_path}")
    print(f"Persist dir: {persist_dir}")

    rebuild_index()
    print("ChromaDB build completed.")


if __name__ == "__main__":
    main()
