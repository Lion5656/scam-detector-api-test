import sys
import os

# ensure project root is on sys.path so `backend` package can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.core.config import settings

def main():
    print("DB_HOST=", settings.DB_HOST)
    print("DB_PORT=", settings.DB_PORT)
    print("DB_NAME=", settings.DB_NAME)
    print("DB_USERNAME=", settings.DB_USERNAME)
    print("DB_SSL_CA=", settings.DB_SSL_CA)


if __name__ == "__main__":
    main()
