from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

from backend.core.config import settings

database_url = URL.create(
    drivername="mysql+pymysql",
    username=settings.DB_USERNAME,
    password=settings.DB_PASSWORD,
    host=settings.DB_HOST,
    port=settings.DB_PORT,
    database=settings.DB_NAME,
    query={"charset": "utf8mb4"},
)

ssl_config = {"ca": settings.DB_SSL_CA} if settings.DB_SSL_CA else {}

connect_args: dict[str, Any] = {"connect_timeout": settings.DB_TIMEOUT}
if ssl_config:
    connect_args["ssl"] = ssl_config

engine = create_engine(
    database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
