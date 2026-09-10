from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

from backend.core.config import settings

host_name = settings.DB_HOST.strip()
port_number = settings.DB_PORT
if host_name and ":" in host_name and host_name.count(":") == 1:
    host_candidate, port_candidate = host_name.rsplit(":", 1)
    if port_candidate.isdigit():
        host_name = host_candidate
        port_number = int(port_candidate)

database_url = URL.create(
    drivername="mysql+pymysql",
    username=settings.DB_USERNAME,
    password=settings.DB_PASSWORD,
    host=host_name,
    port=port_number,
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
