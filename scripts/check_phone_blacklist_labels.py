from sqlalchemy import create_engine, text

from backend.core.config import settings

engine = create_engine(
    (
        f"mysql+pymysql://{settings.DB_USERNAME}:{settings.DB_PASSWORD}@"
        f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}?charset=utf8mb4"
    ),
    connect_args={"ssl": {"ca": settings.DB_SSL_CA}} if settings.DB_SSL_CA else {},
    pool_pre_ping=True,
)

with engine.connect() as conn:
    labels = conn.execute(
        text("SELECT DISTINCT phone_type FROM phone WHERE status = 'black' ORDER BY phone_type")
    ).fetchall()
    values = [row[0] for row in labels]
    print("BLACKLIST_LABELS:", values)
    if not values:
        print("NO_BLACKLIST_LABELS_FOUND")
