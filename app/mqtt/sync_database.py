from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from app.config import setting
import threading


class SyncDatabaseSessionManager:
    """
    Synchronous database session manager for MQTT client and background workers.
    Thread-safe with connection pooling.
    """

    DATABASE_URL = f"postgresql://{setting.database_username}:{setting.database_password}@{setting.database_hostname}:{setting.database_port}/{setting.database_name}"

    def __init__(self):
        self.engine: Engine | None = None
        self.session_factory: sessionmaker | None = None
        self._lock = threading.Lock()

    def init_db(self):
        """Initialize the database engine and session factory."""
        with self._lock:
            if self.engine is not None:
                return

            self.engine = create_engine(
                self.DATABASE_URL,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,
                connect_args={'options': '-c synchronous_commit=off'}
            )

            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions. Ensures proper cleanup."""
        if self.session_factory is None:
            self.init_db()
        db = self.session_factory()
        try:
            yield db
        finally:
            db.close()

    def close(self):
        """Close the database connection pool."""
        with self._lock:
            if self.engine is not None:
                self.engine.dispose()
                self.engine = None
                self.session_factory = None


sync_session_manager = SyncDatabaseSessionManager()
