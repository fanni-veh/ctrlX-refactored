"""
Database session management.

Provides an async SQLAlchemy engine and scoped session factory.
Supports both SQLite (development/snap) and PostgreSQL (production).

Usage:
    - Call `session_manager.init_db()` once at application startup (done in main.py lifespan).
    - Inject `get_db` as a FastAPI dependency to get a session per request.
"""

from typing import AsyncGenerator
from sqlalchemy.orm import sessionmaker
from app.config import setting
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine, async_scoped_session
from asyncio import current_task


class DatabaseSessionManager:
    """
    Manages the async SQLAlchemy engine and session lifecycle.

    Call `init_db()` once at startup, then use `session()` as an async
    context manager to get a scoped session tied to the current asyncio task.
    Call `close()` at shutdown to cleanly dispose of the connection pool.
    """

    def __init__(self, database_url: str):
        self.engine: AsyncEngine | None = None
        self.session_maker = None
        self.session: async_scoped_session[AsyncSession] = None
        self.database_url = database_url

    def init_db(self):
        """
        Create the async engine and session factory.

        SQLite uses a single-thread connection (no pool needed).
        PostgreSQL uses a connection pool (size 50, overflow 20).
        """
        is_sqlite = self.database_url.startswith("sqlite")
        engine_kwargs = (
            {"connect_args": {"check_same_thread": False}}
            if is_sqlite
            else {"pool_size": 50, "max_overflow": 20, "pool_pre_ping": True}
        )
        self.engine = create_async_engine(self.database_url, **engine_kwargs)
        self.session_maker = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine, class_=AsyncSession, expire_on_commit=False
        )
        # Scope each session to the current asyncio task so concurrent requests don't share state
        self.session = async_scoped_session(self.session_maker, scopefunc=current_task)

    async def close(self):
        """Dispose the engine and close all pooled connections. Called at app shutdown."""
        if self.engine is None:
            raise Exception("DatabaseSessionManager is not initialized")
        await self.session.close()
        await self.engine.dispose()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields one AsyncSession per request.

    Usage in a route:
        db: AsyncSession = Depends(get_db)
    """
    async with session_manager.session() as session:
        yield session


session_manager = DatabaseSessionManager(setting.database_url)
