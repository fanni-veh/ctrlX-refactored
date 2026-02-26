from typing import AsyncGenerator
from sqlalchemy.orm import sessionmaker
from app.config import setting
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine, async_scoped_session
from asyncio import current_task


class DatabaseSessionManager:
    def __init__(self, database_url: str):
        self.engine: AsyncEngine | None = None
        self.session_maker = None
        self.session: async_scoped_session[AsyncSession] = None
        self.database_url = database_url

    def init_db(self):
        is_sqlite = self.database_url.startswith("sqlite")
        engine_kwargs = (
            {"connect_args": {"check_same_thread": False}}
            if is_sqlite
            else {"pool_size": 50, "max_overflow": 20, "pool_pre_ping": True}
        )
        # Creating an asynchronous engine
        self.engine = create_async_engine(
            self.database_url, **engine_kwargs
        )

        # Creating an asynchronous session class
        self.session_maker = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine, class_=AsyncSession, expire_on_commit=False
        )

        # Creating a scoped session
        self.session = async_scoped_session(self.session_maker, scopefunc=current_task)

    async def close(self):
        # Closing the database session
        if self.engine is None:
            raise Exception("DatabaseSessionManager is not initialized")
        await self.session.close()
        await self.engine.dispose()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with session_manager.session() as session:
        yield session

session_manager = DatabaseSessionManager(setting.database_url)
