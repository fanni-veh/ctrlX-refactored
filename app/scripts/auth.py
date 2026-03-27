"""
Authentication module — simplified for ctrlX deployment.

ctrlX CORE enforces its own authentication before any snap is reachable,
so per-user login is not needed. Every request is treated as an admin user
that is auto-created in the database on first startup.
"""

import bcrypt
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app import database, models
from app.scripts.tsa_logging import create_logger

logger = create_logger(__name__, output_file="mind_api")

_ADMIN_EMAIL = "admin@mind.local"


async def _get_or_create_admin(db: AsyncSession) -> models.User:
    """Return the built-in admin user, creating it in the DB if needed."""
    user = await db.scalar(
        select(models.User).where(models.User.email == _ADMIN_EMAIL)
    )
    if user is None:
        hashed = bcrypt.hashpw(b"mind-admin", bcrypt.gensalt(rounds=4)).decode()
        user = models.User(
            email=_ADMIN_EMAIL,
            password=hashed,
            role=models.User.Role.ADMIN,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        logger.info("Created built-in admin user: %s", _ADMIN_EMAIL)
    return user


async def get_effective_user(
    db: AsyncSession = Depends(database.get_db),
) -> models.User:
    return await _get_or_create_admin(db)


async def get_current_active_user(request: Request, db: AsyncSession) -> models.User:
    return await _get_or_create_admin(db)
