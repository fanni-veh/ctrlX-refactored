from typing import Annotated, Dict, Optional
from fastapi import Header, Request, status, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from app.config import setting
from app import database, dto_schema, models
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.scripts.tsa_logging import create_logger

logger = create_logger(__name__, output_file="mind_api")


class CookieOAuth2PasswordBearer(OAuth2PasswordBearer):
    """
    OAuth2PasswordBearer that extracts token from cookie
    """

    def __init__(self, tokenUrl: str = "login", scheme_name: Optional[str] = None, scopes: Dict[str, str] | None = None, description: Optional[str] = None, auto_error: bool = True):
        super().__init__(tokenUrl, scheme_name, scopes, description, auto_error)

    async def __call__(self, request: Request) -> str:
        return request.cookies.get("access_token")


oauth2_scheme = CookieOAuth2PasswordBearer(tokenUrl='login')


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=setting.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, setting.jwt_secret, algorithm=setting.algorithm)
    return encoded_jwt


def create_reset_token(user: models.User):
    expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode = {
        'user_id': user.id,
        'type': 'reset',
        'exp': expire
    }
    secret = user.password+'-'+str(user.time_created)
    encoded_jwt = jwt.encode(to_encode, secret)
    return encoded_jwt


def verify_access_token(token: str, credentials_exception: HTTPException) -> dto_schema.TokenData:
    try:
        payload = jwt.decode(token, setting.jwt_secret, algorithms=[setting.algorithm])
        user_id = payload.get("user_id")
        if user_id is None:
            raise credentials_exception

        token_data = dto_schema.TokenData(id=str(user_id))  # validation
    except JWTError as e:
        logger.warning(f"JWTError: {e}")
        raise credentials_exception

    return token_data


def verify_reset_token(user: models.User, token: str):
    if user is None:
        return False

    secret = user.password+'-'+str(user.time_created)
    payload = jwt.decode(token, secret)
    user_id = payload.get("user_id")
    return user.id == user_id


async def get_effective_user(
        token: Annotated[str, Depends(oauth2_scheme)],
        db: AsyncSession = Depends(database.get_db),
        on_behalf_of_user_id: Annotated[Optional[int], Header(alias="X-On-Behalf-Of")] = None) -> models.User:

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials"
    )
    tokenData: dto_schema.TokenData = verify_access_token(token, credentials_exception)

    user = await db.scalar(
        select(models.User)
        .where(models.User.id == int(tokenData.id),
               ~models.User.disabled
               )
    )
    if user is None:
        raise credentials_exception

    if on_behalf_of_user_id is None:
        return user

    # Only admin/service users can act on behalf of another user
    if not user.isAdmin() and not user.isService():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to act on behalf of another user")
    effective_user = await db.scalar(
        select(models.User)
        .where(models.User.id == on_behalf_of_user_id,
               ~models.User.disabled
               )
    )
    if effective_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found")

    logger.info(f"Request called by {user.id} on behalf of user {on_behalf_of_user_id}, switched user context.")
    return effective_user


async def get_current_active_user(request: Request, db: AsyncSession) -> models.User | None:
    token = request.cookies.get('access_token')
    if token is not None:
        try:
            user = await get_effective_user(token, db)
            return user
        except Exception:
            return None
    return None
