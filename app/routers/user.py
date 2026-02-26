
import logging
import re
from typing import Optional
import bcrypt
from fastapi import Form, Request, status, HTTPException, Depends, APIRouter
from fastapi.responses import JSONResponse, RedirectResponse
from datetime import datetime

from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr, field_validator
from app.scripts.auth import create_access_token, create_reset_token, verify_reset_token
from app.scripts.mail import send_email_reset_pw
from app import models
from app import database
from app.utils import Utils
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.config import setting


router = APIRouter(tags=['User'])
logger = logging.getLogger('main')
templates = Jinja2Templates(directory="app/templates")


class UserLoginRequest(BaseModel):
    """ Model for user login request """
    email: EmailStr
    password: str

    @field_validator('email')
    @classmethod
    def normalize_email(cls, email: str) -> str:
        return email.lower()


class UserRegisterRequest(BaseModel):
    """ Model for user registration request """
    email: EmailStr
    password: str

    @field_validator('password')
    @classmethod
    def validate_password(cls, pw: str) -> str:
        if len(pw) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r'[A-Z]', pw):
            raise ValueError("Password requires uppercase letters")
        if not re.search(r'[a-z]', pw):
            raise ValueError("Password requires lowercase letters")
        if not re.search(r'[0-9]', pw):
            raise ValueError("Password requires numbers")
        if not re.search(r'[!@#$%^&*()_+\-=\[\]{};:,.<>?]', pw):
            raise ValueError("Password requires special characters")
        return pw

    @field_validator('email')
    @classmethod
    def strict_email(cls, email: str) -> str:
        return email.lower()


@router.post('/login')
async def login(user_credentials: UserLoginRequest, db: AsyncSession = Depends(database.get_db), context: dict = Depends(Utils.prepareBaseContext)):
    email = user_credentials.email
    password = user_credentials.password

    user = await db.execute(select(models.User).where(models.User.email.ilike(email)))
    user = user.scalars().first()
    if not user or not verify(password, user.password):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=context['invalid_credentials'])
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=context['account_disabled'])
    access_token = create_access_token(data={'user_id': user.id})
    user.last_login = datetime.now()
    await update_user_in_db(user, db)

    response = JSONResponse(content={"access_token": access_token, "token_type": "bearer"})
    # Set cookie with secure attributes
    # httpOnly to prevent JavaScript access
    # samesite to prevent CSRF
    # max_age to set cookie expiration time
    response.set_cookie(key="access_token", value=access_token, httponly=True, samesite="strict", max_age=setting.access_token_expire_minutes * 60)

    return response


@router.get('/reset')
@router.get('/reset/{user_id}/{token}')
async def get_reset_form(context: dict = Depends(Utils.prepareBaseContext),
                         user_id: Optional[int] = None,
                         token: Optional[str] = None,
                         db: AsyncSession = Depends(database.get_db)):
    context['type'] = 'reset'
    user: models.User = context.get('user')
    if user is not None and token is None:
        context['token'] = create_reset_token(user)
        context['userid'] = user.id
    elif user_id and token:
        try:
            if verify_reset_token((await db.execute(select(models.User).where(models.User.id == int(user_id)))).scalars().first(), token):
                context['token'] = token
                context['userid'] = user_id
        except Exception:
            logger.exception("Error verifying reset token - possibly expired")
            context['active_error_message'] = context['reset_token_expired']
            return templates.TemplateResponse("error.html", context)
    return templates.TemplateResponse("reset.html", context)


@router.post('/reset')
async def create_reset_link(request: Request,
                            username: str = Form(...),
                            db: AsyncSession = Depends(database.get_db),
                            context: dict = Depends(Utils.prepareBaseContext)):
    user = (await db.execute(select(models.User).where(models.User.email.ilike(username.strip())))).scalars().first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=context['invalid_credentials'])
    access_token = create_reset_token(user)
    reset_link = f"{str(request.url)}/{user.id}/{access_token}"
    send_email_reset_pw(user.email, reset_link)
    logger.info("User (%s) requested new password.", user.email)
    return True


@router.post('/reset/{user_id}/{token}')
async def get_partial_reset_form(user_id: str,
                                 token: str,
                                 password: str = Form(...),
                                 db: AsyncSession = Depends(database.get_db),
                                 context: dict = Depends(Utils.prepareBaseContext)):
    user = (await db.execute(select(models.User).where(models.User.id == int(user_id)))).scalars().first()
    if user is None or not verify_reset_token(user, token):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=context['no_account_yet'])
    hashed_password = hash(password)
    user.password = hashed_password
    await update_user_in_db(user, db)

    response = JSONResponse(content={}, status_code=status.HTTP_201_CREATED)
    return response


@router.get("/logout")
def logout():
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("access_token")
    return response


@router.post("/register")
async def create_user(user_credentials: UserRegisterRequest, db: AsyncSession = Depends(database.get_db), context: dict = Depends(Utils.prepareBaseContext)):
    email = user_credentials.email
    password = user_credentials.password

    existing_user = (await db.execute(select(models.User).where(models.User.email.ilike(email)))).scalars().first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=context['email_already_registered'])
    new_user = models.User(
        email=email,
        password=hash(password))
    new_user.last_login = datetime.now()

    await update_user_in_db(new_user, db)

    access_token = create_access_token(data={'user_id': new_user.id})

    response = JSONResponse(content={"access_token": access_token, "token_type": "bearer"}, status_code=status.HTTP_201_CREATED)
    # Set cookie with secure attributes
    # httpOnly to prevent JavaScript access
    # samesite to prevent CSRF
    # max_age to set cookie expiration time
    response.set_cookie(key="access_token", value=access_token, httponly=True, samesite="strict", max_age=setting.access_token_expire_minutes * 60)

    return response


async def update_user_in_db(user: models.User, session: AsyncSession):
    session.add(user)
    await session.commit()
    await session.refresh(user)


def hash(password: str):
    return bcrypt.hashpw(password=password.encode('utf-8'), salt=bcrypt.gensalt()).decode('utf-8')


def verify(plain_password: str, hashed_password: str):
    return bcrypt.checkpw(password=plain_password.encode('utf-8'), hashed_password=hashed_password.encode('utf-8'))
