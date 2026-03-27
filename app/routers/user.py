"""
User router — login/register/reset removed.
ctrlX CORE handles authentication at the OS level.
Only the logout route is kept so existing template links don't 404.
"""

from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from fastapi import status

router = APIRouter(tags=['User'])


@router.get("/logout")
def logout():
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("access_token")
    return response
