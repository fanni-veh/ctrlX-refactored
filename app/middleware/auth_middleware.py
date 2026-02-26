from fastapi import Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.database import get_db
from app.scripts.auth import get_current_active_user


class AuthMiddleware(BaseHTTPMiddleware):
    allow_guest_urls = [
        '/static/',  # any static content (css, images etc.)
        '/htmx/login',  # login-form
        '/login',  # login-post
        '/create_user',  # register-form
        '/register',  # register-post
        '/reset',  # reset password
        '/htmx/set-language',  # change language
        '/metrics'  # Prometheus
    ]

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/":
            return await call_next(request)
        if any(request.url.path.startswith(url) for url in self.allow_guest_urls):
            return await call_next(request)

        async for db in get_db():
            user = await get_current_active_user(request, db)
        if not user:
            if request.url.path.startswith('/api'):
                return JSONResponse(
                    content="Login required",
                    status_code=status.HTTP_401_UNAUTHORIZED
                )
            if request.headers.get("HX-Request") == "true":
                response = HTMLResponse("", status_code=status.HTTP_401_UNAUTHORIZED)
                response.headers["HX-Redirect"] = "/"
                return response
            return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
        return await call_next(request)
