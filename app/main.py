from contextlib import asynccontextmanager
import io
from logging.handlers import RotatingFileHandler

from pathlib import Path
import time
from typing import Optional
import zipfile
from fastapi import Form, status
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, Request, Depends, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import pandas as pd
from app import database, models
from app.middleware.auth_middleware import AuthMiddleware
from app.scripts.auth import get_effective_user
from app.scripts.database_helper import load_measurements_by_app
from app.scripts.report_service import ReportService
from app.utils import Utils
from app.routers import user, api, prometheus, htmx
from app.routers.prometheus import PrometheusMiddleware
import logging
from app.database import session_manager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy.orm.attributes import set_committed_value
from app.config import setting
import sass
from app.scripts.tsa_logging import create_logger
from ruamel.yaml import YAML
from uuid import UUID

tsa_logger = create_logger("main", output_file="mind_api")

uvicorn_file_handler = RotatingFileHandler(
    f"{setting.log_dir}/uvicorn.log",
    maxBytes=30*1024*1024,  # 30 MB
    backupCount=5           # Keep 5 backup files
)
uvicorn_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger("uvicorn.access").addHandler(uvicorn_file_handler)
logging.getLogger("uvicorn.error").addHandler(uvicorn_file_handler)


# compile scss to css
start = time.perf_counter()
sass.compile(dirname=('app/static/scss', 'app/static/css'), output_style='compressed')
end = time.perf_counter()
print(f"SASS Compiler runs in: {end-start}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Define the lifespan context manager for the FastAPI application
    """
    # Initialize the database session manager when the app starts
    session_manager.init_db()
    tsa_logger.info("Application startup complete. version: %s", app_version)
    yield  # Yield control back to the application
    tsa_logger.info("Waiting for application shutdown.")
    # Close the database session manager when the app shuts down
    await session_manager.close()
    tsa_logger.info("Application shutdown complete.")

app_version = Utils.get_value_from_pyproject('version')
app = FastAPI(title="MAXON MIND", version=app_version, lifespan=lifespan)
# mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


app.add_middleware(CORSMiddleware,
                   allow_credentials=True,
                   allow_origins=["http://localhost:8000", "http://localhost:8081", "http://srvmint000227", "http://srvmint000227:8081"],
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )
app.add_middleware(AuthMiddleware)
app.add_middleware(PrometheusMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)  # Compress responses larger than 1000 bytes

app.include_router(user.router)
app.include_router(api.router)
app.include_router(prometheus.router)
app.include_router(htmx.router)

yaml = YAML()
yaml.preserve_quotes = True


@app.exception_handler(status.HTTP_404_NOT_FOUND)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return RedirectResponse(url="/")


@app.get('/')
async def root(context: dict = Depends(Utils.prepareBaseContext)):
    response = templates.TemplateResponse("index.html", context)
    response.set_cookie(key="lang", value=context['lang'])
    return response


@app.get('/create_user')
def register(context: dict = Depends(Utils.prepareBaseContext)):
    return templates.TemplateResponse("register.html", context)


@app.get('/info')
async def info(context: dict = Depends(Utils.prepareBaseContext)):
    return templates.TemplateResponse("info.html", context)


@app.get('/train')
def train_ui(context: dict = Depends(Utils.prepareBaseContext)):
    context['task_type'] = 'train'
    response = templates.TemplateResponse("train_predict.html", context)
    response.delete_cookie("uploaded_file")
    response.delete_cookie("run_id")
    return response


@app.get('/import')
def import_zip(context: dict = Depends(Utils.prepareBaseContext)):
    return templates.TemplateResponse("import.html", context)


@app.get('/admin')
async def admin(context: dict = Depends(Utils.prepareBaseContext),
                user: Optional[models.User] = Depends(get_effective_user),
                db: AsyncSession = Depends(database.get_db)):
    if not user.isAdmin():
        return RedirectResponse(url="/", status_code=status.HTTP_401_UNAUTHORIZED)

    # Load all users
    users = (await db.execute(
        select(models.User)
        .options(selectinload(models.User.motors))
        .order_by(models.User.id))
    ).scalars().all()
    dto_users = []
    for user in users:
        dto_users.append({
            'email': user.email,
            'role': user.role.value,
            'id': user.id,
            'disabled': user.disabled,
            'motors': [m.id for m in user.motors],
            'last_login': Utils.strftime(user.last_login)
        })
    context['users'] = dto_users

    # load pipeline.yml keep comments
    file = Path("model_config.yaml")
    config = file.read_text(encoding='utf-8') if file.exists() else ""
    context['model_config'] = config
    return templates.TemplateResponse("admin.html", context)


@app.post('/update-config', response_class=HTMLResponse)
async def update_config(pipelineYml: str = Form(...),
                        user: Optional[models.User] = Depends(get_effective_user)):
    if not user.isAdmin():
        return RedirectResponse(url="/", status_code=status.HTTP_401_UNAUTHORIZED)

    # write input to model_config.yaml
    try:
        config = yaml.load(pipelineYml)
        with open('model_config.yaml', 'w') as file:
            yaml.dump(config, file)
            tsa_logger.info("Configuration updated by user: %s", user.email)
            return '<div class="alert alert-success">Configuration saved successfully!</div>'
    except Exception:
        tsa_logger.exception("Error updating configuration")
        return '<div class="alert alert-danger">Error saving configuration. Please try again.</div>'


@app.post("/update-user", response_class=HTMLResponse)
async def update_user(user_id: int = Form(...),
                      role: Optional[str] = Form(...),
                      motors: Optional[str] = Form(...),
                      disabled: Optional[bool] = Form(None),
                      user: Optional[models.User] = Depends(get_effective_user),
                      db: AsyncSession = Depends(database.get_db)):
    if not user.isAdmin():
        return RedirectResponse(url="/", status_code=status.HTTP_401_UNAUTHORIZED)
    # Update user in the database
    stmt = select(models.User).options(selectinload(models.User.motors)).where(models.User.id == user_id)
    db_user = (await db.execute(stmt)).scalars().one_or_none()
    if db_user is None:
        return '<div class="alert alert-danger">User not found!</div>'
    db_user.role = role
    db_user.disabled = disabled
    motor_ids = [int(mid) for mid in motors.split(',') if mid.strip().isdigit()]
    if motor_ids:
        motor_stmt = select(models.Motor).where(models.Motor.id.in_(motor_ids))
        motors_list = (await db.execute(motor_stmt)).scalars().all()
        db_user.motors.clear()
        db_user.motors.update(motors_list)
    else:
        db_user.motors.clear()
    await db.commit()
    tsa_logger.info("User %s updated by admin %s", db_user.email, user.email)
    return '<div class="alert alert-success">User updated successfully!</div>'


@app.get('/predict')
@app.get('/predict/{application_id}')
@app.get('/predict/{application_id}/{run_id}')
async def predict_ui(context: dict = Depends(Utils.prepareBaseContext),
                     application_id: Optional[int] = None,
                     run_id: Optional[str] = None,
                     user: Optional[models.User] = Depends(get_effective_user),
                     db: AsyncSession = Depends(database.get_db)):
    if application_id:
        stmt = (
            select(models.Application)
            .join(models.Motor)
            .where(models.Application.id == application_id,
                   ~models.Motor.disabled,
                   ~models.Application.disabled)
            .options(selectinload(models.Application.motor))
        )
        if not user.isAdmin():
            stmt = (
                stmt.join(models.motor_user)
                    .where(models.motor_user.c.user_id == user.id)
            )

        application = (await db.execute(stmt)).scalars().one_or_none()
        if application:
            context['application_dto'] = application
            if run_id:
                context['run_id'] = run_id

    context['task_type'] = 'predict'
    response = templates.TemplateResponse("train_predict.html", context)
    response.delete_cookie("uploaded_file")
    response.delete_cookie("run_id")
    return response


@app.get('/analytics')
async def analytics(context: dict = Depends(Utils.prepareBaseContext)):
    return templates.TemplateResponse("analytics.html", context)


@app.get('/report/{run}')
async def report(run: str,
                 user: models.User = Depends(get_effective_user),
                 context: dict = Depends(Utils.prepareBaseContext),
                 db: AsyncSession = Depends(database.get_db)):
    context['data'] = await ReportService.get_report_data(run_id=run, user=user, db=db, context=context)
    return templates.TemplateResponse("/report.html", context)


@app.get('/export/{application_id}')
async def export(application_id: int,
                 user: models.User = Depends(get_effective_user),
                 db: AsyncSession = Depends(database.get_db)):
    stmt = (
        select(models.Application)
        .options(
            selectinload(models.Application.motor),
            selectinload(models.Application.cycledatas))
        .where(models.Application.id == application_id)
    )

    if not user.isAdmin():
        stmt = (
            stmt.join(models.Motor)
            .join(models.motor_user)
            .where(models.motor_user.c.user_id == user.id)
        )
    start = time.perf_counter()
    application = (await db.execute(stmt)).scalars().first()
    tsa_logger.info("Time taken to load application's data from db: %.2f s", time.perf_counter()-start)

    if not application:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not allowed to access this resource.")

    data = await load_measurements_by_app(db, application.id, logger=tsa_logger)
    if data is None:
        tsa_logger.error("Downloading data for application failed or returned None. Provided application-id: %d", application_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Downloading data failed. Please try again later.")

    zip_data = _create_zip(application, data)

    headers = {'Content-Disposition': f'attachment; filename="application_{application_id}.zip"'}
    return Response(zip_data.getvalue(), headers=headers, media_type='application/zip')


@app.get('/export_run/{run_id}')
async def export_run(run_id: str, user: Optional[models.User] = Depends(get_effective_user), db: AsyncSession = Depends(database.get_db)):
    try:
        UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Run id invalid")

    stmt = (
        select(models.Run)
        .options(
            selectinload(models.Run.application).selectinload(models.Application.motor),
            selectinload(models.Run.cycles),
        )
        .where(models.Run.id == run_id)
    )
    if not user.isAdmin():
        stmt = (
            stmt.join(models.Application)
            .join(models.Motor)
            .join(models.motor_user)
            .where(models.motor_user.c.user_id == user.id)
        )

    start = time.perf_counter()
    run = (await db.execute(stmt)).scalars().first()
    tsa_logger.info("Time taken to load run's data from db: %.2f s", time.perf_counter()-start)

    if not run:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not allowed to access this resource.")

    data = await load_measurements_by_app(db, run.application.id, cycle_ids={c.id for c in run.cycles}, logger=tsa_logger)
    if data is None:
        tsa_logger.error("Downloading data for application failed or returned None. Provided run-id: %s", run.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Downloading data failed. Please try again later.")

    # ensure that only the cycles belonging to the run are included in the export, even if the application has more cycles in the database
    set_committed_value(run.application, 'cycledatas', list(run.cycles))

    zip_data = _create_zip(run.application, data)

    headers = {'Content-Disposition': f'attachment; filename="run_{run.id}.zip"'}
    return Response(zip_data.getvalue(), headers=headers, media_type='application/zip')


def _create_zip(application: models.Application, data: pd.DataFrame) -> io.BytesIO:
    start = time.perf_counter()

    # Round real values
    if not data.empty:
        data['value'] = data['value'].round(6)

    zip_data = io.BytesIO()
    motor_info = (
        f"Motor:{{ serial:{application.motor.serial}, part:{application.motor.part}, "
        f"machine:{application.motor.machine}, client:{application.motor.client}, "
        f"time:{application.motor.time_created}, disabled:{application.motor.disabled} }}"
    )
    app_info = (
        f"Application:{{ context_code:{application.context_code}, recipe:{application.recipe}, "
        f"time:{application.time_created}, disabled:{application.disabled}}}"
    )

    # Group data by cycle_id for efficient CSV generation
    grouped = {
        cid: group[['timestamp', 'field', 'value']].to_csv(index=False, float_format='%g')
        for cid, group in data.groupby('cycle_id', sort=False)
    } if not data.empty else {}

    with zipfile.ZipFile(zip_data, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zip_file:
        for cycle in application.cycledatas:
            cid = cycle.id
            cycle_info = (
                f"CycleData:{{ driveConfig:{cycle.driveConfig}, cycleConfig:{cycle.cycleConfig}, "
                f"classification:{cycle.classification.label}, testCycleCounter:{cycle.testCycleCounter}, "
                f"id:{cid}, time:{cycle.time_created}}}"
            )

            csv_data = grouped.get(cid, "No measurements available")
            csv_content = f"{motor_info}\n{app_info}\n{cycle_info}\n{csv_data}"
            zip_file.writestr(f"cycle_{cycle.id}.csv", csv_content.encode("utf-8"))
    end = time.perf_counter()
    tsa_logger.info("Time taken generate the zip-file compression=zipfile.ZIP_DEFLATED, compresslevel=1: %.2f s", end-start)
    return zip_data
