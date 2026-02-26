from datetime import datetime
import logging
import math
import typing
from uuid import UUID
from fastapi import BackgroundTasks, Depends, APIRouter, HTTPException, UploadFile, status as http_status, Query, Body
from typing import Optional
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import Column, delete, exists, func, or_
from app.dto_models import PreviewResponse
from app.mqtt.mqtt_client import MqttValidator
from app.mqtt.rule_engine import rule_engine
from app.scripts.report_service import ReportService
from app.scripts.serialisation import make_json_serializable
from app.update_message import UpdateMessage
from app.utils import Utils
from app.scripts.tsk_param import TskParam
from app.scripts.auth import get_effective_user
from app import UserRole, database, RunState, RunTask, Classification, models
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload


router = APIRouter(prefix='/api', tags=['Api'])
logger = logging.getLogger('main')


def get_swagger_responses(example):
    return {
        200: {
            "description": "Success",
            "content": {
                "application/json": {
                    "example": example
                }
            }
        }
    }


@router.get('/model', responses=get_swagger_responses(models.Model.example().db_model_to_json()))
async def get_model(db: AsyncSession = Depends(database.get_db),
                    user: models.User = Depends(get_effective_user),
                    model_id: Optional[int] = None,
                    application_id: Optional[int] = None,
                    model_name: Optional[str] = None,
                    selection: Optional[str] = Query(None, description="Select specific attributes like: model_name,model_metadata")):
    try:
        stmt = (
            select(models.Model)
            .join(models.Run)
            .join(models.Application)
            .join(models.Motor)
        )
        if not user.isAdmin():
            stmt = stmt.where(models.Motor.users.any(models.User.id == user.id))  # only allow if user has access to motor
        if model_id:
            stmt = stmt.where(models.Model.id == model_id)
        if application_id:
            stmt = stmt.where(models.Application.id == application_id)
        if model_name:
            stmt = stmt.where(models.Model.model_name.ilike(model_name))

        db_models = (await db.scalars(stmt)).all()
        fields = selection.replace(" ", "").split(",") if selection else None
        return [model.db_model_to_json(fields) for model in db_models]
    except Exception:
        await db.rollback()  # close transaction
        logger.exception("Error fetching model data")
        raise


@router.delete('/model', responses=get_swagger_responses(models.Model.example().db_model_to_json()))
async def delete_model(db: AsyncSession = Depends(database.get_db),
                       user: models.User = Depends(get_effective_user),
                       model_id: Optional[int] = Query(None, description="The id of one model to delete"),
                       run_id: Optional[str] = Query(None, description="The run_id to delete each associated model")):
    try:
        if bool(model_id) == bool(run_id):
            raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="Exactly one of 'model_id' or 'run_id' is required.")

        stmt = (
            select(models.Model.id)
            .join(models.Application)
            .join(models.Motor)
        )
        if not user.isAdmin():
            stmt = stmt.where(models.Motor.users.any(models.User.id == user.id))  # only allow if user has access to motor
        if model_id:
            stmt = stmt.where(models.Model.id == model_id)
        else:
            stmt = stmt.where(models.Model.run_id == run_id)

        model_ids = (await db.scalars(stmt)).all()
        if not model_ids:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Model not found")

        delete_stmt = delete(models.Model).where(models.Model.id.in_(model_ids))
        await db.execute(delete_stmt)
        await db.commit()
        count = len(model_ids)
        return f"{count} Model{'s' if count > 1 else ''} deleted"
    except Exception:
        await db.rollback()  # close transaction
        logger.exception("Error deleting model")
        raise


@router.get('/motor', responses=get_swagger_responses(models.Motor.example().db_model_to_json()))
async def get_motor(db: AsyncSession = Depends(database.get_db),
                    user: models.User = Depends(get_effective_user),
                    id: Optional[int] = None,
                    serial: Optional[str] = None,
                    part: Optional[str] = None,
                    machine: Optional[str] = None,
                    client: Optional[str] = None,
                    selection: Optional[str] = Query(None, description="Select specific attributes like: serial, part")):
    try:
        stmt = select(models.Motor)
        if not user.isAdmin():
            stmt = stmt.where(models.Motor.users.any(models.User.id == user.id))

        filters: dict[Column, typing.Any] = {
            models.Motor.id: id,
            models.Motor.serial: serial,
            models.Motor.part: part,
            models.Motor.machine: machine,
            models.Motor.client: client,
        }
        for column, value in filters.items():
            if value:
                stmt = stmt.where(column.ilike(value) if isinstance(value, str) else column == value)

        db_motors = (await db.scalars(stmt)).all()
        fields = selection.replace(" ", "").split(",") if selection else None
        return [motor.db_model_to_json(fields) for motor in db_motors]
    except Exception:
        await db.rollback()  # close transaction
        logger.exception("Error fetching motor")
        raise


@router.delete('/motor/{motor_id}',  status_code=http_status.HTTP_204_NO_CONTENT)
async def delete_motor(motor_id: int,
                       db: AsyncSession = Depends(database.get_db),
                       user: models.User = Depends(get_effective_user)):
    try:
        stmt = (
            select(models.Motor)
            .options(selectinload(models.Motor.application))
            .where(models.Motor.id == motor_id)
        )
        if not user.isAdmin():
            stmt = stmt.where(models.Motor.users.any(models.User.id == user.id))  # only allow if user has access to motor

        db_motor = (await db.scalars(stmt)).first()
        if not db_motor:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Motor not found")

        await db.delete(db_motor)
        await db.commit()
        for app in db_motor.application:
            rule_engine.write_rules(app.id, None, user.id)
    except Exception:
        await db.rollback()  # close transaction
        logger.exception("Error deleting motor")
        raise


@router.get('/application', responses=get_swagger_responses(models.Application.example().db_model_to_json()))
async def get_application(db: AsyncSession = Depends(database.get_db),
                          user: models.User = Depends(get_effective_user),
                          id: Optional[int] = None,
                          motor_id: Optional[int] = None,
                          context_code: Optional[str] = None,
                          recipe: Optional[str] = None,
                          selection: Optional[str] = Query(None, description="Select specific attributes like: id, motor_id")):
    try:
        stmt = select(models.Application).join(models.Motor)
        if not user.isAdmin():
            stmt = stmt.where(models.Motor.users.any(models.User.id == user.id))  # only allow if user has access to motor

        filters: dict[Column, typing.Any] = {
            models.Application.id: id,
            models.Application.context_code: context_code,
            models.Application.recipe: recipe,
        }
        for column, value in filters.items():
            if value:
                stmt = stmt.where(column.ilike(value) if isinstance(value, str) else column == value)

        db_application = (await db.scalars(stmt)).all()
        fields = selection.replace(" ", "").split(",") if selection else None
        return [app.db_model_to_json(fields) for app in db_application]
    except Exception:
        await db.rollback()  # close transaction
        logger.exception("Error fetching application")
        raise


@router.delete('/application/{application_id}', status_code=http_status.HTTP_204_NO_CONTENT)
async def delete_application(application_id: int,
                             db: AsyncSession = Depends(database.get_db),
                             user: models.User = Depends(get_effective_user)):
    try:
        stmt = (
            select(models.Application)
            .join(models.Motor)
            .where(
                models.Application.id == application_id
            )
        )
        if not user.isAdmin():
            stmt = stmt.where(models.Motor.users.any(models.User.id == user.id))  # only allow if user has access to motor

        # Delete application from database
        db_app = (await db.scalars(stmt)).first()
        if not db_app:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Application not found")

        await db.delete(db_app)
        await db.commit()
        rule_engine.write_rules(application_id, None, user.id)
    except Exception:
        await db.rollback()  # close transaction
        logger.exception("Error deleting application")
        raise


@router.get('/cycle', responses=get_swagger_responses(models.CycleData.example().db_model_to_json()))
async def get_cycle(db: AsyncSession = Depends(database.get_db),
                    user: models.User = Depends(get_effective_user),
                    cycle_id: Optional[int] = None,
                    application_id: Optional[int] = None,
                    classification: Optional[Classification] = None,
                    selection: Optional[str] = Query(None, description="Select specific attributes like: model_name,model_metadata")):
    try:
        stmt = (
            select(models.CycleData)
            .join(models.Application)
            .join(models.Motor)
        )
        if not user.isAdmin():
            stmt = stmt.where(models.Motor.users.any(models.User.id == user.id))  # only allow if user has access to motor

        if cycle_id:
            stmt = stmt.where(models.CycleData.id == cycle_id)
        if application_id:
            stmt = stmt.where(models.CycleData.application_id == application_id)
        if classification:
            stmt = stmt.where(models.CycleData.classification == classification)

        db_data = (await db.scalars(stmt)).all()
        fields = selection.replace(" ", "").split(",") if selection else None
        return [cycle.db_model_to_json(fields) for cycle in db_data]

    except Exception:
        await db.rollback()  # close transaction
        logger.exception("Error fetching cycle data")
        raise


@router.delete('/cycle/{cycle_id}', status_code=http_status.HTTP_204_NO_CONTENT)
async def delete_cycle(cycle_id: int,
                       db: AsyncSession = Depends(database.get_db),
                       user: models.User = Depends(get_effective_user)):
    try:
        stmt = (
            select(models.CycleData)
            .join(models.Application)
            .join(models.Motor)
            .where(models.CycleData.id == cycle_id)
        )
        if not user.isAdmin():
            stmt = stmt.where(models.Motor.users.any(models.User.id == user.id))  # only allow if user has access to motor

        db_cycle = (await db.scalars(stmt)).first()
        if db_cycle is None:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Cycle not found")

        await db.delete(db_cycle)
        await db.commit()
    except Exception:
        await db.rollback()  # close transaction
        logger.exception("Error deleting cycle")
        raise


@router.delete('/run/{run_id}', status_code=http_status.HTTP_204_NO_CONTENT)
async def delete_run(run_id: str,
                     db: AsyncSession = Depends(database.get_db),
                     user: models.User = Depends(get_effective_user)):
    try:
        try:
            UUID(run_id)
        except ValueError:
            raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="Run id invalid")

        stmt = select(models.Run).where(models.Run.id == run_id)
        if not user.isAdmin():
            stmt = stmt.where(models.Run.user_id == user.id)  # only allow if user is owner of the run

        db_run = (await db.scalars(stmt)).one_or_none()
        if db_run is None:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Run not found")

        # If run is a training run, delete all associated models, predictions and prediction-runs
        dependent_runs = await db.execute(
            select(models.Run).where(
                models.Run.task == models.Run.Task.PREDICTION,
                models.Run.state_metadata['used_train_run_id'].astext == str(run_id)
            )
        )
        for pred_run in dependent_runs.scalars():
            await db.delete(pred_run)

        await db.delete(db_run)
        await db.commit()
    except Exception:
        await db.rollback()  # close transaction
        logger.exception("Error deleting run")
        raise


@router.get('/run/{run_id}')
async def get_run(run_id: str,
                  user: models.User = Depends(get_effective_user),
                  db: AsyncSession = Depends(database.get_db)):
    stmt = (
        select(models.Run)
        .where(models.Run.id == run_id)
        .options(selectinload(models.Run.application).selectinload(models.Application.motor))
    )
    if not user.isAdmin():
        stmt = (
            stmt
            .join(models.Application)
            .join(models.Motor)
            .join(models.motor_user)
            .where(models.motor_user.c.user_id == user.id)
        )  # only allow if user has access to motor
    run = (await db.scalars(stmt)).one_or_none()
    if not run:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="Run not found")

    return run.db_model_to_json()


@router.post('/preview', responses=get_swagger_responses(PreviewResponse().example()))
async def preview(
        file: UploadFile = None,
        application_id: int = None,
        db: AsyncSession = Depends(database.get_db),
        user: models.User = Depends(get_effective_user),
        context: dict = Depends(Utils.prepareBaseContext)):

    tskParam = await validate_and_prepare_task(db, user.id, TskParam.TaskName.PREVIEW, file, application_id)
    if file:
        tskParam.write_file(file)
    else:
        tskParam.application_id = int(application_id)

    result = await Utils.previewSignals(user, context, db, tskParam=tskParam, plots=False)
    return make_json_serializable(result)


@router.post('/train', responses=get_swagger_responses({"run_id": "00000000-0000-0000-0000-000000000000"}))
async def train(background_tasks: BackgroundTasks,
                file: UploadFile = None,
                application_id: int = None,
                filter_used: bool = Query(False, description="Filter used cycles"),
                user: models.User = Depends(get_effective_user),
                db: AsyncSession = Depends(database.get_db)):
    tskParam = await validate_and_prepare_task(db, user.id, TskParam.TaskName.TRAIN, file, application_id)
    if filter_used:
        tskParam.filters.append("used")
    tskParam.dataset_name = f'auto_train_{application_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}' if application_id else f'auto_train_{file.filename}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # save the tskParam on db
    await Utils.update(UpdateMessage(status=RunState.IDLE, task=RunTask.TRAIN), tskParam)

    # run training in background
    background_tasks.add_task(Utils.tsk_train, tskParam)
    return {
        "run_id": tskParam.run_id,
    }


@router.post('/predict', responses=get_swagger_responses({"run_id": "00000000-0000-0000-0000-000000000000"}))
async def predict(background_tasks: BackgroundTasks,
                  file: UploadFile = None,
                  application_id: int = None,
                  filter_used: bool = Query(False, description="Filter used cycles"),
                  train_run_id: str = Query(None, description="Selection of a specific train-run (see model ➔ run_id)"),
                  user: models.User = Depends(get_effective_user),
                  db: AsyncSession = Depends(database.get_db)):
    tskParam = await validate_and_prepare_task(db, user.id, TskParam.TaskName.PREDICT, file, application_id)
    if filter_used:
        tskParam.filters.append("used")
    tskParam.train_run_id = train_run_id
    tskParam.dataset_name = f'auto_pred_{application_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}' if application_id else f'auto_pred_{file.filename}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    # save the tskParam on db
    await Utils.update(UpdateMessage(status=RunState.IDLE, task=RunTask.PREDICTION), tskParam)

    # run prediction in background
    background_tasks.add_task(Utils.tsk_predict, tskParam)

    return {
        "run_id": tskParam.run_id,
    }


@router.get('/report/{run_id}')
async def get_report(run_id: str,
                     user: models.User = Depends(get_effective_user),
                     db: AsyncSession = Depends(database.get_db)):
    return JSONResponse(await ReportService.get_report_data(run_id=run_id, user=user, db=db))


@router.post('/import')
async def importer(
        file: UploadFile,
        add_seen_cycles: bool = Query(False),
        db: AsyncSession = Depends(database.get_db),
        user: Optional[models.User] = Depends(get_effective_user)):
    try:
        content = await file.read()
        result = await Utils.import_file_zip(user, content, db, add_seen_cycles)
        return JSONResponse(content=result)
    except Exception:
        await db.rollback()  # close transaction
        logger.exception("Error during import")
        raise HTTPException(status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Error during import")


@router.put('/validate_mqtt')
async def validate_mqtt(json_payload: str):
    result = MqttValidator.validate_mqtt_json(json_payload)
    if result is True:
        return {"status": "valid"}
    else:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail={"status": "invalid", "errors": result})


@router.put('/run/{run_id}/{disabled}')
async def set_run_state(run_id: str,
                        disabled: bool,
                        context: dict = Depends(Utils.prepareBaseContext),
                        db: AsyncSession = Depends(database.get_db),
                        user: models.User = Depends(get_effective_user)):
    try:
        try:
            UUID(run_id)
        except ValueError:
            raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="Run id invalid")

        stmt = (
            select(models.Run)
            .join(models.Application)
            .join(models.Motor)
            .where(models.Run.id == run_id)
        )
        if not user.isAdmin():
            stmt = stmt.where(models.Motor.users.any(models.User.id == user.id))  # only allow if user has access to motor

        run = (await db.scalars(stmt)).first()
        if not run:
            raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="Run not found")
        run.disabled = disabled
        await db.commit()

        btn_class = "btn-outline-secondary" if disabled else "btn-outline-success"
        icon_class = "bi-slash-circle" if disabled else "bi-check-circle"
        title = context["disable"] if not disabled else context["enable"]
        return HTMLResponse(f"""<button type="button"
                            hx-put="/api/run/{run.id}/{not disabled}"
                            hx-target="this"
                            hx-swap="outerHTML"
                            class="btn btn-sm {btn_class}"
                            title="{title}">
                            <i class="bi {icon_class}"></i></button>""")
    except Exception:
        await db.rollback()  # close transaction
        logger.exception("Error updating run state")
        raise


@router.put('/cycle/{cycle_id}/{disabled}', status_code=http_status.HTTP_200_OK)
async def set_cycle_state(cycle_id: int,
                          disabled: bool,
                          db: AsyncSession = Depends(database.get_db),
                          user: models.User = Depends(get_effective_user)):
    stmt = (
        select(models.CycleData)
        .where(models.CycleData.id == cycle_id)
    )
    if not user.isAdmin():
        stmt = stmt.join(models.Application).join(models.Motor).where(models.Motor.users.any(models.User.id == user.id))  # only allow if user has access to motor

    cycle = (await db.scalars(stmt)).one_or_none()
    if not cycle:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Cycle not found")
    cycle.disabled = disabled
    await db.commit()


async def validate_and_prepare_task(db: AsyncSession,
                                    user_id: int,
                                    task: TskParam.TaskName,
                                    file: Optional[UploadFile] = None,
                                    application_id: Optional[int] = None) -> TskParam:
    if bool(file) == bool(application_id):
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="Exactly one of 'file' or 'application_id' is required.")

    task_param = TskParam(user_id, task_for_filename=task)
    if file:
        task_param.write_file(file)
        return task_param

    # Validate if user is allowed to use this application
    user_authorized = exists(
        select(1)
        .select_from(models.User)
        .outerjoin(models.motor_user, models.motor_user.c.user_id == models.User.id)
        .where(
            models.User.id == user_id,
            ~models.User.disabled,
            or_(
                models.User.role == UserRole.ADMIN,
                models.motor_user.c.motor_id == models.Motor.id,
            ),
        )
    )

    stmt = (
        select(models.Application)
        .join(models.Motor)
        .where(
            models.Application.id == application_id,
            ~models.Motor.disabled,
            ~models.Application.disabled,
            user_authorized,
        )
    )
    application = await db.scalar(stmt)
    if not application:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Application not found")

    task_param.application_id = application_id
    return task_param


@router.get('/rules/{application_id}')
async def get_rules(application_id: int,
                    db: AsyncSession = Depends(database.get_db),
                    user: models.User = Depends(get_effective_user)):
    try:
        stmt = (
            select(models.Application)
            .join(models.Motor)
            .where(
                models.Application.id == application_id,
                models.Motor.users.any(models.User.id == user.id) if not user.isAdmin() else True
            )
        )
        application = await db.scalar(stmt)
        if not application:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Application not found or access denied")

        rules = rule_engine.get_rules(application_id)
        if not rules:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="No rules configured for this application")
        return JSONResponse(content=rules)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error getting rules for app_id %d", application_id)
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get rules")


@router.put('/rules/{application_id}')
async def update_rules(application_id: int,
                       rules: dict = Body(...),
                       db: AsyncSession = Depends(database.get_db),
                       user: models.User = Depends(get_effective_user)):
    try:
        stmt = (
            select(models.Application)
            .join(models.Motor)
            .where(
                models.Application.id == application_id,
                models.Motor.users.any(models.User.id == user.id) if not user.isAdmin() else True
            )
        )
        application = await db.scalar(stmt)
        if not application:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Application not found or access denied")

        # Calc cycle duration if not exists
        if not rules.get("action_delay_time_seconds"):
            stmt = (
                select(
                    func.min(models.MeasuringPoint.timestamp),
                    func.max(models.MeasuringPoint.timestamp),
                )
                .join(models.Application)
                .where(models.Application.id == application_id)
                .group_by(models.MeasuringPoint.cycle_id)
                .order_by(models.MeasuringPoint.cycle_id)
                .limit(1)
            )
            time_min, time_max = (await db.execute(stmt)).one_or_none()
            duration_seconds = (time_max - time_min).total_seconds() if time_min and time_max else 0
            rules["action_delay_time_seconds"] = math.ceil(duration_seconds*2)  # double cycle duration for safety

        try:
            rule_engine.write_rules(application_id, rules, user.id)
        except ValueError as e:
            raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=f"Invalid rules structure: {str(e)}")
        return JSONResponse(content={"status": "success", "message": "Rules updated successfully"}, status_code=http_status.HTTP_200_OK)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error updating rules for app_id %d", application_id)
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update rules")
