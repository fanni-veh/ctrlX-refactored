
from collections import defaultdict
from datetime import datetime
import html
import logging
from pathlib import Path
from statistics import mean
from typing import DefaultDict, List, Optional, Tuple
from fastapi import BackgroundTasks, Cookie, File, Form, HTTPException, Depends, APIRouter, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import numpy as np
from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
import yaml
from app import Classification, MeasurementMetric, RunState, RunTask, database, models
from app.mqtt.rule_engine import rule_engine
from app.scripts.auth import get_effective_user
from app.scripts.data_visualization import create_health_history_plot
from app.scripts.database_helper import load_measurements_by_app
from app.scripts.report_service import ReportService
from app.scripts.tsa_logging import create_logger
from app.scripts.tsk_param import TskParam
from app.update_message import UpdateMessage
from app.utils import Utils
from sqlalchemy.orm import selectinload, with_loader_criteria
import time


class MuteHtmxFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/htmx") == -1


router = APIRouter(prefix='/htmx', tags=['Htmx'])
templates = Jinja2Templates(directory="app/templates")

# Suspress access logs for this controller
logging.getLogger("uvicorn.access").addFilter(MuteHtmxFilter())
logger = create_logger("htmx_logger", output_file="mind_api")


@router.get("/training_progress", response_class=HTMLResponse)
async def update_table(user: Optional[models.User] = Depends(get_effective_user), context: dict = Depends(Utils.prepareBaseContext), db: AsyncSession = Depends(database.get_db)):
    stmt = (
        select(models.Run, models.Run.time_elapsed)
        .join(models.Run.application)
        .join(models.Application.motor)
        .where(models.Run.task == RunTask.TRAIN,
               models.Run.state == RunState.RUNNING,
               models.Run.user_id == user.id if not user.isAdmin() else True)
        .options(
            selectinload(models.Run.application)
            .selectinload(models.Application.motor)
        )
    )
    active_trainings: List[Tuple[models.Run, float]] = (await db.execute(stmt)).all()
    data = []
    for run, elapsed in active_trainings:
        motor = getattr(run.application, "motor", None)
        if not motor:
            continue

        step = getattr(run.state_metadata, "get", lambda x, d=None: d)("data", {}).get("step", 0)
        progress = round((step / 4) * 100)

        seconds_total = round(elapsed)
        minutes, seconds = divmod(seconds_total, 60)
        formatted_time = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

        data.append({
            "machine": motor.machine,
            "part": motor.part,
            "run_id": run.id,
            "serial": motor.serial,
            "application_id": run.application.id,
            "application": run.application.context_code,
            "run_since_seconds": seconds_total,
            "formatted_time": formatted_time,
            "progress": progress,
        })

    return templates.TemplateResponse("partials/_training_progress_tbody.html", {**context, "data": data})


@router.get("/my_models", response_class=HTMLResponse)
async def update_my_models(user: Optional[models.User] = Depends(get_effective_user), context: dict = Depends(Utils.prepareBaseContext), db: AsyncSession = Depends(database.get_db)):
    stmt = (
        select(models.Application)
        .join(models.Run)
        .join(models.Motor)
        .join(models.Model)
        .where(
            ~models.Application.disabled,
            ~models.Motor.disabled
        )
        .options(
            selectinload(models.Application.runs).selectinload(models.Run.models),
            with_loader_criteria(
                models.Run,
                (models.Run.task == RunTask.TRAIN) &
                (models.Run.state == RunState.SUCCESS) &
                ~models.Run.disabled,
                include_aliases=True
            )
        )
        .group_by(models.Application.id)
        .order_by(models.Application.context_code)
    )

    if not user.isAdmin():
        stmt = (
            stmt.join(models.motor_user)
            .where(models.motor_user.c.user_id == user.id)
        )

    # Execute the query and get all columns
    rows = (await db.execute(stmt)).scalars().all()
    data = [
        {
            "application_context_code": app.context_code,
            "model_name": run.dataset_name,
            "application_id": app.id,
            "model_count": len(run.models),
            "run_id": run.id
        }
        for app in rows
        for run in sorted(app.runs, key=lambda r: r.time_last_activity)
    ]

    return templates.TemplateResponse("partials/_my_models_tbody.html", {**context, "data": data})


@router.get("/drive_systems", response_class=HTMLResponse)
async def update_drive_systems(user: Optional[models.User] = Depends(get_effective_user), context: dict = Depends(Utils.prepareBaseContext), db: AsyncSession = Depends(database.get_db)):
    start = time.perf_counter()
    motor_conditions = await Utils.calculate_motor_conditions(user, db)

    data = []
    # Iterate over sorted motor_conditions by motor_id, newest first
    for motor_id, motor_condition in sorted(motor_conditions.items(), key=lambda item: item[0], reverse=True):
        motor_health_score_mean = motor_condition.get('motor_health_score_mean')
        latest_prediction_time = motor_condition.get('latest_prediction_time')
        machine_health_color, machine_health_text = Utils.get_color(motor_health_score_mean, context)

        data.append({
            "id": motor_id,
            "machine": motor_condition.get('machine'),
            "serial_number": motor_condition.get('serial'),
            "part": motor_condition.get('part'),
            "motor_health_score": motor_health_score_mean,
            "motor_health_text": machine_health_text,
            "motor_health_color": machine_health_color,
            "last_prediction_time": Utils.strftime(datetime.fromisoformat(latest_prediction_time)),
        })

    logger.debug("Time taken for load drive-systems: %.2f s", time.perf_counter()-start)
    return templates.TemplateResponse("partials/_my_drive_systems_accordion.html", {**context, "data": data})


@router.get("/prediction_runs", response_class=HTMLResponse)
async def prediction_runs(mot_id: int, user: Optional[models.User] = Depends(get_effective_user), context: dict = Depends(Utils.prepareBaseContext), db: AsyncSession = Depends(database.get_db)):
    start = time.perf_counter()
    motors = await Utils.calculate_motor_conditions(user, db, motor_ids=[mot_id])
    logger.debug("Time taken for load prediction_runs: %.2f s", time.perf_counter()-start)
    if not motors:
        # not motor found, send 286 to stop htmx-polling
        return HTMLResponse('', status_code=286)
    # extract the single motor condition
    motor = motors.get(mot_id)

    # Process results more efficiently
    start = time.perf_counter()
    data = []
    applications = motor.get("applications", {})
    # Iterate over sorted applications by app_id, newest first
    for app_id, app_condition in sorted(applications.items(), key=lambda item: item[0], reverse=True):
        if not app_condition:
            # skip not evaluated applications
            continue

        # load used and unused cycles
        stmt = (
            select(models.CycleData.id)
            .where(
                ~models.CycleData.disabled,
                models.CycleData.classification == Classification.UNKNOWN,
                models.CycleData.application_id == app_id)
        )
        total_cycles = (await db.execute(stmt)).scalars().all()

        # load all used cycles in successful prediction runs
        stmt = (
            select(models.cycle_run.c.cycle_id)
            .join(models.Run)
            .where(
                ~models.Run.disabled,
                models.Run.task == RunTask.PREDICTION,
                models.Run.state == RunState.SUCCESS,
                models.Run.application_id == app_id)
        )
        used_cycles = (await db.execute(stmt)).scalars().all()
        unused_cycles = set(total_cycles) - set(used_cycles)

        # load newest run to get the state
        stmt = (
            select(
                models.Run,
                func.count(models.cycle_run.c.cycle_id).label("cycle_count")
            )
            .outerjoin(models.cycle_run, models.cycle_run.c.run_id == models.Run.id)
            .where(
                ~models.Run.disabled,
                models.Run.task == models.Run.Task.PREDICTION,
                models.Run.state == models.Run.State.SUCCESS,
                models.Run.application_id == app_id
            )
            .group_by(models.Run.id)
            .order_by(desc(models.Run.time_last_activity))
            .limit(1)
        )
        current_run, cycles_count = (await db.execute(stmt)).one()

        newest_prediction = max(
            app_condition.get("predictions", []),
            key=lambda p: datetime.fromisoformat(p["time_created"]),
            default=None
        )
        score = newest_prediction["health_score"] if newest_prediction else 0
        prediction_in_progress = True if current_run.state == RunState.RUNNING else False

        data.append({
            "id": app_id,
            "context_code": app_condition.get('context_code'),
            "cycles": cycles_count,
            "score": score,
            "score_color": Utils.get_color(score, context)[0],
            "time": Utils.strftime(datetime.fromisoformat(newest_prediction["time_last_activity"])),
            "unused_cycles": len(unused_cycles),
            "train_in_progress": prediction_in_progress,
            "run_id": newest_prediction["run_id"]
        })

    logger.debug("Time taken for process prediction_runs: %.2f s", time.perf_counter()-start)
    return templates.TemplateResponse("partials/_my_drive_system_pred_run.html", {**context, "data": data})


@router.get("/validate-predict", response_class=HTMLResponse)
async def validate_prediction(dataset_name: str,
                              cycles_count_unknown: int,
                              application_id: str = None,
                              train_app_id: str = None,
                              context: dict = Depends(Utils.prepareBaseContext),
                              run_id: str = Cookie(None),
                              db: AsyncSession = Depends(database.get_db)):
    if not dataset_name:
        return HTMLResponse(f'''<div class="text-danger">{html.escape(context['name_should_not_empty'])}</div>''')
    if cycles_count_unknown < 5:
        return HTMLResponse(f'''<div class="text-danger">{html.escape(context['too_few_predict_cycles'])}</div>''')

    application_id = int(application_id) if application_id and application_id.isdigit() else None
    if not (application_id or train_app_id):
        return HTMLResponse(f'''<div class="text-danger">{html.escape(context['application_is_unknown'])}</div>''')

    stmt = select(models.Run).where(
        models.Run.application_id == application_id,
        models.Run.dataset_name == dataset_name,
        models.Run.task == RunTask.PREDICTION
    )
    if run_id:
        # If run_id is already set, the user want to re-try a prediction
        stmt = stmt.where(models.Run.id == run_id)

    result = (await db.execute(stmt)).scalars().all()
    # Return validation-error if dataset already exists in a non-error run
    if any(result) and (not run_id and result[0].state != RunState.ERROR):
        return HTMLResponse(f'''<div class="text-danger">{html.escape(context['name_already_exist'])}</div>''')

    # check if any train-run exists
    train_app_id = int(train_app_id) if train_app_id and train_app_id.isdigit() else None
    if train_app_id:
        stmt = select(models.Run).where(
            models.Run.application_id == train_app_id,
            models.Run.state == RunState.SUCCESS,
            models.Run.task == RunTask.TRAIN,
            ~models.Run.disabled
        )
    else:
        stmt = select(models.Run).where(
            models.Run.application_id == application_id,
            models.Run.state == RunState.SUCCESS,
            models.Run.task == RunTask.TRAIN,
            ~models.Run.disabled
        )
    result = (await db.execute(stmt)).scalars().first()
    if not result:
        return HTMLResponse(f'''<div class="text-danger">{html.escape(context['no_models_found'])}</div>''')

    return HTMLResponse('''<div class="text-success"></div>''')


@router.get("/validate-train", response_class=HTMLResponse)
async def validate_train(dataset_name: str,
                         cycles_count_good: int,
                         cycles_count_bad: int,
                         application_id: str = None,
                         context: dict = Depends(Utils.prepareBaseContext),
                         db: AsyncSession = Depends(database.get_db)):
    if not dataset_name:
        return HTMLResponse(f'''<div class="text-danger">{html.escape(context['name_should_not_empty'])}</div>''')
    if cycles_count_bad < 5 or cycles_count_good < 5:
        return HTMLResponse(f'''<div class="text-danger">{html.escape(context['too_few_train_cycles'])}</div>''')

    application_id = int(application_id) if application_id and application_id.isdigit() else None
    if application_id:
        stmt = select(models.Run).where(
            models.Run.application_id == application_id,
            models.Run.dataset_name == dataset_name,
            models.Run.task == RunTask.TRAIN
        )
        result = await db.execute(stmt)
        if any(result.scalars().all()):
            return HTMLResponse(f'''<div class="text-danger">{html.escape(context['name_already_exist'])}</div>''')
    return HTMLResponse('''<div class="text-success"></div>''')


@router.get("/set-language/{lang}", response_class=HTMLResponse)
async def set_language(request: Request, current_url: str = None, context: dict = Depends(Utils.prepareBaseContext)):
    response = RedirectResponse(url=current_url or request.base_url)
    response.set_cookie(key="lang", value=context['lang'])
    return response


@router.get('/login')
def login(context: dict = Depends(Utils.prepareBaseContext)):
    return templates.TemplateResponse("partials/_login.html", context)


@router.post('/import')
async def import_csv(
        file: UploadFile = File(...),
        add_seen_cycles: bool = Form(False),
        context: dict = Depends(Utils.prepareBaseContext),
        user: Optional[models.User] = Depends(get_effective_user),
        db: AsyncSession = Depends(database.get_db)):
    content = await file.read()
    result = await Utils.import_file_zip(user, content, db, add_seen_cycles)
    context["filename"] = file.filename
    context["motors"] = result['motors']

    return templates.TemplateResponse("partials/_import_result.html", context)


@router.post('/train/preview')
async def train_preview(
        file: UploadFile = File(None),
        db_application: int = Form(False),
        context: dict = Depends(Utils.prepareBaseContext),
        user: Optional[models.User] = Depends(get_effective_user),
        db: AsyncSession = Depends(database.get_db)):

    tskParam = TskParam(user.id, TskParam.TaskName.PREVIEW)
    try:
        if file:
            tskParam.write_file(file)
        elif db_application:
            tskParam.application_id = int(db_application)
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Data missing")

        result = await Utils.previewSignals(user, context=context, db=db, tskParam=tskParam, plots=True, kind=RunTask.TRAIN)
        context['data'] = result
        if user.isAdmin():
            config_file = Path("model_config.yaml")
            config = config_file.read_text(encoding='utf-8') if config_file.exists() else ""
            context['model_config'] = config
        response = templates.TemplateResponse("partials/train/_step2.html", context)
        response.set_cookie(key="run_id", value=tskParam.run_id)
        if file:
            response.set_cookie(key="uploaded_file", value=file.filename)
        return response
    except Exception as err:
        tskParam.logger.exception("Error in train_preview")
        context['data'] = {'error_message': err}
        return templates.TemplateResponse("partials/_error.html", context)


@router.post('/predict/preview')
async def predict_preview(file: UploadFile = File(None),
                          db_application: int = Form(None),
                          train_app_id: int = Form(None),
                          context: dict = Depends(Utils.prepareBaseContext),
                          user: Optional[models.User] = Depends(get_effective_user),
                          db: AsyncSession = Depends(database.get_db)):
    try:
        tskParam = TskParam(user.id, TskParam.TaskName.PREVIEW)
        if file:
            tskParam.write_file(file)
        elif db_application:
            tskParam.application_id = int(db_application)
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Data missing")

        result = await Utils.previewSignals(user, context=context, db=db, tskParam=tskParam, plots=True, kind=RunTask.PREDICTION)

        # Get available train-runs
        app_id = train_app_id or result.application.get('id')
        if app_id:
            stmt = (
                select(models.Run)
                .join(models.Application)
                .join(models.Motor)
                .where(models.Run.application_id == app_id,
                       models.Run.task == RunTask.TRAIN,
                       models.Run.state == RunState.SUCCESS,
                       ~models.Run.disabled,
                       ~models.Application.disabled,
                       ~models.Motor.disabled)
                .options(
                    selectinload(models.Run.application),
                    selectinload(models.Run.models))
                .order_by(models.Run.time_last_activity.desc())
            )
            if not user.isAdmin():
                stmt = (
                    stmt.join(models.motor_user)
                    .where(models.motor_user.c.user_id == user.id)
                )
            runs = (await db.execute(stmt)).scalars().all()
            if runs:
                if train_app_id and train_app_id != result.application.get('id'):
                    app = {
                        "app_override": train_app_id != result.application.get('id'),
                        "id": app_id,
                        "context_code": runs[0].application.context_code,
                    }
                    context['train_app'] = app

                run_data = [
                    {
                        "id": run.id,
                        "datasetname": run.dataset_name,
                        "time": Utils.strftime(run.time_last_activity),
                        "count_models": sum(1 for model in run.models if not model.disabled),
                        "accuracy": f"{round(mean([model.model_metadata['scores']['accuracy'] for model in run.models if not model.disabled]) * 100, 2)}%"
                    } for run in runs
                ]
                context['train_runs'] = run_data

        context['data'] = result
        response = templates.TemplateResponse("partials/predict/_step2.html", context)
        response.set_cookie(key="run_id", value=tskParam.run_id)
        if file:
            response.set_cookie(key="uploaded_file", value=file.filename)
        return response
    except Exception as err:
        context['data'] = {'error_message': err}
        return templates.TemplateResponse("partials/_error.html", context)


@router.post('/train/run')
async def train_run(background_tasks: BackgroundTasks,
                    good_cycles: List[str] = Form(...),
                    bad_cycles: List[str] = Form(...),
                    cycles_to_remove: List[str] = Form(...),
                    dataset_name: str = Form(...),
                    application_id: Optional[int] = Form(None),
                    config: Optional[str] = Form(None),
                    run_id: str = Cookie(None),
                    uploaded_file: str = Cookie(None),
                    context: dict = Depends(Utils.prepareBaseContext),
                    user: models.User = Depends(get_effective_user),
                    db: AsyncSession = Depends(database.get_db)):
    try:
        tskParam = TskParam(user.id, TskParam.TaskName.TRAIN, run_id)

        if uploaded_file:
            tskParam.read_file(uploaded_file)
        elif application_id:
            tskParam.application_id = int(application_id)
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No input data found")

        if cycles_to_remove:
            # Disable the cycles deselected by the user - they will never show up in the UI again.
            stmt = (
                update(models.CycleData)
                .where(models.CycleData.id.in_(map(int, cycles_to_remove)))
                .values(disabled=True)
            )
            await db.execute(stmt)
            await db.commit()
            logger.info(f"Disabled cycles with ids: {', '.join(cycles_to_remove)} by user_id: {user.id}")

        tskParam.using_cycle_ids = set(map(int, good_cycles + bad_cycles))
        tskParam.dataset_name = dataset_name

        if user.isAdmin() and config:
            tskParam.pipeline_config = yaml.safe_load(config)  # just to check if yaml is valid

        # save the tskParam on db
        await Utils.update(UpdateMessage(status=RunState.RUNNING, task=RunTask.TRAIN, metadata={'step': 0}), tskParam)

        background_tasks.add_task(Utils.tsk_train, tskParam)

        context['data'] = {'run_id': tskParam.run_id}
        return templates.TemplateResponse("partials/train/_step3.html", context)

    except Exception as e:
        await Utils.update(UpdateMessage(status=RunState.ERROR, task=RunTask.TRAIN, metadata={'message': str(e)}), tskParam)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid data: {str(e)}")


@router.post('/predict/run')
async def predict_run(background_tasks: BackgroundTasks,
                      unknown_cycles: List[str] = Form(...),
                      cycles_to_remove: List[str] = Form(...),
                      dataset_name: str = Form(...),
                      train_run_id: str = Form(...),
                      application_id: Optional[int] = Form(None),
                      run_id: str = Cookie(None),
                      uploaded_file: str = Cookie(None),
                      context: dict = Depends(Utils.prepareBaseContext),
                      user: models.User = Depends(get_effective_user),
                      db: AsyncSession = Depends(database.get_db)):
    try:
        tskParam = TskParam(user.id, TskParam.TaskName.PREDICT, run_id)
        tskParam.train_run_id = train_run_id

        if uploaded_file:
            tskParam.read_file(uploaded_file)
        elif application_id:
            tskParam.application_id = int(application_id)
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No input data found")

        if cycles_to_remove:
            # Disable the cycles deselected by the user - they will never show up in the UI again.
            stmt = (
                update(models.CycleData)
                .where(models.CycleData.id.in_(map(int, cycles_to_remove)))
                .values(disabled=True)
            )
            await db.execute(stmt)
            await db.commit()
            logger.info(f"Disabled cycles with ids: {', '.join(cycles_to_remove)} by user_id: {user.id}")

        tskParam.using_cycle_ids = set(map(int, unknown_cycles))
        tskParam.dataset_name = dataset_name

        # save the tskParam on db
        await Utils.update(UpdateMessage(status=RunState.RUNNING, task=RunTask.PREDICTION, metadata={'step': 0}), tskParam)

        background_tasks.add_task(Utils.tsk_predict, tskParam)

        context['data'] = {'run_id': tskParam.run_id}
        return templates.TemplateResponse("partials/predict/_step3.html", context)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid data: {str(e)}")


@router.get('/rules/{application_id}', response_class=HTMLResponse)
async def get_rules_editor(application_id: int,
                           user: models.User = Depends(get_effective_user),
                           context: dict = Depends(Utils.prepareBaseContext),
                           db: AsyncSession = Depends(database.get_db)):
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
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found or access denied")

        rules = rule_engine.get_rules(application_id)
        # Convert triggers from array format [{type: value}] to dict format {type: value}
        for action_type in ["train", "prediction"]:
            if action_type in rules.get("actions", {}):
                for rule in rules["actions"][action_type]:
                    triggers_list = rule.get("triggers", [])
                    # Convert array to dict
                    triggers_dict = {}
                    for trigger in triggers_list:
                        trigger_type = list(trigger.keys())[0]
                        triggers_dict[trigger_type] = trigger[trigger_type]
                    rule["triggers"] = triggers_dict

        context['data'] = {
            "application_id": application_id,
            "rules": rules
        }

        return templates.TemplateResponse("partials/rules/_editor.html", context)
    except Exception as e:
        logger.exception("Error loading rules editor for app_id %d", application_id)
        return HTMLResponse(f'<div class="alert alert-danger">Failed to load rules: {str(e)}</div>')


@router.post('/rules/{application_id}/{action_type}/add-rule', response_class=HTMLResponse)
async def add_rule(application_id: int,
                   action_type: str,
                   context: dict = Depends(Utils.prepareBaseContext)):
    """Add a new rule card - just renders the template, no persistence."""
    try:
        # Generate unique ID for the rule (using timestamp in nanoseconds)
        unique_id = int(time.time() * 1000000)  # Microseconds for uniqueness

        context['data'] = {
            "app_id": application_id,
            "action_type": action_type,
            "rule_index": unique_id,  # Use unique ID instead of counting
            "rule": {
                "name": f"New {action_type} rule",
                "logic": "AND",
                "filter_used_cycles": True,
                "triggers": {}
            }
        }
        return templates.TemplateResponse("partials/rules/_rule_card.html", context)
    except Exception as e:
        logger.exception("Error adding rule")
        return HTMLResponse(f'<div class="alert alert-danger">Failed to add rule: {str(e)}</div>')


@router.get('/report/{run_id}')
async def get_train_result(run_id: str,
                           user: Optional[models.User] = Depends(get_effective_user),
                           context: dict = Depends(Utils.prepareBaseContext),
                           db: AsyncSession = Depends(database.get_db)):
    results = await ReportService.get_report_data(run_id=run_id, user=user, db=db, context=context)
    context['data'] = results
    return templates.TemplateResponse("partials/_report.html", context)


@router.get('/analytics')
async def analytics_accordion(user: Optional[models.User] = Depends(get_effective_user),
                              context: dict = Depends(Utils.prepareBaseContext),
                              db: AsyncSession = Depends(database.get_db)):
    start_full = time.perf_counter()
    stmt = (
        select(
            models.Application,
            func.count(func.distinct(models.CycleData.id).label("cycledata_count")),
            func.max(models.CycleData.time_created).label("latest_cycledata")
        )
        .join(models.Motor)
        .outerjoin(models.Run, models.Run.application_id == models.Application.id)
        .outerjoin(models.CycleData, models.CycleData.application_id == models.Application.id)
        .options(
            selectinload(models.Application.motor),
            selectinload(models.Application.runs).selectinload(models.Run.cycles).load_only(models.CycleData.id)
        )
        .group_by(models.Application.id)
    )

    if not user.isAdmin():
        stmt = (
            stmt.join(models.motor_user)
            .where(models.motor_user.c.user_id == user.id)
        )

    rows = (await db.execute(stmt)).fetchall()
    logger.debug("Time taken for load analytics from db: %.2f s", time.perf_counter()-start_full)

    applications_by_motor: DefaultDict[models.Motor, list[tuple[models.Application, int, datetime]]] = defaultdict(list)
    for app, cycledata_count, latest_cycledata in rows:
        applications_by_motor[app.motor].append((app, cycledata_count, latest_cycledata))

    start = time.perf_counter()
    results: list[dict] = []

    for motor, apps in sorted(applications_by_motor.items(), key=lambda x: x[0].id):
        sorted_apps: list[tuple[models.Application, int, datetime]] = sorted(apps, key=lambda x: x[0].id)

        motor_dict = {
            "id": motor.id,
            "serial_number": motor.serial,
            "part": motor.part,
            "machine": motor.machine,
            "client": motor.client,
            "time_created": Utils.strftime(motor.time_created),
            "plots": {"health_history": _get_health_history_plot(motor, [x[0] for x in sorted_apps], context)},
            "applications": []
        }

        # Process applications per motor, sorted by application id
        for app, cycledata_count, latest_cycledata in sorted_apps:
            sorted_runs: list[models.Run] = sorted(app.runs, key=lambda r: r.id, reverse=True)
            app_dict = {
                "id": app.id,
                "context_code": app.context_code,
                "recipe": app.recipe,
                "time_created": Utils.strftime(app.time_created),
                "cycles_amount": cycledata_count,
                "newest_cycle": Utils.strftime(latest_cycledata),
                "runs": [
                    {
                        "id": run.id,
                        "task": run.task.value,
                        "state": run.state.value,
                        "state_display": _get_state_display(run, context),
                        "cycles_amount": _get_cycles_count(run),
                        "dataset_name": run.dataset_name,
                        "last_activity": Utils.strftime(run.time_last_activity),
                        "disabled": run.disabled,
                    }
                    for run in sorted_runs
                ]
            }
            motor_dict["applications"].append(app_dict)

        results.append(motor_dict)

    context['motors'] = results
    logger.debug("Time taken for process analytics: %.2f s", time.perf_counter()-start)
    return templates.TemplateResponse("partials/_analytics_accordion.html", context)


@router.get('/apps_from_db/{kind}')
async def db_table(kind: str,
                   app_id: int = None,
                   user: Optional[models.User] = Depends(get_effective_user),
                   context: dict = Depends(Utils.prepareBaseContext),
                   db: AsyncSession = Depends(database.get_db)):
    stmt = (
        select(
            models.Motor.id,
            models.Motor.part,
            models.Motor.serial,
            models.Motor.client,
            models.Application.id.label("application_id"),
            models.Application.context_code,
            models.Application.recipe
        )
        .join(models.Application)
        .join(models.CycleData)
        .where(~models.Motor.disabled,
               ~models.Application.disabled,
               ~models.CycleData.disabled)
        .group_by(models.Motor.id, models.Application.id)
        .order_by(models.Motor.part, models.Motor.serial, models.Application.context_code)
    )
    if kind == RunTask.TRAIN.value:
        stmt = stmt.where(models.CycleData.classification.in_([Classification.GOOD, Classification.BAD]))
    else:
        stmt = stmt.where(models.CycleData.classification == Classification.UNKNOWN)

    if not user.isAdmin():
        stmt = (
            stmt.join(models.motor_user)
            .where(models.motor_user.c.user_id == user.id)
        )

    result = (await db.execute(stmt)).mappings().all()

    applications = [
        {
            "motor_id": row.id,
            "part": row.part,
            "serial": row.serial,
            "application_id": row.application_id,
            "application_contextCode": row.context_code,
            "application_recipe": row.recipe,
        }
        for row in result if not (await load_measurements_by_app(db, row.application_id, preflight=True, metrics=[MeasurementMetric.ACT_CURRENT], logger=logger)).empty
    ]

    context["data"] = applications
    if app_id:
        # set the pre-definded application_id to filter possible data
        context["application_id"] = app_id
    return templates.TemplateResponse("partials/_select_database_data.html", context)


def _get_health_history_plot(motor: models.Motor, applications: list[models.Application], context: dict) -> Optional[str]:
    plot_data = {}
    for app_ext in applications:
        app = app_ext
        # Filter runs only once
        runs_to_process = [
            run for run in app.runs
            if run.state == RunState.SUCCESS and run.task == RunTask.PREDICTION
        ]
        if not runs_to_process:
            continue
        run_data = []
        for run in runs_to_process:
            array = np.concatenate([
                np.zeros(len(run.state_metadata.get('bad_cycles', [])), dtype=int),
                np.ones(len(run.state_metadata.get('good_cycles', [])), dtype=int)
            ])
            value = round(Utils.classifications_to_percentage(array, run.state_metadata.get('confidence_mean')), 2)
            run_data.append((run.time_created, value))

        if run_data:
            plot_data[app.context_code] = run_data

    if plot_data:
        return create_health_history_plot(
            title='',
            y_title=f"{context['health_state']} [%]",
            x_title=context['time'],
            data=plot_data
        ).to_html(full_html=False, include_plotlyjs=False, div_id=f"mot_health_{motor.id}", config={'displayModeBar': False})
    return None


def _get_state_display(run: models.Run, context: dict) -> str:
    """Get human-readable state display"""
    if run.state == RunState.SUCCESS:
        return context['successful']
    elif run.state == RunState.ERROR:
        return context['faulty']
    else:
        return run.state.value


def _get_cycles_count(run: models.Run) -> int:
    """Calculate cycle count from run"""
    cycles: list = getattr(run, 'cycles', [])
    if cycles:
        return len(cycles)

    metadata: dict = run.state_metadata
    return len(metadata.get("good_cycles", [])) + len(metadata.get("bad_cycles", []))
