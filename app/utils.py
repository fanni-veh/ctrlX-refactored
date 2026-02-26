import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import glob
import json
import logging
import os
import statistics
import time
from typing import Tuple
from uuid import UUID
from fastapi import Request, Depends
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from app import Classification, MeasurementMetric, RunTask, RunState, UserRole, database, models
from app.dto_models import PredictResponse, PreviewResponse, TrainResponse
from app.pipeline.pipeline import BaseClass, Predictor, Trainer
from app.scripts.auth import get_current_active_user
from app.scripts.data_visualization import ts_visualisation
from app.scripts.database_helper import add_cycles, load_measurements_by_app
from app.pipeline.tools.onnx_converter import sk_to_onnx
from app.scripts.preprocessing_signal import process_data
from app.multiprocessing.read_zip import read_zip
from app.scripts.tsa_logging import create_logger
from app.scripts.tsk_param import TskParam
from sqlalchemy import delete, insert, select, or_, and_, distinct, update
from sqlalchemy.orm import selectinload, contains_eager
from app.update_message import UpdateMessage
from app.config import setting
from fastapi.concurrency import run_in_threadpool


# Create a global or shared lock
update_lock = asyncio.Lock()

# Initialize the logger for this module
logger = create_logger(__name__, output_file="mind_api")


class Utils:
    _initialized = False
    _default_fallback_language = 'en'
    _languages = {}
    _app_version = 'unknown'

    def __new__(cls):
        if not cls._initialized:
            cls._initialized = True
            cls.shared_data = cls._initialize_shared_data()
        return cls

    @staticmethod
    def get_value_from_pyproject(key: str, file_path='pyproject.toml'):
        """
        Read the version from `pyproject.toml` using a simple manual parsing.
        """
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if line.strip().startswith(key+' ='):
                        return line.split('=')[1].strip().strip('"')
        except Exception:
            logger.exception("Error reading %s", file_path)
        return None

    @staticmethod
    def _initialize_shared_data():
        Utils._app_version = Utils.get_value_from_pyproject('version')
        language_list = glob.glob("app/languages/*.json")
        for lang in language_list:
            lang_code = os.path.splitext(os.path.basename(lang))[0]
            with open(lang, 'r', encoding='utf8') as file:
                Utils._languages[lang_code] = json.load(file)

    @staticmethod
    def get_color(score, context):
        if score >= 95:
            return ('limegreen', context['health_like_new'] if context else 'Like New')
        if score >= 85:
            return ('green', context['health_good'] if context else 'Good')
        if score >= 75:
            return ('orange', context['health_used'] if context else 'Used')
        if score >= 55:
            return ('darkorange', context['health_worn'] if context else 'Worn')
        if score >= 40:
            return ('orangered', context['health_warn'] if context else 'Warn')
        return ('red', context['health_critical'] if context else 'Critical')

    @staticmethod
    def classifications_to_percentage(classifications, confidence_mean, classification_weight=0.3, confidence_weight=0.7):
        """
        Convert the classifications to a score using the overall mean confidence and the classifications-distribution.

        Args:
            classifications (list): List of binary classifications (0 for bad, 1 for good)
            confidence_mean (float): The mean of the original confidence values (0-1)
            classification_weight (float): Weight to give to the classification percentage (0-1)
            confidence_weight (float): Weight to give to the confidence mean (0-1)

        Returns:
            float: User-friendly percentage (0-100)
        """
        # Return 0 if the input list is empty
        if not len(classifications):
            return 0

        # Calculate the percentage of good classifications (0-100 scale)
        percentage_good = (sum(classifications) / len(classifications)) * 100

        # Weighted combination: 70% confidence, 30% classification ratio
        weighted_score = (confidence_weight * (confidence_mean * 100)) + (classification_weight * percentage_good)

        # No restrictions based on classification distribution
        return float(weighted_score)

    @staticmethod
    def get_default_language():
        return Utils._default_fallback_language

    @staticmethod
    async def prepareBaseContext(request: Request, lang: str = None, db: AsyncSession = Depends(database.get_db)):
        context = {"request": request}
        lang = lang \
            or request.cookies.get('lang') \
            or request.headers.get("Accept-Language")[:2] if 'Accept-Language' in request.headers else Utils.get_default_language() \
            or Utils._default_fallback_language
        if lang not in Utils._languages:
            lang = Utils._default_fallback_language
        lang_data = Utils._languages.get(lang, {})

        context.update(lang_data)
        context.update({'languages': Utils._languages.keys()})
        context.update({'app_version': Utils._app_version})

        user = await get_current_active_user(request, db)
        context.update({'user_id': user.id} if user else {})
        context.update({'user_role': user.role.value} if user else {})
        context.update({'user': user} if user else {})
        timestamp = time.time()
        context.update({'timestamp': int(timestamp)})
        context.update({'time_display_long': Utils.strftime(datetime.fromtimestamp(timestamp, tz=timezone.utc))})
        context.update({'time_display_short': datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y%m%d%H%M%S")})
        return context

    @staticmethod
    async def read_zip_for_train(received_data):
        start = time.perf_counter()
        datas = read_zip(received_data)
        logger.info("Time taken for read_zip: %.4f s", time.perf_counter() - start)

        result = []
        for data in datas:
            motor_info, app_info, cycle_info, df = data

            # Skip unknown cycles if not all cycles should be predicted
            if not setting.predict_all_cycles and cycle_info['classification'] == Classification.UNKNOWN.label:
                continue

            # Filter for act_current in one operation
            if 'field' in df.columns:
                df_filtered = df[df['field'] == 'act_current']
                if not df_filtered.empty:
                    result.append((motor_info, app_info, cycle_info, df_filtered))
        return result

    @staticmethod
    def read_zip_for_test(received_data):
        start = time.perf_counter()
        datas = read_zip(received_data)
        logger.info("Time taken for read_zip: %.4f s", time.perf_counter() - start)

        result = []
        for data in datas:
            motor_info, app_info, cycle_info, df = data

            # Early exit if filtering condition not met
            if not setting.predict_all_cycles and cycle_info['classification'] != Classification.UNKNOWN.label:
                continue

            # Vectorized column operations
            if 'label' in df.columns:
                df = df.drop(columns=['label'])

            # Filter for act_current in one operation
            if 'field' in df.columns:
                df_filtered = df[df['field'] == 'act_current']
                if not df_filtered.empty:
                    result.append((motor_info, app_info, cycle_info, df_filtered))
        return result

    @staticmethod
    async def convert_csv_into_entities(datas: Tuple[dict, dict, dict, pd.DataFrame], db: AsyncSession, add_new_entities_to_db_by_user: models.User = None) -> list[models.CycleData]:
        """
        Convert the data from CSV into entities and store motor and application if new.
        If motor already exists, the user must be owner of the motor or admin.
        """

        # Get motors from data
        import_motors = {(entry[0]['serial'], entry[0]['part']) for entry in datas}
        # Prepare statement to get motors from db
        stmt = (
            select(models.Motor)
            .options(
                selectinload(models.Motor.application)
                .selectinload(models.Application.cycledatas),  # We need the cycle data to link later
                selectinload(models.Motor.users)
            )
        )
        filter = [and_(models.Motor.serial.ilike(serial), models.Motor.part.ilike(part)) for serial, part in import_motors]
        if filter:
            stmt = stmt.where(or_(*filter))

        db_motors = (await db.execute(stmt)).scalars().all()

        # Create hash map of motors and applications to check existing entities
        db_motors_hashmap = {}
        for motor in db_motors:
            mot_hash = models.Motor.get_hash(serial=motor.serial, part=motor.part)
            db_motors_hashmap[mot_hash] = motor
            motor.app_hashmap = {}
            for application in motor.application:
                motor.app_hashmap[application.get_hash(
                    motor=mot_hash,
                    context_code=str(application.context_code).lower(),
                    recipe=str(application.recipe).lower()
                )] = application

        new_motors: list[models.Motor] = []
        new_applications: list[models.Application] = []
        new_cycles: list[models.CycleData] = []
        for entry in datas:
            application = None
            serial = str(entry[0]['serial'])
            part = str(entry[0]['part'])
            context_code = str(entry[1]['context_code'])
            recipe = str(entry[1]['recipe'])

            mot_hash = models.Motor.get_hash(serial=serial, part=part)
            if mot_hash in db_motors_hashmap:
                motor = db_motors_hashmap[mot_hash]
                # check if user is priviled to use this motor
                if add_new_entities_to_db_by_user and add_new_entities_to_db_by_user not in motor.users and not add_new_entities_to_db_by_user.isAdmin():
                    raise Exception('User is not owner of this the motor.')

                app_hash = models.Application.get_hash(
                    motor=mot_hash,
                    context_code=context_code,
                    recipe=recipe)
                if app_hash in motor.app_hashmap:
                    application = motor.app_hashmap[app_hash]
            else:
                motor = models.Motor(
                    serial=serial,
                    part=part,
                    machine=entry[0].get("machine", None),  # optional field
                    client=entry[0].get("client")  # required field
                )
                db_motors_hashmap[mot_hash] = motor
                motor.app_hashmap = {}
                new_motors.append(motor)
                if add_new_entities_to_db_by_user:
                    motor.users.add(add_new_entities_to_db_by_user)
                    db.add(motor)

            if application is None:
                application = models.Application(
                    context_code=context_code,
                    recipe=recipe
                )
                app_hash = models.Application.get_hash(motor=mot_hash, context_code=context_code, recipe=recipe)
                motor.app_hashmap[app_hash] = application
                new_applications.append(application)
                if add_new_entities_to_db_by_user:
                    db.add(application)

            cycle = models.CycleData(
                testCycleCounter=entry[2].get('testcyclecounter', None),
                classification=Classification.from_str(entry[2].get("classification", '')),
                driveConfig=entry[2].get('driveconfig', None),
                cycleConfig=entry[2].get('cycleconfig', None)
            )
            cycle.dataframe = entry[3]
            application.motor = motor
            cycle.application = application
            # if add_new_entities_to_db_by_user:
            #     db.add(cycle)
            new_cycles.append(cycle)

        # Validate data
        count_mot = len(new_motors)
        count_app = len(new_applications)
        # count_cycle = count_new_elements(db, models.CycleData)
        if count_mot > 1 or count_app > 1:
            raise ValueError(f"Inconsistent data! Only 1 motor and 1 application may be defined per ZIP. But we found {count_mot} Motor(s) and {count_app} Application(s))")
        if add_new_entities_to_db_by_user:
            try:
                await db.commit()
            except Exception:
                await db.rollback()  # close transaction
                logger.exception("Error occurred while committing to DB")
                raise
        return new_cycles

    @staticmethod
    async def load_cycle_metrics(db: AsyncSession, db_application: models.Application, cycle_ids: set[int] = None, logger: logging.Logger = logger):
        cycle_train_ids = defaultdict(list)
        # Fill mapping of cycle ids to classification
        for cycle in db_application.cycledatas:
            cycle_train_ids[cycle.classification.label].append(cycle.id)

        all_df = await load_measurements_by_app(db, db_application.id, cycle_ids=cycle_ids, metrics=[MeasurementMetric.ACT_CURRENT], logger=logger)
        if all_df.empty:
            raise ValueError("We could not find any raw signals of the cycles")

        # Copy the dataframe to avoid changing the manipulations
        all_df = all_df.copy()

        # set label based on classification [0 = bad, 1 = good, -1 = unknown]
        # crate series for good and bad ids for faster mapping
        id_labels = pd.concat([
            pd.Series(Classification.BAD.code, index=cycle_train_ids[Classification.BAD.label]),
            pd.Series(Classification.GOOD.code, index=cycle_train_ids[Classification.GOOD.label])
        ])
        all_df['label'] = all_df['cycle_id'].map(id_labels).fillna(-1).astype(int)
        return all_df

    @staticmethod
    async def update(message: UpdateMessage, tsk_param: TskParam):
        # Use new db session
        async with database.session_manager.session_maker() as db:
            db: AsyncSession
            try:
                # Serialize tskParam
                tsk_param.dump()
                run = await db.get(models.Run, tsk_param.run_id)
                if run is None:
                    run = models.Run(
                        id=tsk_param.run_id,
                        task=message.task.value,
                        state=message.status.value,
                        application_id=tsk_param.application_id,
                        state_metadata=message.metadata,
                        dataset_name=tsk_param.dataset_name,
                        user_id=tsk_param.user_id,
                        time_created=datetime.now(timezone.utc)
                    )
                    db.add(run)
                else:
                    if run.state == RunState.SUCCESS.value or run.state == RunState.ERROR.value:
                        tsk_param.logger.warning("Run %s is already in terminal state %s - ignoring update %s", run.id, run.state, message)
                        return
                    run.task = message.task.value
                    run.state = message.status.value
                    run.application_id = tsk_param.application_id
                    run.state_metadata = message.metadata
                    run.dataset_name = tsk_param.dataset_name
                    run.user_id = tsk_param.user_id
                    run.time_last_activity = datetime.now(timezone.utc)

                tsk_param.logger.debug("Update run %s to state %s with application_id %s and message %s", run.id, run.state, tsk_param.application_id, message)
                await db.commit()
                await db.refresh(run)

                # Update cycle_run relations only if not zip file
                if not tsk_param.file:
                    # Fetch existing cycle_ids
                    existing_cycle_ids = set(
                        (await db.execute(
                            select(models.cycle_run.c.cycle_id)
                            .where(models.cycle_run.c.run_id == run.id)
                        )).scalars().all()
                    )
                    # Compute additions and removals
                    new_cycle_ids = set(tsk_param.using_cycle_ids)
                    to_add = new_cycle_ids - existing_cycle_ids
                    to_remove = existing_cycle_ids - new_cycle_ids
                    if to_remove:
                        await db.execute(
                            delete(models.cycle_run)
                            .where(models.cycle_run.c.run_id == run.id)
                            .where(models.cycle_run.c.cycle_id.in_(to_remove))
                        )
                    if to_add:
                        values = [{"run_id": run.id, "cycle_id": cid} for cid in to_add]
                        await db.execute(insert(models.cycle_run), values)
                    await db.commit()
            except Exception:
                await db.rollback()  # close transaction
                logger.exception("Error in Utils.update()")
                tsk_param.logger.exception("Error in Utils.update()")
                raise

    @staticmethod
    async def tsk_train(tsk_param: TskParam):
        async with database.session_manager.session() as db:
            try:
                db_user = (await db.execute(
                    select(models.User)
                    .where(models.User.id == int(tsk_param.user_id))
                    .options(
                        selectinload(models.User.motors).selectinload(models.Motor.application).selectinload(models.Application.cycledatas))
                )).scalars().one_or_none()
                if not db_user:
                    error_msg = "User unknown or not allow to run trainings."
                    tsk_param.logger.error(error_msg)
                    raise ValueError(error_msg)

                # Get the current event loop that is running
                loop = asyncio.get_running_loop()

                # Define the notification function that will be called during the training
                def trainer_notify(data: UpdateMessage):
                    # Use asyncio to run the update_client function in the correct event loop
                    asyncio.run_coroutine_threadsafe(Utils.update(data, tsk_param), loop)

                tsk_param.logger.debug('Starting Model Creation.')
                await Utils.update(UpdateMessage(status=RunState.RUNNING, task=RunTask.TRAIN, metadata={'step': 0}), tsk_param)  # step 0 = processing_data
                start_full = time.perf_counter()

                # Zip File train
                if tsk_param.file:
                    datas = await Utils.read_zip_for_train(tsk_param.file)
                    new_cycles = await Utils.convert_csv_into_entities(datas, db, db_user)  # convert and persisting to db
                    all_df = pd.concat([cycle.dataframe for cycle in new_cycles if cycle.isTrainCycle()])

                    applications = {c.application for c in new_cycles}
                    if not applications:
                        raise ValueError("No applications found.")
                    if len(applications) != 1:
                        raise ValueError("Only one application per ZIP file is allowed.")
                    db_application = applications.pop()

                # DB train
                elif tsk_param.application_id:
                    stmt = (select(models.Application)
                            .where(models.Application.id == int(tsk_param.application_id))
                            .options(
                                selectinload(models.Application.motor),
                                selectinload(models.Application.cycledatas)))

                    if not db_user.isAdmin():
                        stmt = (
                            stmt.join(models.Motor)
                            .join(models.motor_user)
                            .where(models.motor_user.c.user_id == db_user.id)
                        )

                    db_application = (await db.execute(stmt)).scalars().one_or_none()
                    if not db_application:
                        raise ValueError(f"Application not found. Application ID: {tsk_param.application_id}")

                    # Fetch relevant cycles from DB
                    relevant_cycles = tsk_param.using_cycle_ids or [c.id for c in db_application.cycledatas if c.isTrainCycle() and not c.disabled]
                    if 'used' in tsk_param.filters:
                        start_temp = time.perf_counter()
                        stmt = (
                            select(models.cycle_run.c.cycle_id)
                            .distinct()
                            .join(models.Run)
                            .where(
                                models.cycle_run.c.cycle_id.in_(relevant_cycles),
                                ~models.Run.disabled,
                                models.Run.application_id == int(db_application.id),
                                models.Run.task == RunTask.TRAIN.value,
                                models.Run.state == RunState.SUCCESS.value,
                            )
                        )
                        used_cycle_ids = set((await db.scalars(stmt)).all())
                        relevant_cycles = [c for c in relevant_cycles if c not in used_cycle_ids]
                        tsk_param.logger.debug("Time taken for filtering used cycles: %.4f s", time.perf_counter()-start_temp)

                    if not relevant_cycles:
                        error_msg = "No cycles available for train. Please check your selection."
                        tsk_param.logger.error(error_msg)
                        await Utils.update(UpdateMessage(status=RunState.ERROR, task=RunTask.TRAIN, metadata={'message': error_msg}), tsk_param)
                        raise ValueError(error_msg)

                    # log entry

                    tsk_param.logger.debug('Reading database data. Application ID: %d', tsk_param.application_id)
                    all_df = await Utils.load_cycle_metrics(db, db_application, cycle_ids=relevant_cycles, logger=tsk_param.logger)

                    # drop entries with unknown label
                    entries_to_drop = all_df['label'] == -1
                    all_df = all_df[~entries_to_drop]

                else:
                    error_msg = "Values of TskParam are not valid"
                    tsk_param.logger.error(error_msg)
                    raise ValueError(error_msg)

                # check if application has a valid id
                if not db_application.id:
                    error_msg = "Application-ID must have a value at this point."
                    tsk_param.logger.error(error_msg)
                    raise ValueError(error_msg)
                tsk_param.application_id = db_application.id

                # Send an update with all relations of user, motor, app
                await Utils.update(UpdateMessage(status=RunState.RUNNING, task=RunTask.TRAIN, metadata={'step': 0}), tsk_param)  # step 0 = processing_data

                # Train only with user-selected cycles
                orig_cycles_count = all_df['cycle_id'].nunique()
                if tsk_param.using_cycle_ids:
                    all_df = all_df[all_df['cycle_id'].isin(tsk_param.using_cycle_ids)]
                else:
                    tsk_param.logger.info("No user-selection of cycles - using all cycles without db-relation.")
                using_cycle_ids = all_df['cycle_id'].unique()
                tsk_param.logger.debug('Slice data from %d to %d cycles by user-selection.', orig_cycles_count, len(using_cycle_ids))
                tsk_param.using_cycle_ids = using_cycle_ids.tolist()

                # Validate data
                # Calculate unique IDs for each label
                unique_ids_label_1 = all_df[all_df.label == 1]['cycle_id'].nunique()  # good
                unique_ids_label_0 = all_df[all_df.label == 0]['cycle_id'].nunique()  # bad

                # Validate the amount of data
                if unique_ids_label_0 < 5 or unique_ids_label_1 < 5:
                    error_msg = "Data set is too small to train the model. Please upload at least 5x good and 5x bad cycles."
                    tsk_param.logger.error(error_msg)
                    raise ValueError(error_msg)

                # Call the Trainer-Pipeline
                trainer = Trainer(data=all_df.copy(), pipeline_config=tsk_param.pipeline_config, logger=tsk_param.logger, callback=trainer_notify)
                start = time.perf_counter()
                try:
                    await run_in_threadpool(trainer.train)
                except Exception as e:
                    await Utils.update(UpdateMessage(status=RunState.ERROR, task=RunTask.TRAIN, metadata={'message': e.args[0]}), tsk_param)
                    raise
                elapsed_time = time.perf_counter() - start

                top_models: list[models.Model] = trainer.get_top_models()
                reduction_summary = trainer._reduction_summary
                if reduction_summary and reduction_summary.get('cycles_summary'):
                    # Disable all cycles that were removed during preprocessing
                    removed_cycles_ids = [
                        k for k, v in reduction_summary['cycles_summary'].items()
                        if v.get('status') == 'Removed'
                    ]
                    if removed_cycles_ids:
                        stmt = (
                            update(models.CycleData)
                            .where(models.CycleData.id.in_(removed_cycles_ids))
                            .values(disabled=True)
                        )
                        await db.execute(stmt)
                await Utils.update(UpdateMessage(status=RunState.RUNNING, task=RunTask.TRAIN, metadata={'step': 3, 'message': 'Persisting models...'}), tsk_param)  # step 3 = saving_models
                await Utils.validate_and_persist_models(top_models, tsk_param, db)

                await Utils.update(UpdateMessage(status=RunState.RUNNING, task=RunTask.TRAIN, metadata={'step': 4, 'message': 'Generating summary...'}), tsk_param)  # step 4 = generating_summary
                trainResponse = TrainResponse(
                    motor_id=db_application.motor.id,
                    motor_serial=db_application.motor.serial,
                    motor_part=db_application.motor.part,
                    application_id=db_application.id,
                    application_recipe=db_application.recipe,
                    application_context_code=db_application.context_code,
                    amount_good_cycles=unique_ids_label_1,
                    amount_bad_cycles=unique_ids_label_0,
                    used_total_time_s=round(elapsed_time, 6),
                    dataset_name=tsk_param.dataset_name,
                    run_id=tsk_param.run_id
                )
                for model in top_models:
                    model_title = model.model_name
                    trainResponse.models[model_title] = {'accuracy': model.model_metadata['scores']['accuracy']}

                await Utils.update(
                    UpdateMessage(
                        status=RunState.SUCCESS,
                        task=RunTask.TRAIN,
                        metadata={
                            "bad_cycles": all_df[all_df.label == 0]['cycle_id'].unique().tolist(),
                            "good_cycles": all_df[all_df.label == 1]['cycle_id'].unique().tolist(),
                            "skipped_cycles": [],  # Trainer does currently not drop cycles on Training
                            "used_total_time_s": trainResponse.used_total_time_s
                        }
                    ),
                    tsk_param)

                tsk_param.logger.info("Total time for tsk_train: %.6f s", time.perf_counter() - start_full)
                tsk_param.finish()
                return trainResponse
            except Exception as e:
                await db.rollback()  # close transaction
                tsk_param.logger.exception('Error on execute task')
                await Utils.update(UpdateMessage(status=RunState.ERROR, task=RunTask.TRAIN, metadata={'message': str(e)}), tsk_param)
                raise

    @staticmethod
    async def tsk_predict(tsk_param: TskParam):
        async with database.session_manager.session() as db:
            db_user = (await db.execute(
                select(models.User)
                .where(models.User.id == int(tsk_param.user_id))
                .options(
                    selectinload(models.User.motors).selectinload(models.Motor.application))
            )).scalars().one_or_none()
            if not db_user:
                error_msg = "User unknown or not allow to run prediction"
                tsk_param.logger.error(error_msg)
                raise ValueError(error_msg)

            # Validate the train_run_id if available
            if tsk_param.train_run_id:
                try:
                    UUID(tsk_param.train_run_id)
                except ValueError:
                    raise ValueError("Run id invalid")

            try:
                # Get the current event loop that is running
                loop = asyncio.get_running_loop()

                # Define the notification function that will be called during the training
                def predictor_notify(data: UpdateMessage):
                    # Use asyncio to run the update_client function in the correct event loop
                    future = asyncio.run_coroutine_threadsafe(Utils.update(data, tsk_param), loop)
                    future.result()  # wait for completion to avoid overlapping updates

                tsk_param.logger.debug('Starting Model Application')
                # the application is not verified at this point and should not used in notify
                await Utils.update(UpdateMessage(status=RunState.RUNNING, task=RunTask.PREDICTION, metadata={'step': 0}), tsk_param)
                start_full = time.perf_counter()

                # Retrieve TS data
                start = time.perf_counter()
                if tsk_param.file:  # No preview - load full data
                    start_data = time.perf_counter()
                    datas = Utils.read_zip_for_test(tsk_param.file)
                    tsk_param.logger.debug('Time taken to read_zip_for_test %.4f', time.perf_counter() - start_data)

                    start_data = time.perf_counter()
                    new_cycles = await Utils.convert_csv_into_entities(datas, db, db_user)
                    tsk_param.logger.debug('Time taken to convert_csv_into_entities %.4f', time.perf_counter() - start_data)

                    applications = {c.application for c in new_cycles}
                    if not applications:
                        raise ValueError("No applications found.")
                    if len(applications) != 1:
                        raise ValueError("Only one application per ZIP file is allowed.")
                    db_application = applications.pop()
                    tsk_param.logger.debug('Based on ZIP file, application id: %s', db_application.id)
                    input_ts_df = pd.concat([cycle.dataframe for cycle in new_cycles if setting.predict_all_cycles or not cycle.isTrainCycle()])
                    # Cause we do not persist cycles from ZIP, we have no 'used' filter here.

                elif tsk_param.application_id:
                    stmt = (select(models.Application)
                            .where(models.Application.id == int(tsk_param.application_id))
                            .options(
                                selectinload(models.Application.motor),
                                selectinload(models.Application.cycledatas.and_(
                                    models.CycleData.classification == Classification.UNKNOWN,
                                    ~models.CycleData.disabled
                                )),
                                selectinload(models.Application.runs.and_(
                                    models.Run.task == RunTask.PREDICTION.value,
                                    models.Run.state == RunState.SUCCESS.value,
                                    ~models.Run.disabled
                                ))))

                    if not db_user.isAdmin():
                        stmt = (
                            stmt.join(models.Motor)
                            .join(models.motor_user)
                            .where(models.motor_user.c.user_id == db_user.id)
                        )

                    start_temp = time.perf_counter()
                    db_application = (await db.execute(stmt)).scalars().one_or_none()
                    tsk_param.logger.debug("Time taken for postgres-load: %.4f s", time.perf_counter()-start_temp)

                    if not db_application:
                        raise ValueError(f"Application not found. Application ID: {tsk_param.application_id}")

                    tsk_param.logger.debug('Application_id: %s', db_application.id)
                    relevant_cycles = tsk_param.using_cycle_ids or {c.id for c in db_application.cycledatas}
                    if 'used' in tsk_param.filters:
                        start_temp = time.perf_counter()
                        stmt = (
                            select(distinct(models.cycle_run.c.cycle_id))
                            .join(models.Run)
                            .where(
                                models.cycle_run.c.cycle_id.in_([c.id for c in db_application.cycledatas]),
                                ~models.Run.disabled,
                                models.Run.application_id == int(db_application.id),
                                models.Run.task == RunTask.PREDICTION.value,
                                models.Run.state == RunState.SUCCESS.value,
                            )
                        )
                        cycle_runs = (await db.execute(stmt)).all()
                        used_cycle_ids = {row[0] for row in cycle_runs}
                        # remove already used cylces
                        relevant_cycles -= used_cycle_ids
                        tsk_param.logger.debug("Time taken for filtering used cycles: %.4f s", time.perf_counter()-start_temp)

                    if not relevant_cycles:
                        error_msg = "No cycles available for prediction. Please check your selection."
                        tsk_param.logger.error(error_msg)
                        await Utils.update(UpdateMessage(status=RunState.ERROR, task=RunTask.PREDICTION, metadata={'message': error_msg}), tsk_param)
                        raise ValueError(error_msg)
                    start_temp = time.perf_counter()
                    input_ts_df = (await load_measurements_by_app(db, db_application.id, cycle_ids=relevant_cycles, metrics=[MeasurementMetric.ACT_CURRENT], logger=tsk_param.logger)).copy()
                    tsk_param.logger.debug("Time taken for database-load: %.4f s", time.perf_counter()-start_temp)
                    input_ts_df.rename(columns={'cycleData_id': 'cycle_id'}, inplace=True)
                else:
                    error_msg = "Values of TskParam are not valid"
                    tsk_param.logger.error(error_msg)
                    raise ValueError(error_msg)

                # Log entry
                tsk_param.logger.debug("Time taken for read data: %.4f s", time.perf_counter()-start)

                start = time.perf_counter()
                # Send an update with all relations of user, motor, app
                tsk_param.application_id = db_application.id
                await Utils.update(UpdateMessage(status=RunState.RUNNING, task=RunTask.PREDICTION, metadata={'step': 0}), tsk_param)

                # Predict only user-selected cycles
                orig_cycles_count = input_ts_df['cycle_id'].nunique()

                if 'outliers' in tsk_param.filters:
                    # Get only inliers cycles
                    BaseClass.mask_outliers(input_ts_df)

                    # disable outlier cycles in db
                    outliers = input_ts_df.loc[input_ts_df['outlier'], 'cycle_id'].unique().tolist()
                    if outliers:
                        await db.execute(
                            models.CycleData.__table__.update()
                            .where(models.CycleData.id.in_(outliers))
                            .values(disabled=True)
                        )
                        await db.commit()
                        tsk_param.logger.debug("Disabled outlier cycles: %s", outliers)

                    # keep only inliers
                    # input_ts_df = input_ts_df[input_ts_df['outlier'] == False]
                    input_ts_df = input_ts_df[~input_ts_df['outlier']]

                if tsk_param.using_cycle_ids:
                    input_ts_df = input_ts_df[input_ts_df['cycle_id'].isin(tsk_param.using_cycle_ids)]

                sliced_cycles_count = input_ts_df['cycle_id'].nunique()
                tsk_param.using_cycle_ids = input_ts_df['cycle_id'].unique().tolist()
                tsk_param.logger.debug('Slice data from %d to %d cycles by user-selection.', orig_cycles_count, sliced_cycles_count)

                if sliced_cycles_count == 0:
                    error_msg = "No cycles available for prediction. Please check your selection."
                    tsk_param.logger.error(error_msg)
                    await Utils.update(UpdateMessage(status=RunState.ERROR, task=RunTask.PREDICTION, metadata={'message': error_msg}), tsk_param)
                    raise ValueError(error_msg)

                tsk_param.logger.debug("Time taken for filtering data: %.4f s", time.perf_counter()-start)

                # Load the specific train-run and them ai-models
                stmt = (select(models.Run)
                        .where(
                            models.Run.application_id == tsk_param.application_id,
                            models.Run.task == RunTask.TRAIN.value,
                            models.Run.state == RunState.SUCCESS.value,
                            ~models.Run.disabled)
                        .options(
                            selectinload(models.Run.models),
                            selectinload(models.Run.application).selectinload(models.Application.motor))
                        .limit(1))
                if tsk_param.train_run_id:
                    stmt = stmt.where(models.Run.id == tsk_param.train_run_id)
                else:
                    # take latest
                    stmt = stmt.order_by(models.Run.id.desc())

                start = time.perf_counter()
                db_train_run = (await db.execute(stmt)).scalars().first()
                tsk_param.logger.debug("Time taken for reading train-run data: %.4f s", time.perf_counter()-start)

                if not db_train_run or all(model.disabled for model in db_train_run.models):
                    raise ValueError("No matching trained models found")

                # Get all active models of the run
                db_models = [model for model in db_train_run.models if not model.disabled]

                # filter only act_current in case there are fields returned
                input_ts_df = input_ts_df[input_ts_df['field'] == "act_current"]

                start = time.perf_counter()
                predictor = Predictor(data=input_ts_df.copy(), logger=tsk_param.logger, models=db_models, callback=predictor_notify)
                try:
                    predictor_result_df, _ = await run_in_threadpool(predictor.predict)
                    reduction_summary = predictor._reduction_summary
                    if reduction_summary and reduction_summary.get('cycles_summary'):
                        # Disable all cycles that were removed or ignored during preprocessing
                        removed_cycles_ids = [
                            k for k, v in reduction_summary['cycles_summary'].items()
                            if v.get('status') in ('Removed', 'Ignored')
                        ]
                        if removed_cycles_ids:
                            tsk_param.logger.debug("Disabling removed or ignored cycles by predictor: %s", removed_cycles_ids)
                            stmt = (
                                update(models.CycleData)
                                .where(models.CycleData.id.in_(removed_cycles_ids))
                                .values(disabled=True)
                            )
                            await db.execute(stmt)
                            await db.commit()
                except Exception as e:
                    await Utils.update(UpdateMessage(status=RunState.ERROR, task=RunTask.PREDICTION, metadata={'message': e.args[0]}), tsk_param)
                    raise
                end = time.perf_counter()
                tsk_param.logger.debug("Time taken for init and run Predictor: %.4f s", end-start)

                # Check if any cycles were predicted
                if predictor_result_df.empty:
                    error_msg = "No cycles were predicted successfully - all were removed or ignored during prediction."
                    tsk_param.logger.warning(error_msg)
                    await Utils.update(UpdateMessage(status=RunState.ERROR, task=RunTask.PREDICTION, metadata={'message': error_msg}), tsk_param)
                    # Return empty result cause prediction failed but it is not really an exception
                    return

                # Update the effective used cycle-ids in taskparam
                tsk_param.using_cycle_ids = predictor_result_df['cycle_id'].unique().tolist()

                predictResponse = PredictResponse(
                    motor_serial=db_application.motor.serial,
                    motor_part=db_application.motor.part,
                    application_id=db_application.id,
                    application_context_code=db_application.context_code,
                    application_recipe=db_application.recipe,
                    used_train_run_id=db_train_run.id,
                    amount_unknown_cycles=predictor_result_df['cycle_id'].nunique(),
                    used_total_time_s=round(end-start, 6),
                    dataset_name=tsk_param.dataset_name
                )

                majority = predictor_result_df["label"].value_counts().idxmax()
                predictResponse.result.prediction = Classification.GOOD.label if majority == Classification.GOOD.code else Classification.BAD.label
                predictResponse.result.confidence_median = float(predictor_result_df["confidence_1"].median())
                predictResponse.result.confidence_mean = float(predictor_result_df["confidence_1"].mean())
                predictResponse.result.feature_dimension = [model.model_metadata['input_dim'] for model in db_models]

                skipped_cycles = list(map(int, set(input_ts_df['cycle_id'].unique()) - set(predictor_result_df['cycle_id'].unique())))
                await Utils.update(
                    UpdateMessage(
                        status=RunState.SUCCESS,
                        task=RunTask.PREDICTION,
                        metadata={
                            # Some values here are redundant stored in Prediction, but we keep it for easier/faster access
                            "bad_cycles": predictor_result_df.loc[predictor_result_df["label"] == 0, "cycle_id"].tolist(),
                            "good_cycles": predictor_result_df.loc[predictor_result_df["label"] == 1, "cycle_id"].tolist(),
                            "skipped_cycles": skipped_cycles,
                            "confidence_mean": predictResponse.result.confidence_mean,
                            "confidence_median": predictResponse.result.confidence_median,
                            "used_total_time_s": predictResponse.used_total_time_s,
                            "used_train_run_id": str(predictResponse.used_train_run_id),
                        }),
                    tsk_param)

                # Persist predictions
                db_run = (await db.execute(select(models.Run).where(models.Run.id == tsk_param.run_id))).scalar_one_or_none()
                if db_run:
                    metrics_cols = [c for c in predictor_result_df.columns if c not in ("cycle_id", "model_id")]
                    df_long = predictor_result_df.melt(
                        id_vars=["cycle_id", "model_id"],
                        value_vars=metrics_cols,
                        var_name="metrics",
                        value_name="value"
                    )

                    for _, row in df_long.iterrows():
                        prediction = models.Prediction(
                            run_id=db_run.id,
                            model_id=row["model_id"],
                            cycle_id=row["cycle_id"],
                            metrics=row["metrics"],
                            value=row["value"]
                        )
                        db.add(prediction)
                    await db.commit()

                # log entry
                tsk_param.logger.debug("Total time for predict: %.6f s", time.perf_counter() - start_full)
                return predictResponse
            except Exception as e:
                tsk_param.logger.exception('Error on execute task')
                await Utils.update(UpdateMessage(status=RunState.ERROR, task=RunTask.PREDICTION, metadata={'message': str(e)}), tsk_param)
                raise
            finally:
                tsk_param.finish()

    @staticmethod
    async def validate_and_persist_models(top_models: list[models.Model], tsk_param: TskParam, db: AsyncSession):
        """
        Function to coordinate the upload of a motors models to the postgres database based
        on whether or not the motor has a better model saved based on the F1 score.
        """

        start_g1 = time.perf_counter()
        for new_model in top_models:
            start = time.perf_counter()
            new_model.run_id = tsk_param.run_id
            await Utils.persist_mot_model(tsk_param, new_model, db)
            new_model.stored = True
            logger.debug("Time taken for persist model %s: %.4f s", new_model.model_name, time.perf_counter()-start)

        logger.info("Time taken for persist all models: %.4f s", time.perf_counter()-start_g1)

    @staticmethod
    async def persist_mot_model(tsk_param: TskParam, model: models.Model, db: AsyncSession):
        """
        Function for the upload of the ONNX serialised user-created models to the "models" table in the database

        Once the user has created ML/DL models, the model is serialised using ONNX and
        is uploaded to the database. This model can then be retrieved when running
        validation of the created model on a new signal.
        """
        try:
            model.run_id = tsk_param.run_id
            if type(model.model_onnx) is not bytes:
                input_dim = model.model_metadata['input_dim']
                tsk_param.logger.info("Converting model to ONNX: %s", model.model_name)
                start = time.perf_counter()
                model.model_onnx = sk_to_onnx(input_dim, model.model_onnx)
                tsk_param.logger.info("Time taken for onnx-converting: %.4f s", time.perf_counter()-start)
            db.add(model)
            await db.commit()
        except Exception:
            await db.rollback()  # close transaction
            raise

    @staticmethod
    async def previewSignals(user: models.User, context: dict, db: AsyncSession, tskParam: TskParam, plots: bool = True, kind: RunTask = None):
        previewResponse = PreviewResponse()
        application_id: int = None
        cycle_ids: set[int] = None

        if tskParam.file:
            # Read the data from the zip file
            start = time.perf_counter()
            datas = read_zip(tskParam.file)
            tskParam.logger.info("Time taken for read_zip: %.4f s", time.perf_counter() - start)

            start = time.perf_counter()
            cycles = await Utils.convert_csv_into_entities(datas, db)
            tskParam.logger.debug("Time taken for convert_csv_into_entities: %.4f s", time.perf_counter() - start)

            start = time.perf_counter()
            if not cycles:
                raise ValueError("No valid cycle data found in zip file")

            if cycles[0].application.id:
                previewResponse.application = cycles[0].application.db_model_to_json()
            else:
                app = datas[0][1]
                previewResponse.application = {"context_code": app['context_code'], "recipe": app['recipe']}

            if cycles[0].application.motor.id:
                previewResponse.motor = cycles[0].application.motor.db_model_to_json()
            else:
                motor = datas[0][0]
                previewResponse.motor = {'serial': motor['serial'], 'part': motor['part'], 'machine': motor['machine'], 'client': motor['client']}

            df = pd.concat([d[3] for d in datas])
            df = df[df['field'] == 'act_current']
            df['label'] = df['label'].fillna(-1)  # set unknown label to -1

            application_id = int(cycles[0].application.id) if cycles[0].application.id else None
            cycle_ids = set([c.id for c in cycles[0].application.cycledatas])
            tskParam.logger.debug("Time taken for processing zip data: %.4f s", time.perf_counter() - start)
        else:

            if not kind:
                classification_filter = True
            elif kind == RunTask.TRAIN:
                classification_filter = models.CycleData.classification.in_([Classification.GOOD, Classification.BAD])
            else:
                classification_filter = models.CycleData.classification == Classification.UNKNOWN

            stmt = (
                select(models.Application)
                .where(models.Application.id == int(tskParam.application_id))
                .options(
                    selectinload(models.Application.motor),
                    selectinload(
                        models.Application.cycledatas.and_(
                            ~models.CycleData.disabled,
                            classification_filter
                        ))
                )
            )

            if not user.isAdmin():
                stmt = (
                    stmt.join(models.Motor)
                    .join(models.motor_user)
                    .where(models.motor_user.c.user_id == user.id)
                )

            db_application = (await db.execute(stmt)).scalars().one_or_none()
            if not db_application:
                raise ValueError("Application not found")

            application_id = db_application.id
            cycle_ids = set([c.id for c in db_application.cycledatas])

            # Load metrics efficiently from TimescaleDB
            df = await Utils.load_cycle_metrics(db, db_application, cycle_ids=list(cycle_ids), logger=tskParam.logger)
            previewResponse.motor = db_application.motor.db_model_to_json()
            previewResponse.application = db_application.db_model_to_json()

        label_counts_per_cycle = df.groupby('cycle_id')['label'].first()
        stats = PreviewResponse.Statistics(
            cycles={
                Classification.GOOD.label: int(label_counts_per_cycle.eq(Classification.GOOD.code).sum()),
                Classification.BAD.label: int(label_counts_per_cycle.eq(Classification.BAD.code).sum()),
                Classification.UNKNOWN.label: int(label_counts_per_cycle.eq(Classification.UNKNOWN.code).sum()),
                'total': len(label_counts_per_cycle)
            },
            points={
                Classification.GOOD.label: int(df['label'].eq(Classification.GOOD.code).sum()),
                Classification.BAD.label: int(df['label'].eq(Classification.BAD.code).sum()),
                Classification.UNKNOWN.label: int(df['label'].eq(Classification.UNKNOWN.code).sum()),
                'total': len(df['label'])
            }
        )
        previewResponse.statistics = stats

        if plots:
            previewResponse.plots['scatter'] = {}
            plot_start = time.perf_counter()
            process_data(df)  # preprocess data for plotting
            figs, fig_balance_signal = ts_visualisation(df,
                                                        title_pie=context['label_distribution'],
                                                        title_unknown_signals=context['unknown_signals'],
                                                        title_bad_signals=context['bad_quality_signals'],
                                                        title_good_signals=context['good_quality_signals'],
                                                        title_cycles=context['cycles'],
                                                        title_scatter_x=context['scatter_x'],
                                                        title_scatter_y=context['scatter_y'])
            tskParam.logger.debug("Time taken for plot generation: %.4f s", time.perf_counter() - plot_start)
            if kind == RunTask.TRAIN:
                figs = [f for f in figs if f['layout']['meta']['label_id'] in [0, 1]]
                df = df[df['label'] != -1]
            elif kind == RunTask.PREDICTION:
                figs = [f for f in figs if f['layout']['meta']['label_id'] == -1]
                df = df[df['label'] == -1]

            # continue only if df is not empty
            if df.empty:
                tskParam.logger.warning("No data available for plotting after filtering by kind: %s", kind)
                return previewResponse

            # Calc auto-outliers based on values mean
            plot_start = time.perf_counter()
            BaseClass.mask_outliers(df)
            tskParam.logger.debug("Time taken for outlier masking: %.4f s", time.perf_counter() - plot_start)

            # Get the used cycle_ids
            used_cycle_ids_set = set()
            if application_id and cycle_ids and cycle_ids != {None}:
                stmt = (
                    select(distinct(models.cycle_run.c.cycle_id))
                    .join(models.Run)
                    .where(
                        models.cycle_run.c.cycle_id.in_(cycle_ids),
                        ~models.Run.disabled,
                        models.Run.application_id == application_id,
                        models.Run.task == kind.value if kind else True,
                        models.Run.state != RunState.ERROR.value,
                    )
                )
                cycle_runs = (await db.execute(stmt)).all()
                used_cycle_ids_set = set({row[0] for row in cycle_runs})

            plot_start = time.perf_counter()
            # Pre-compute cycle_id filtering and group data once
            cycle_groups = df.groupby('cycle_id').agg({
                'timestamp': ['min', 'max'],
                'outlier': 'any'
            }).to_dict('index')

            cycle_metadatas = {}
            if not tskParam.file and db_application:
                cycle_metadatas = Utils._extract_cycle_metadata(db_application.cycledatas, from_db=True)
            elif datas:
                cycle_metadatas = Utils._extract_cycle_metadata(datas)
            else:
                tskParam.logger.warning("No application data available to extract cycle metadata.")

            for f in figs:
                label_id = f.layout.meta['label_id']
                for trace in f.data:
                    trace_id_int = int(trace.name)
                    # Direct dictionary lookups instead of filtering
                    cycle_data = cycle_groups.get(trace_id_int)
                    if cycle_data is None:
                        continue

                    metadata = cycle_metadatas.get(trace_id_int, {})
                    customdata = {
                        'cycle_start_timestamp': int(cycle_data[('timestamp', 'min')].timestamp() * 1000),
                        'cycle_end_timestamp': int(cycle_data[('timestamp', 'max')].timestamp() * 1000),
                        'used': trace_id_int in used_cycle_ids_set,
                        'outlier': cycle_data[('outlier', 'any')],
                        **metadata.get('cycleConfig', {}),
                    }
                    if user.isAdmin():
                        customdata.update(metadata.get('driveConfig', {}))

                    # Filter None values during dict creation
                    trace.customdata = [{k: v for k, v in customdata.items() if v is not None}]

                html_img = f.to_html(full_html=False, include_plotlyjs=False, div_id=f"plot_cycles_{label_id}", config={'displayModeBar': False})
                previewResponse.plots['scatter'][f"plot_cycles_{label_id}"] = html_img
            tskParam.logger.debug("Time taken for plot customdata assignment: %.4f s", time.perf_counter() - plot_start)

            plot_start = time.perf_counter()
            if kind == RunTask.TRAIN:
                fig_balance_signal.update_layout(
                    title=None,
                    showlegend=False,
                    height=200,
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                html_img = fig_balance_signal.to_html(full_html=False, include_plotlyjs=False, div_id='plot_distribution', config={'displayModeBar': False})
                previewResponse.plots["distribution"] = html_img
            tskParam.logger.debug("Time taken for distribution plot generation: %.4f s", time.perf_counter() - plot_start)
        return previewResponse

    @staticmethod
    def _extract_cycle_metadata(cycles, from_db=False) -> dict[int, dict]:
        result: dict[int, dict] = {}
        for c in cycles:
            cycle = c[2] if not from_db else c
            cycle_cfg = cycle.get('cycleconfig') if not from_db else cycle.cycleConfig
            drive_cfg = cycle.get('driveconfig') if not from_db else cycle.driveConfig

            meta = {
                'cycleConfig': {k: cycle_cfg.get(v) for k, v in {
                    'additionalTestConfig': 'additionaltestconfig',
                    'homePos': 'homepos',
                    'spikeTorque': 'spiketorque',
                    'spikeLength': 'spikelength',
                    'loadTorque': 'loadtorque',
                    'velocity': 'velocity',
                    'targetPos': 'targetpos'
                }.items()},
                'driveConfig': {k: drive_cfg.get(v) for k, v in {
                    'positionPGain': 'positionpgain',
                    'positionIGain': 'positionigain',
                    'positionDGain': 'positiondgain',
                    'velocityPGain': 'velocitypgain',
                    'velocityIGain': 'velocityigain'
                }.items()}
            }

            result[cycle.get('id') if not from_db else cycle.id] = meta
        return result

    @staticmethod
    async def import_file_zip(user: models.User, content: bytes, db: AsyncSession, add_seen_cycles: bool = False):
        # Parse the zip file
        start = time.perf_counter()
        datas = read_zip(content)
        logger.info("Time taken for read_zip: %.4f s", time.perf_counter() - start)

        # convert csv data into entities
        new_cycles = await Utils.convert_csv_into_entities(datas, db, user)

        # Check if motor and application are unique
        motors, applications = zip(
            *[((c.application.motor.serial, c.application.motor.part), (c.application.context_code, c.application.recipe)) for c in new_cycles]
        )
        if len(set(motors)) > 1:
            raise ValueError("Multiple motors found in zip file. Only one motor per zip is allowed.")
        if len(set(applications)) > 1:
            raise ValueError("Multiple applications found in zip file. Only one application per zip is allowed.")

        db_cycles = {}
        if not add_seen_cycles:
            # Load existing cycles from database for motor and application
            stmt = (
                select(models.CycleData)
                .options(selectinload(models.CycleData.application))
                .join(models.Application)
                .join(models.Motor)
                .where(models.Motor.serial == motors[0][0],
                       models.Motor.part == motors[0][1],
                       models.Application.context_code == applications[0][0],
                       models.Application.recipe == applications[0][1])
            )
            db_cycles = (await db.execute(stmt)).scalars().all()

            # Calculate hashes for existing cycles
            db_cycles = {cycle.get_hash(application_id=cycle.application.id,
                                        classification=cycle.classification.code,
                                        testCycleCounter=cycle.testCycleCounter,
                                        driveConfig=cycle.driveConfig,
                                        cycleConfig=cycle.cycleConfig): cycle for cycle in db_cycles}

        unseen_cycles: list[models.CycleData] = list()

        # Check if entities already exist in the database
        for cycle in new_cycles:
            if add_seen_cycles or cycle.get_hash(application_id=cycle.application.id,
                                                 classification=cycle.classification.code,
                                                 testCycleCounter=cycle.testCycleCounter,
                                                 driveConfig=cycle.driveConfig,
                                                 cycleConfig=cycle.cycleConfig) not in db_cycles:
                unseen_cycles.append(cycle)

        # persist new cycles into databases
        cycles_with_df = [x for x in unseen_cycles if x.dataframe is not None]
        db.add_all(cycles_with_df)
        try:
            await db.commit()
        except Exception:
            await db.rollback()  # close transaction
            raise

        amount_points = await add_cycles(db, cycles_with_df, logger=logger)
        logger.info("Time taken for import zip: %.4f s", time.perf_counter()-start)

        dto = {"motors": {}}
        for cycle in cycles_with_df:
            motor_id = cycle.application.motor_id
            application_id = cycle.application_id

            # Add Motor-Information if it is not already present
            if motor_id not in dto['motors'].keys():
                dto['motors'][motor_id] = {
                    'serial': cycle.application.motor.serial,
                    'part': cycle.application.motor.part,
                    'machine': cycle.application.motor.machine,
                    'client': cycle.application.motor.client,
                    'time_created': Utils.strftime(cycle.application.motor.time_created),
                    'applications': {}
                }

            # Add Application-Information if it is not already present
            if application_id not in dto['motors'][motor_id]['applications']:
                dto['motors'][motor_id]['applications'][application_id] = {
                    'context_code': cycle.application.context_code,
                    'recipe': cycle.application.recipe,
                    'cycles': {
                        Classification.GOOD.label: 0,
                        Classification.BAD.label: 0,
                        Classification.UNKNOWN.label: 0,
                        'points': amount_points
                    }
                }

            # Count the classification
            dto['motors'][motor_id]['applications'][application_id]['cycles'][cycle.classification.label] += 1

        return dto

    @staticmethod
    async def calculate_motor_conditions(user: models.User, db: AsyncSession, motor_ids: list[int] = []) -> dict | None:
        motor_ids = [int(i) for i in motor_ids if isinstance(i, int) or str(i).isdigit()] if motor_ids else []  # sanitize input
        stmt = (
            select(models.Motor)
            .join(models.Application)
            .join(models.Run)
            .where(
                ~models.Motor.disabled,
                ~models.Application.disabled,
                ~models.Run.disabled,
                models.Motor.id.in_(motor_ids) if motor_ids else True,
                models.Run.task == RunTask.PREDICTION,
                models.Run.state == RunState.SUCCESS,
                models.Motor.users.any(models.User.id == user.id) if not user.isAdmin() else True  # only allow if user has access to motor
            )
            .options(
                # contains_eager -> to load only the filtered applications and runs
                contains_eager(models.Motor.application)
                .contains_eager(models.Application.runs)
            )
            .group_by(models.Motor.id, models.Application.id, models.Run.id)
        )

        # Execute the query and get all columns
        start_db = time.perf_counter()
        motors = (await db.execute(stmt)).unique().scalars().all()
        end_db = time.perf_counter()
        logger.debug("Time taken to load data for motor-score calculation %.2f s", end_db-start_db)
        if not motors:
            return {}

        # Choose EMA period N depending on user role
        # Smaller N -> more fluctuations visible (for exhibitions)
        # Larger N -> smoother curve (for production)
        N = 5 if user.role == UserRole.EXHIBITION else 10

        result = {}
        for motor in motors:
            condition_data = {}
            for app in motor.application:
                if not app.runs:
                    # Skip applications without runs, should not happen due to query filter
                    logger.warning("Skip application %s without runs", app.id)
                    continue

                condition_data[app.id] = {
                    "context_code": app.context_code,
                    "recipe": app.recipe,
                    "prediction_running": False,
                    "has_new_cycles": False,
                    "predictions": []
                }
                runs_sorted = sorted(app.runs, key=lambda r: r.time_created, reverse=True)  # sort runs by created time, newest first
                for run in runs_sorted[:N]:  # only consider latest N runs for score calculation
                    if run.state != RunState.SUCCESS or run.task != RunTask.PREDICTION:
                        logger.warning("Load unnecessary runs %s with state %s and task %s", run.id, run.state, run.task)
                        continue

                    count_good_cycles = len(run.state_metadata.get("good_cycles", []))
                    count_bad_cycles = len(run.state_metadata.get("bad_cycles", []))
                    labels = np.array([1] * count_good_cycles + [0] * count_bad_cycles)
                    confidence = run.state_metadata.get('confidence_mean', 0)

                    # Calculate score if labels exist
                    if labels.size > 0:
                        score = Utils.classifications_to_percentage(labels, confidence)
                        condition_data[app.id]["predictions"].append({
                            "run_id": str(run.id),
                            "health_score": round(score),
                            "health_score_exp": score,
                            "time_last_activity": run.time_last_activity.isoformat(),
                            "time_created": run.time_created.isoformat(),
                            "count_good_cycles": count_good_cycles,
                            "count_bad_cycles": count_bad_cycles,
                        })

            # Calculate machine health score as average across applications
            app_averages = [
                statistics.mean(scores)
                for app_data in condition_data.values()
                if (scores := [pred["health_score_exp"] for pred in app_data.get("predictions", [])])
            ]
            machine_health_avg_score = statistics.mean(app_averages) if app_averages else 0.0
            times = [
                pred["time_last_activity"]
                for app_data in condition_data.values()
                for pred in app_data.get("predictions", [])
                if "time_last_activity" in pred
            ]

            result[motor.id] = {
                "serial": motor.serial,
                "part": motor.part,
                "machine": motor.machine,
                "client": motor.client,
                "motor_health_score_mean": machine_health_avg_score,
                "latest_prediction_time": max(times) if times else None,
                "applications": condition_data
            }

        await db.commit()
        logger.debug("Time taken to calculate motor (%s) condition: %.2f s (db: %.2f s)", motor_ids, time.perf_counter()-end_db, end_db-start_db)
        return result

    @staticmethod
    def strftime(dt: datetime, default='-'):
        '''
        Format datetime to string or return default value
        '''
        return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else default

    @staticmethod
    def ema_from_values(values: list[float], N: int) -> float:
        '''
        Compute "Exponential Moving Average (EMA)" across averages with period N.
        '''
        if not values:
            return 0.0
        alpha = 2 / (N + 1)  # Smoothing factor, smaller N gives more weight to recent values (sharper curve)
        e_t = values[0]  # base case
        for avg in values[1:]:
            e_t = alpha * avg + (1 - alpha) * e_t
        return e_t


# Initialisiere das Singleton
Utils()
