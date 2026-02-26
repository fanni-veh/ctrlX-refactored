from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import os
from pathlib import Path
from uuid import UUID
import numpy as np
import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app import Classification, MeasurementMetric, RunState, RunTask, models
from app.scripts.data_visualization import create_performance_plots, ts_visualisation
from app.scripts.database_helper import load_measurements_by_app
from app.scripts.preprocessing_signal import process_data
from app.scripts.tsa_logging import create_logger
from app.utils import Utils
from app.config import setting

# Initialize the logger for this service
logger = create_logger(__name__, output_file="mind_api")


@dataclass
class RunData:
    run: models.Run  # models.Run
    cycle_ids: List[int]


class ReportService:
    """Service for generating report data from runs"""

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        """Validate run ID format and length"""
        try:
            UUID(str(run_id))
        except ValueError:
            raise ValueError("Run id invalid")

    @staticmethod
    async def _fetch_run_with_permissions(run_id: str, user: models.User, db: AsyncSession) -> RunData:
        """Fetch run data with user permission checks"""
        stmt = (
            select(models.Run, func.array_agg(models.cycle_run.c.cycle_id).label("cycle_ids"))
            .outerjoin(models.cycle_run)
            .where(models.Run.id == run_id)
            .group_by(models.Run.id)
        )

        # Add permission filter for non-admin users
        if not user.isAdmin():
            stmt = (
                stmt.join(models.Application)
                .join(models.Motor)
                .join(models.motor_user)
                .where(models.motor_user.c.user_id == user.id)
            )

        # Add eager loading based on task
        options = [selectinload(models.Run.application).selectinload(models.Application.motor)]

        start = time.perf_counter()
        result = (await db.execute(stmt.options(*options))).one_or_none()
        logger.debug("Database query time: %.4f s", time.perf_counter() - start)

        if not result or not result.Run:
            raise ValueError("Run not found")

        # Add conditional loading after we know the task
        if result.Run.task == RunTask.PREDICTION:
            await db.refresh(result.Run, ['predictions'])
        else:
            await db.refresh(result.Run, ['models'])

        cycle_ids = result.cycle_ids if result.cycle_ids and any(result.cycle_ids) else []
        return RunData(run=result.Run, cycle_ids=cycle_ids)

    @staticmethod
    def _build_base_result(run_data: RunData) -> Dict[str, Any]:
        """Build base result structure with motor, application and run info"""
        run: models.Run = run_data.run
        return {
            "motor": {
                "id": run.application.motor.id,
                "serial_number": run.application.motor.serial,
                "part": run.application.motor.part,
                "machine": run.application.motor.machine,
                "client": run.application.motor.client,
                "time_created": Utils.strftime(run.application.motor.time_created),
            },
            "application": {
                "id": run.application.id,
                "context_code": run.application.context_code,
                "recipe": run.application.recipe,
                "time_created": Utils.strftime(run.application.time_created)
            },
            "run": {
                "id": str(run.id),
                "task": run.task.value,
                "state": run.state.value,
                "cycle_ids": run_data.cycle_ids if run_data.cycle_ids else '-',
                "dataset_name": run.dataset_name,
                "count_models": len(run.models) if run.task == RunTask.TRAIN else '-',
                "last_activity": Utils.strftime(run.time_last_activity),
            }
        }

    @staticmethod
    async def get_report_data(run_id: str, user: Any, db: AsyncSession, context: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate complete report data for a run"""
        ReportService._validate_run_id(run_id)

        run_data = await ReportService._fetch_run_with_permissions(run_id, user, db)
        result = ReportService._build_base_result(run_data)

        # Handle error state
        if run_data.run.state == RunState.ERROR:
            result["run"]["message"] = run_data.run.state_metadata.get('message')
            return result

        # Handle incomplete states
        if run_data.run.state != RunState.SUCCESS:
            result["run"]["step"] = run_data.run.state_metadata.get('step', '-')
            return result

        # Process based on task type
        if run_data.run.task == RunTask.PREDICTION:
            prediction_data = await PredictionReportService.process(run_data, context, db)
            result["run"].update(prediction_data)
        else:
            training_data = await TrainingReportService.process(run_data)
            result["run"].update(training_data)

        return result


class PredictionReportService:
    """Service for processing prediction run reports"""

    @staticmethod
    async def process(run_data: RunData, context: Optional[Dict[str, str]], db: AsyncSession) -> Dict[str, Any]:
        """Process prediction run data"""
        run = run_data.run
        metadata = run.state_metadata

        good_cycles = metadata.get("good_cycles", [])
        bad_cycles = metadata.get("bad_cycles", [])
        skipped_cycles = metadata.get("skipped_cycles", [])

        count_good = len(good_cycles)
        count_bad = len(bad_cycles)

        # Determine prediction result
        prediction = Classification.GOOD.label if count_good > count_bad else Classification.BAD.label
        confidence_mean = metadata.get("confidence_mean")

        # Load visualization data
        plots = await PredictionReportService._generate_plots(run_data, good_cycles, bad_cycles, context, db) if context else None

        # Calculate health metrics
        label_numpy = np.array([1] * count_good + [0] * count_bad)
        score = Utils.classifications_to_percentage(label_numpy, confidence_mean)
        health_color, health_text = Utils.get_color(score, context)

        result = {
            "used_train_run_id": metadata.get("used_train_run_id"),
            "prediction": {
                "prediction": prediction,
                "health_text": health_text,
                "health_color": health_color,
                "health_score": round(score, 2),
                "confidence_mean": confidence_mean,
                "confidence_median": metadata.get("confidence_median"),
                "good_cycles": good_cycles,
                "bad_cycles": bad_cycles,
                "skipped_cycles": skipped_cycles,
                "plots": plots,
            }
        }

        # Add cycle IDs for ZIP uploads without DB data
        if not run_data.cycle_ids:
            result["cycle_ids"] = good_cycles + bad_cycles + skipped_cycles

        return result

    @staticmethod
    async def _generate_plots(run_data: RunData, good_cycles: List[int], bad_cycles: List[int],
                              context: Dict[str, str], db: AsyncSession) -> Dict[str, str]:
        """Generate visualization plots for prediction data"""
        run: models.Run = run_data.run
        relevant_cycles = list({p.cycle_id for p in run.predictions}) if run.predictions else run_data.cycle_ids

        if not relevant_cycles:
            return {}

        # Try loading from database
        df = await load_measurements_by_app(db, run.application_id, cycle_ids=relevant_cycles, metrics=[MeasurementMetric.ACT_CURRENT], logger=logger)
        if not df.empty:
            df.rename(columns={'cycleData_id': 'cycle_id'}, inplace=True)

        # Fallback to ZIP file
        if df.empty:
            df = await PredictionReportService._load_from_zip(str(run.id), relevant_cycles, db)

        if df.empty:
            logger.warning("No raw data available for prediction run %s", run.id)
            return {}

        # Process data for visualization
        df = PredictionReportService._prepare_dataframe(df, good_cycles, bad_cycles)

        # Generate plots
        figs, fig_balance_signal = ts_visualisation(df,
                                                    title_pie=context['label_distribution'] if context else 'Label Distribution',
                                                    title_unknown_signals=context['unknown_signals'] if context else 'Unknown Signals',
                                                    title_bad_signals=context['bad_quality_signals'] if context else 'Bad Quality Signals',
                                                    title_good_signals=context['good_quality_signals'] if context else 'Good Quality Signals',
                                                    title_cycles=context['cycles'] if context else 'Cycles',
                                                    title_scatter_x=context['scatter_x'] if context else 'Scatter X',
                                                    title_scatter_y=context['scatter_y'] if context else 'Scatter Y')

        plots = {}
        if fig_balance_signal:
            plots["pie_signals"] = fig_balance_signal.to_html(full_html=False, include_plotlyjs=False)
        if figs and len(figs) > 0:
            plots["bad_signals"] = figs[0].to_html(full_html=False, include_plotlyjs=False)
        if figs and len(figs) > 1:
            plots["good_signals"] = figs[1].to_html(full_html=False, include_plotlyjs=False)

        return plots

    @staticmethod
    async def _load_from_zip(run_id: str, relevant_cycles: List[int], db: AsyncSession) -> pd.DataFrame:
        """Load data from ZIP file fallback"""
        run_log_path = os.path.join(setting.log_dir, str(run_id))
        base_path = os.path.abspath(run_log_path)
        zip_path = next(Path(base_path).glob("*.zip"), None)

        if not zip_path:
            return pd.DataFrame()

        try:
            datas = Utils.read_zip_for_test(zip_path.read_bytes())
            new_cycles = await Utils.convert_csv_into_entities(datas, db)
            df = pd.concat([cycle.dataframe for cycle in new_cycles
                            if setting.predict_all_cycles or not cycle.isTrainCycle()])
            return df[df['cycle_id'].isin(relevant_cycles)]
        except Exception:
            logger.exception("Failed to load ZIP data for run %s", run_id)
            return pd.DataFrame()

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame, good_cycles: List[int], bad_cycles: List[int]) -> pd.DataFrame:
        """Prepare dataframe with labels for visualization"""
        conditions = [
            df["cycle_id"].isin(bad_cycles),
            df["cycle_id"].isin(good_cycles)
        ]
        choices = [0, 1]
        df["label"] = np.select(conditions, choices, default=-1)

        df = process_data(df)
        return df[df['label'] != -1]


class TrainingReportService:
    """Service for processing training run reports"""

    @staticmethod
    async def process(run_data: RunData) -> Dict[str, Any]:
        """Process training run data"""
        run = run_data.run
        metadata = run.state_metadata

        good_cycles = metadata.get("good_cycles", [])
        bad_cycles = metadata.get("bad_cycles", [])
        skipped_cycles = metadata.get("skipped_cycles", [])

        result = {
            "train": {
                "models": {},
                "used_total_time_s": metadata.get("used_total_time_s", '-'),
                "amount_good_cycles": len(good_cycles) or '-',
                "amount_bad_cycles": len(bad_cycles) or '-'
            }
        }

        # Add cycle IDs for ZIP uploads without DB data
        if not run_data.cycle_ids:
            result["cycle_ids"] = good_cycles + bad_cycles + skipped_cycles

        # Process model data
        for model in run.models:
            model_data = TrainingReportService._process_model(model)
            result["train"]["models"][model.model_name] = model_data

        return result

    @staticmethod
    def _process_model(model: Any) -> Dict[str, Any]:
        """Process individual model data"""
        validation_data = model.model_metadata.get('validation', {})
        y_validation = validation_data.get('y_validation', [])
        y_predicted = validation_data.get('y_predicted', [])

        fig_matrix_html, fig_bar_html, fig_snake_html = create_performance_plots(y_validation, y_predicted)

        scores = model.model_metadata.get('scores', {})
        return {
            "id": model.id,
            "f1": f"{round(scores.get('f1', 0) * 100, 2)}%",
            "recall": f"{round(scores.get('recall', 0) * 100, 2)}%",
            "roc_auc": f"{round(scores.get('roc_auc', 0) * 100, 2)}%",
            "accuracy": f"{round(scores.get('accuracy', 0) * 100, 2)}%",
            "precision": f"{round(scores.get('precision', 0) * 100, 2)}%",
            "balanced_accuracy": f"{round(scores.get('balanced_accuracy', 0) * 100, 2)}%",
            "matthews_cor": round(scores.get('matthews_cor', 0), 4),
            "plots": {
                "matrix": fig_matrix_html,
                "fig_bar_html": fig_bar_html,
                "fig_snake_html": fig_snake_html
            }
        }
