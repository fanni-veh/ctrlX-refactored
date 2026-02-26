from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Literal, TypedDict

from app.models import Motor, Application


@dataclass
class TrainResponse():
    """
    DTO class for train response.
    """

    title: str = "Summary"
    status: Literal['success', 'error'] = 'success'
    motor_id: int = None
    motor_serial: str = None
    motor_part: str = None
    application_id: int = None
    application_context_code: str = None
    application_recipe: str = None
    amount_good_cycles: int = None
    amount_bad_cycles: int = None
    used_total_time_s: float = None
    dataset_name: str = None
    run_id: str = None
    models: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))

    @classmethod
    def example(cls):
        return cls(
            status="success",
            motor_id=1,
            motor_serial="m12345",
            motor_part="p12345",
            application_id=1,
            application_context_code="move_home",
            application_recipe="af0db0bca39c4295f0738ae377ac3396bd47cbe83361ef5597e7656c0e3e9a0c",
            amount_good_cycles=110,
            amount_bad_cycles=80,
            used_total_time_s=13.4894,
            models={
                "model_1": {
                    "accuracy": 0.9,
                },
                "model_2": {
                    "accuracy": 0.81,
                },
                "model_3": {
                    "accuracy": 0.96,
                },
                "model_4": {
                    "accuracy": 0.85,
                }
            }
        ).__dict__


@dataclass
class PredictResponse():
    """
    DTO class for predict response.
    """
    @dataclass
    class ModelResult():
        prediction: Literal['good', 'bad'] = ''
        feature_dimension: list[int] = field(default_factory=list)
        confidence_median: float = 0.0
        confidence_mean: float = 0.0

    title: str = "PredictSummary"
    status: Literal['success', 'error'] = 'success'
    motor_serial: str = None
    motor_part: str = None
    application_id: int = None
    application_recipe: str = None
    application_context_code: str = None
    used_train_run_id: str = None  # UUID
    amount_unknown_cycles: int = None
    dataset_name: str = None
    used_total_time_s: float = None
    result: ModelResult = field(default_factory=ModelResult)

    @classmethod
    def example(cls):
        return cls(
            motor_serial="m12345",
            motor_part="p12345",
            application_id=1,
            used_train_run_id="067d2ac8-e324-7c63-8000-7e0693ccfaf4",
            amount_unknown_cycles=432,
            used_total_time_s=3.94,
            result=PredictResponse.ModelResult(
                prediction="good",
                feature_dimension=[7, 8, 9],
                confidence_median=0.85,
                confidence_mean=0.82
            ).__dict__
        ).__dict__


@dataclass
class PreviewResponse():
    """
    DTO class for preview response.
    """

    class Statistics(TypedDict):
        cycles: dict[str, int]
        points: dict[str, int]

    title = "Preview"
    motor: Motor = None
    application: Application = None
    statistics: Statistics = field(default_factory=Statistics)
    plots: Dict = field(default_factory=dict)

    @classmethod
    def example(cls):
        return cls(
            motor=Motor.example().db_model_to_json(),
            application=Application.example().db_model_to_json(),
            statistics=PreviewResponse.Statistics(
                cycles={
                    "good": 100,
                    "bad": 100,
                    "unknown": 400,
                    "total": 600
                },
                points={
                    "good": 60000,
                    "bad": 60000,
                    "unknown": 240000,
                    "total": 360000
                })
        ).__dict__
