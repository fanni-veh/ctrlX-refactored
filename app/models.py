import datetime
import json
import uuid
import pandas as pd
from sqlalchemy import REAL, BigInteger, Column, Float, Index, Integer, JSON, LargeBinary, String, Boolean, ForeignKey, Table, UniqueConstraint, Uuid, cast, text, CheckConstraint
from sqlalchemy.sql.sqltypes import TIMESTAMP
from sqlalchemy.orm import relationship, declarative_base, Mapped, mapped_column
from typing import Set
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import func
from sqlalchemy import Enum as SQLEnum
from enum import Enum

Base = declarative_base()


motor_user = Table(
    "motor_user",
    Base.metadata,
    Column("motor_id", ForeignKey("motor.id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", ForeignKey("user.id", ondelete="CASCADE"), primary_key=True)
)

cycle_run = Table(
    "cycle_run",
    Base.metadata,
    Column("cycle_id", ForeignKey("cycledata.id", ondelete="CASCADE"), primary_key=True),
    Column("run_id", ForeignKey("run.id", ondelete="CASCADE"), primary_key=True)
)


class TSABaseModel(Base):
    __abstract__ = True

    def db_model_to_json(self, select=None):
        exclude = [c.name for c in self.__table__.columns if c.info.get('json_exclude')]
        result = {}
        for c in self.__table__.columns:
            if c.name not in exclude and (not select or c.name in select):
                value = getattr(self, c.name)
                if isinstance(value, datetime.datetime):
                    value = value.isoformat()  # Serialize DateTime as an ISO format string
                result[c.name] = value
        return result


class User(TSABaseModel):
    __tablename__ = "user"

    class Role(Enum):
        ADMIN = "admin"
        GUEST = "guest"
        SERVICE = "service"
        EXHIBITION = "exhibition"

    id = Column(Integer, primary_key=True, nullable=False)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)
    disabled = Column(Boolean, default=False, nullable=False, server_default='false')
    last_login = Column(TIMESTAMP(timezone=True))
    time_created = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    role: Mapped[Role] = mapped_column(SQLEnum(Role, name="user_role", values_callable=lambda x: [e.value for e in x]), nullable=False, server_default='guest')
    motors: Mapped[Set["Motor"]] = relationship("Motor", back_populates="users", secondary=motor_user)
    api_keys: Mapped[Set["ApiKey"]] = relationship("ApiKey", back_populates="user", passive_deletes=True)

    def isAdmin(self):
        return self.role == User.Role.ADMIN

    def isService(self):
        return self.role == User.Role.SERVICE


class ApiKey(TSABaseModel):
    __tablename__ = "api_key"

    id = Column(Integer, primary_key=True, nullable=False)
    user_id = mapped_column(ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    access_key = Column(String, nullable=False)
    counter = Column(Integer, default=0, nullable=False, server_default='0')
    role = Column(String, nullable=False, server_default='read')
    last_used = Column(TIMESTAMP(timezone=True))
    expired_at = Column(TIMESTAMP(timezone=True), nullable=True)
    disabled = Column(Boolean, default=False, nullable=False, server_default='false')
    time_created = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    user: Mapped["User"] = relationship(back_populates="api_keys", passive_deletes=True)


class Model(TSABaseModel):
    __tablename__ = "model"

    id = Column(Integer, primary_key=True, nullable=False)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("run.id", ondelete="CASCADE"), nullable=False, index=True)
    model_name = Column(String, nullable=False)
    model_onnx = Column(LargeBinary, nullable=False, info={'json_exclude': True})
    model_metadata = Column(JSON, nullable=False)
    disabled = Column(Boolean, default=False, nullable=False, server_default='false')
    time_created = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    run: Mapped["Run"] = relationship("Run", back_populates='models', passive_deletes=True)
    predictions: Mapped[Set["Prediction"]] = relationship("Prediction", back_populates='model', passive_deletes=True)

    @classmethod
    def example(cls):
        return cls(
            id=1,
            run_id='045ea36f-3a01-4ee1-a815-c32a0660a9f4',
            model_name="CNN",
            model_metadata={
                "train_time_ns": 147917200,
                "evaluate_time_ns": 6328100,
                "input_dim": 41,
                "scores": {
                    "accuracy": 0.81,
                    "precision": 0.83,
                    "recall": 0.9,
                    "conf_matrix": [
                        [14, 7],
                        [5, 32]
                    ],
                    "f1": 0.78,
                    "matthews_cor": 0.88,
                    "roc_auc": 0.9,
                    "balanced_accuracy": 0.91
                },
                "run_id": 'a39cdcad-d67f-415f-89c0-fbf5a516d7f9',
                "features": ["feature_1", "feature_2", "feature_3"],
                "feature_dict": {'feature_1': [{'paramA': 1, 'paramB': 'B-Value'}, {'paramA': 7, 'paramB': 'C-Value'}]},
            },
            time_created="2024-04-09T14:03:44.659521+02:00",
            disabled=False
        )


class Motor(TSABaseModel):
    __tablename__ = "motor"

    id = Column(Integer, autoincrement=True, primary_key=True, nullable=False)
    serial = Column(String, nullable=False, index=True)
    part = Column(String, nullable=False, index=True)
    machine = Column(String, nullable=True)
    client = Column(String, nullable=False)
    disabled = Column(Boolean, default=False, nullable=False, server_default='false')
    time_created = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    users: Mapped[Set["User"]] = relationship("User", back_populates="motors", secondary=motor_user)
    application: Mapped[Set["Application"]] = relationship("Application", back_populates="motor", passive_deletes="all")

    __table_args__ = (
        UniqueConstraint('serial', 'part', name='unique_motor'),
        CheckConstraint("serial <> ''", name='check_serial_not_empty'),
        CheckConstraint("part <> ''", name='check_part_not_empty'),
        CheckConstraint("client <> ''", name='check_client_not_empty'),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.serial.strip() == '':
            raise ValueError("Serial must not be empty.")
        if self.part.strip() == '':
            raise ValueError("Part must not be empty.")
        if self.client.strip() == '':
            raise ValueError("Client must not be empty.")

    @staticmethod
    def get_hash(serial: str, part: str):
        """
        Get a hash of the serial and part number.
        """
        return hash((str(serial).lower(), str(part).lower()))

    @classmethod
    def example(cls):
        return cls(
            id=1,
            serial="B7EF02920",
            part="ABC",
            machine="W3_L39_C92_M4",
            client="K1_0299_CH",
            disabled=False,
            time_created="2024-04-09T14:03:44.659521+02:00")


class Application(TSABaseModel):
    __tablename__ = "application"

    id = Column(Integer, autoincrement=True, primary_key=True, nullable=False)
    motor_id = mapped_column(ForeignKey("motor.id", ondelete="CASCADE"), nullable=False, index=True)
    context_code = Column(String, nullable=False)
    recipe = Column(String, nullable=False)
    disabled = Column(Boolean, default=False, nullable=False, server_default='false')
    time_created = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    motor: Mapped["Motor"] = relationship("Motor", back_populates="application", passive_deletes='all')
    cycledatas: Mapped[Set["CycleData"]] = relationship("CycleData", back_populates="application", passive_deletes='all')
    runs: Mapped[Set["Run"]] = relationship("Run", back_populates="application", passive_deletes='all')
    metrics: Mapped[Set["MeasuringPoint"]] = relationship("MeasuringPoint", back_populates="application", passive_deletes='all')

    __table_args__ = (
        UniqueConstraint('motor_id', 'context_code', 'recipe', name='unique_application'),
        CheckConstraint("context_code <> ''", name='check_contextcode_not_empty'),
        CheckConstraint("recipe <> ''", name='check_recipe_not_empty'),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.context_code.strip() == '':
            raise ValueError("Context_code must not be empty.")
        if self.recipe.strip() == '':
            raise ValueError("Recipe must not be empty.")

    @staticmethod
    def get_hash(motor, context_code: str, recipe: str):
        """
        Get a hash of the motor, context_code and recipe.
        """
        return hash((motor, str(context_code).lower(), str(recipe).lower()))

    @classmethod
    def example(cls):
        return cls(
            id=1,
            motor_id=1,
            context_code="CC_1_23",
            recipe="30edf67a69b5f8ed517735a5676b6a47e500aeacf5f955eb464279fde97dbebe",
            disabled=False,
            time_created="2024-04-09T14:03:44.659521+02:00",
        )


class CycleData(TSABaseModel):
    __tablename__ = "cycledata"

    class Classification(Enum):
        GOOD = ("good", 1)
        BAD = ("bad", 0)
        UNKNOWN = ("unknown", -1)

        @property
        def label(self) -> str:
            return self.value[0]

        @property
        def code(self) -> int:
            return self.value[1]

        @classmethod
        def from_str(cls, raw='unknown') -> "CycleData.Classification":
            raw = "unknown" if raw is None or str(raw).lower() == "none" else str(raw).lower()
            for c in cls:
                if c.label == raw:
                    return c
            return cls.UNKNOWN

        @classmethod
        def from_code(cls, raw) -> "CycleData.Classification":
            for c in cls:
                if c.code == raw:
                    return c
            return cls.UNKNOWN

    id = Column(Integer, autoincrement=True, primary_key=True, nullable=False)
    application_id = mapped_column(ForeignKey("application.id", ondelete="CASCADE"), nullable=False, index=True)
    driveConfig = Column(JSON, nullable=True)
    cycleConfig = Column(JSON, nullable=True)
    classification: Mapped[Classification] = mapped_column(SQLEnum(Classification, name="cycle_classification", values_callable=lambda x: [e.label for e in x]), nullable=False)
    testCycleCounter = Column(Integer, nullable=True)
    disabled = Column(Boolean, default=False, nullable=False, server_default='false')
    time_created = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    application: Mapped["Application"] = relationship("Application", back_populates="cycledatas", passive_deletes='all')
    runs: Mapped[Set["Run"]] = relationship("Run", secondary=cycle_run, back_populates="cycles")
    measuringpoints: Mapped[Set["MeasuringPoint"]] = relationship("MeasuringPoint", back_populates="cycle", passive_deletes='all')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataframe: pd.DataFrame = None  # can hold temporal 'metrics' dataframe

    def isTrainCycle(self):
        return self.classification and self.classification != CycleData.Classification.UNKNOWN

    @staticmethod
    def get_hash(application_id, classification, testCycleCounter, driveConfig, cycleConfig):
        driveConfig_str = json.dumps(driveConfig, sort_keys=True) if driveConfig else None
        cycleConfig_str = json.dumps(cycleConfig, sort_keys=True) if cycleConfig else None
        return hash((application_id, classification, testCycleCounter, driveConfig_str, cycleConfig_str))

    @classmethod
    def example(cls):
        return cls(
            id=1,
            application_id=1,
            driveConfig={
                "current_i": 10,
                "current_p": 24,
                "position_d": 1,
                "position_i": 2,
                "position_p": 3,
                "velocity_ff_vel_gain": 1,
                "velocity_ff_acc_gain": 24
            },
            cycleConfig={
                "homePos": 1,
                "spikeTorque": 2,
                "spikeStartTime": 3,
                "spikeLength": 4,
                "loadTorque": 5,
                "loadIncrease": 6,
                "jerkmin1": 7,
                "jerkmin2": 8,
                "jerkmin3": 9,
                "jerkmin4": 10,
                "rampDown": 11,
                "rampUp": 12,
                "timeOn": 13,
                "timePause": 14,
                "velocity": 15,
            },
            classification="good",
            testCycleCounter=1,
            disabled=False,
            time_created="2024-04-09T14:03:44.659521+02:00",
        )


class MeasuringPoint(TSABaseModel):
    __tablename__ = "measuringpoint"

    class Metric(Enum):
        """Available metric columns in MeasuringPoint table."""
        ACT_CURRENT = "act_current"
        ACT_FOLLOWING_ERROR = "act_following_error"
        ACT_POSITION = "act_position"
        ACT_TORQUE = "act_torque"
        ACT_VELOCITY = "act_velocity"
        CMD_POSITION = "cmd_position"
        CMD_VELOCITY = "cmd_velocity"
        TEMP_MOTOR = "temp_motor"
        TEMP_POWER_STAGE = "temp_power_stage"

        def get_column(self) -> Column:
            """Get the corresponding SQLAlchemy Column object for this metric."""
            return getattr(MeasuringPoint, self.value)

    id = Column(BigInteger, primary_key=True, nullable=False)
    application_id = Column(Integer, ForeignKey("application.id", ondelete="CASCADE"), nullable=False)
    cycle_id = Column(Integer, ForeignKey("cycledata.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    # Metric fields as real/(float(32))
    act_current = Column(REAL, nullable=True, info={"metric": True})
    act_following_error = Column(REAL, nullable=True, info={"metric": True})
    act_position = Column(REAL, nullable=True, info={"metric": True})
    act_torque = Column(REAL, nullable=True, info={"metric": True})
    act_velocity = Column(REAL, nullable=True, info={"metric": True})
    cmd_position = Column(REAL, nullable=True, info={"metric": True})
    cmd_velocity = Column(REAL, nullable=True, info={"metric": True})
    temp_motor = Column(REAL, nullable=True, info={"metric": True})
    temp_power_stage = Column(REAL, nullable=True, info={"metric": True})

    disabled = Column(Boolean, default=False, nullable=False, server_default='false')
    time_created = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP'))

    application: Mapped["Application"] = relationship("Application", back_populates="metrics", passive_deletes='all')
    cycle: Mapped["CycleData"] = relationship("CycleData", back_populates="measuringpoints", passive_deletes='all')

    __table_args__ = (
        Index('idx_mp_app_ts_current',
              'application_id', 'timestamp',
              postgresql_include=['cycle_id', 'act_current'],
              postgresql_where=text("act_current IS NOT NULL")),
    )

    @classmethod
    def get_metrics(cls) -> set[Column]:
        """ Get all metric columns """
        return {c for c in cls.__table__.columns if c.info.get("metric")}

    @classmethod
    def get_non_metrics(cls) -> set[Column]:
        """ Get all non-metric columns """
        return {c for c in cls.__table__.columns if not c.info.get("metric")}


class Run(TSABaseModel):
    __tablename__ = "run"

    class State(Enum):
        IDLE = "idle"
        RUNNING = "running"
        ERROR = "error"
        SUCCESS = "success"

    class Task(Enum):
        PREDICTION = "prediction"
        TRAIN = "train"

    id = Column(Uuid(as_uuid=True), primary_key=True)
    application_id: Mapped[int] = mapped_column(ForeignKey("application.id", ondelete="CASCADE"), nullable=True, index=True)  # Possible None-value at the beginning of train
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False, index=True)
    task: Mapped[Task] = mapped_column(SQLEnum(Task, name="run_task", values_callable=lambda x: [e.value for e in x]), nullable=False, index=True)
    state: Mapped[State] = mapped_column(SQLEnum(State, name="run_state", values_callable=lambda x: [e.value for e in x]), nullable=False, index=True)
    dataset_name = Column(String, nullable=True)
    cycles: Mapped[Set["CycleData"]] = relationship("CycleData", secondary=cycle_run, back_populates="runs")
    state_metadata = Column(JSON, nullable=True)
    time_last_activity = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    disabled = Column(Boolean, default=False, nullable=False, server_default='false')
    time_created = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    application: Mapped["Application"] = relationship("Application", back_populates='runs', passive_deletes='all')
    models: Mapped[Set["Model"]] = relationship("Model", back_populates='run', passive_deletes='all')
    predictions: Mapped[list["Prediction"]] = relationship("Prediction", back_populates="run", passive_deletes='all')

    __table_args__ = (
        UniqueConstraint('application_id', 'task', 'dataset_name', name='unique_dataset_name'),
    )

    @hybrid_property
    def time_elapsed(self):
        raise NotImplementedError("time_elapsed can used only in SQL-Queries.")

    @time_elapsed.expression
    def time_elapsed(cls) -> float:
        return cast(func.extract('epoch', func.now() - cls.time_created), Float)  # extract the elapsed time in seconds


class Prediction(TSABaseModel):
    __tablename__ = "prediction"

    class Metric(Enum):
        LABEL = "label"
        CONFIDENCE_1 = "confidence_1"

    id = Column(Integer, primary_key=True, nullable=False)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("run.id", ondelete="CASCADE"), nullable=False, index=True)  # One-to-one relationship with Predict-Run
    model_id: Mapped[int] = mapped_column(ForeignKey("model.id", ondelete="CASCADE"), nullable=False, index=True)
    cycle_id = Column(Integer, nullable=False)  # Not a foreign key cause ZIP-Predictions have no CycleData
    metrics = Column(String, nullable=False)  # Not binded to enum, to allow future extensions easily - but should be converted to enum later.
    value = Column(Float, nullable=False)
    run: Mapped["Run"] = relationship("Run", back_populates='predictions', passive_deletes='all')
    model: Mapped["Model"] = relationship("Model", back_populates='predictions', lazy="noload", passive_deletes='all')
    cycle: Mapped["CycleData"] = relationship("CycleData", primaryjoin="foreign(Prediction.cycle_id) == CycleData.id", viewonly='all', lazy="noload")  # Define relationship without FK constraint
