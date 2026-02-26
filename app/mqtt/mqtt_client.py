from collections import defaultdict
from datetime import datetime
import re
import time
import json
import socket
import logging
from typing import Any, Dict, Optional
from cachetools import TTLCache
import paho.mqtt.client as mqtt
from sqlalchemy import select
import threading
from concurrent.futures import ThreadPoolExecutor

from app import RunTask
from app.models import Application, CycleData, MeasuringPoint, Motor
from app.mqtt.rule_engine.rule_engine import trigger_auto_action
from app.mqtt.cycle_batch_collector import CycleBatchCollector
from app.config import setting
from app.mqtt.sync_database import sync_session_manager
from app.mqtt.prometheus_metrics import (
    DB_BULK_INSERT_ROW_DURATION_SECONDS,
    DB_BULK_INSERT_RECORDS,
    MQTT_MESSAGES_RECEIVED,
    MQTT_MESSAGE_PROCESSING_TIME,
    MQTT_INVALID_MESSAGES,
    MQTT_DB_ERRORS,
    APPLICATION_CYCLES_CREATED,
    APPLICATION_MESSAGE_POINTS,
    MQTT_CONNECTIONS,
    MQTT_DISCONNECTIONS,
    ENTITY_CACHE_HITS,
    ENTITY_CACHE_MISSES,
    ENTITY_CACHE_SIZE,
    run_metrics_server
)
from app.scripts.database_helper import parse_metric_name
from app.scripts.tsa_logging import create_logger

METRIC_MAPPING = {
    "act_current": "actCurrent",
    "act_following_error": "actFollowingError",
    "act_position": "actPosition",
    "act_torque": "actTorque",
    "act_velocity": "actVelocity",
    "cmd_position": "cmdPosition",
    "cmd_velocity": "cmdVelocity",
    "temp_motor": "tempMotor",
    "temp_power_stage": "tempPowerStage",
}


class MqttClient():
    client = None
    message_counters: Dict = {}
    mqtt_client_id = f'tsa_subscriber_{socket.gethostname()}'
    logger = create_logger('mqtt_client', level=logging.DEBUG, output_file='mqtt_client', maxFileSizeMb=10, backup_count=5)
    entity_cache = TTLCache(maxsize=100, ttl=600)  # Cache for Motor, Application IDs for max 10 minutes and 100 entries
    counter = 0

    def __init__(self):
        self.logger.info("Starting MQTT Client (client_id:%s), Broker:%s, Topic:%s, PostgresDB:%s, TSA:%s",
                         self.mqtt_client_id, setting.mqtt_broker, setting.mqtt_topic,
                         setting.database_hostname, setting.tsa_hostname)

        # Initialize shared sync database session manager
        sync_session_manager.init_db()

        # Thread pool for background auto_action tasks
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="auto_action")

        # Initialize CycleBatchCollector for collecting cycles and triggering actions when batch is complete
        self.cycle_batch_collector = CycleBatchCollector(
            on_idle_callback=self._on_cycles_idle,
            logger=self.logger
        )

        # Start Prometheus metrics server in a daemon thread
        metrics_thread = threading.Thread(target=run_metrics_server, daemon=True)
        metrics_thread.start()
        self.logger.info("Prometheus metrics server started on port 9095")

        self.connect_to_broker()

    def _on_cycles_idle(self, application_id: int, task: RunTask, cycles: int) -> None:
        """
        Callback when no more cycles are received for an application after the wait time.
        Triggers auto_action in a background thread.

        Args:
            application_id: The application ID.
            task: The type of run task (TRAIN or PREDICTION).
            cycles: Number of cycles in this batch.
        """
        self.logger.info("Triggering auto_action for app_id: %d, task=%s, cycles=%d",
                         application_id, task.name.lower(), cycles)

        # Submit to thread pool for background processing
        self._executor.submit(
            trigger_auto_action,
            application_id=application_id,
            task=task,
            cycles=cycles,
        )

    def __del__(self):
        """
        Destructor to close all connections
        """
        self.close()

    def close(self):
        """
        Explicitly close all connections
        """
        if hasattr(self, '_closed') and self._closed:
            return

        try:
            # Shutdown cycle batch collector
            if hasattr(self, 'cycle_batch_collector'):
                self.cycle_batch_collector.shutdown()

            # Shutdown thread pool
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=False)

            # Note: sync_session_manager is shared, don't close it here
        except Exception as e:
            self.logger.error("Error during cleanup: %s", e)
        finally:
            self._closed = True

    def connect_to_broker(self):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.mqtt_client_id, clean_session=True)
        self.client.on_message = self.on_message
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.reconnect_delay_set(min_delay=1, max_delay=120)

        if setting.mqtt_user and setting.mqtt_password:
            self.client.username_pw_set(setting.mqtt_user, setting.mqtt_password)
        if setting.mqtt_tls:
            self.client.tls_set()

        while True:
            try:
                self.logger.info("Try to connect...")
                self.client.connect(setting.mqtt_broker, setting.mqtt_port, keepalive=60)
                self.client.loop_forever()
            except Exception as e:
                self.logger.error("Error connecting to MQTT broker: %s", e)
                self.logger.exception("MQTT broker is not available. Retrying in 5 seconds...")
                time.sleep(5)

    def on_message(self, client: mqtt.Client, userdata: object, message: mqtt.MQTTMessage) -> None:
        start = time.perf_counter()
        self.counter += 1
        # self.logger.debug("Incoming message %d", self.counter)
        with sync_session_manager.session() as db:
            try:
                # check if payload is byte_string
                if isinstance(message.payload, bytes):
                    payload_str = message.payload.decode("utf-8").replace('\x00', '')
                else:
                    payload_str = message.payload.replace('\x00', '')
                payload_str = payload_str.strip()

                if not payload_str or payload_str[0] != '{':
                    self.logger.debug("Drop invalid json | topic: %s | message: %s", message.topic, payload_str)
                    MQTT_INVALID_MESSAGES.labels(reason="invalid_json").inc()
                    return
                try:
                    payload = json.loads(payload_str)
                except json.JSONDecodeError:
                    self.logger.warning("Invalid JSON | topic: %s | payload: %s", message.topic, payload_str[:100])
                    MQTT_INVALID_MESSAGES.labels(reason="json_decode_error").inc()
                    return

                metrics = payload.get("metrics", {})
                # Validate required structure early
                if not metrics or "metadata" not in metrics or "measurement" not in metrics:
                    self.logger.warning("Invalid metrics structure | topic: %s", message.topic)
                    MQTT_INVALID_MESSAGES.labels(reason="missing_structure").inc()
                    return

                # Normalize metadata keys once
                metadata = {k.lower(): v for k, v in metrics["metadata"].items()}

                # Extract and validate required fields early
                try:
                    serial_number = str(metadata["serialnumber"]).lower()
                    part_number = str(metadata["partnumber"]).lower()
                    context_code = str(metadata["contextcode"]).lower()
                    recipe = str(metadata["recipe"]).lower()
                    client_id = str(metadata.get("clientid", 'default')).lower().replace(' ', '_')
                except KeyError as e:
                    self.logger.error("Missing required field: %s | topic: %s", e, message.topic)
                    MQTT_INVALID_MESSAGES.labels(reason="missing_field").inc()
                    return

                # Extract optional fields
                machine_id = metadata.get("machineid")
                drive_config = metadata.get("driveconfig")
                cycle_config = metadata.get("cycleconfig")
                classification_str = metadata.get("classification", "")
                test_cycle_counter = metadata.get("testcyclecounter")

                # Create cycle data object
                cycle_data = CycleData(
                    driveConfig=drive_config,
                    cycleConfig=cycle_config,
                    classification=CycleData.Classification.from_str(classification_str),
                    testCycleCounter=test_cycle_counter
                )

                # self.logger.debug("Time used to parse metadata: %.4f", time.perf_counter() - start)

                start1 = time.perf_counter()
                db_motor: Optional[Motor] = None
                db_application: Optional[Application] = None
                cache_key = f"{serial_number}:{part_number}:{context_code}:{recipe}"
                cached = self.entity_cache.get(cache_key)
                if cached:
                    ENTITY_CACHE_HITS.inc()
                    motor_id, app_id = cached
                    db_motor = db.get(Motor, motor_id)
                    db_application = db.get(Application, app_id)
                    if not db_motor or not db_application:
                        del self.entity_cache[cache_key]
                        cached = None  # Force fresh lookup
                else:
                    ENTITY_CACHE_MISSES.inc()
                    # Load the motor from DB or create new one
                    stmt = (
                        select(Motor)
                        .where(
                            Motor.serial == serial_number,
                            Motor.part == part_number
                        )
                    )
                    db_motor = db.execute(stmt).scalar_one_or_none()
                    self.logger.debug("Time used for loading motor from db: %.4f", time.perf_counter() - start1)
                    if not db_motor:
                        db_motor = Motor(
                            serial=serial_number,
                            part=part_number,
                            machine=machine_id,
                            client=client_id
                        )
                        db.add(db_motor)

                    # Filter application efficiently
                    for app in db_motor.application:
                        if app.context_code.lower() == context_code and app.recipe.lower() == recipe:
                            db_application = app
                            break

                    if not db_application:
                        db_application = Application(
                            context_code=context_code,
                            recipe=recipe
                        )
                        db_motor.application.add(db_application)

                if not db_motor or not db_application:
                    self.logger.error("Failed to get or create Motor/Application | topic: %s", message.topic)
                    MQTT_INVALID_MESSAGES.labels(reason="motor_application_error").inc()
                    raise ValueError("Failed to get or create Motor/Application")

                db_application.cycledatas.add(cycle_data)
                db.flush()  # Ensure IDs are generated

                # Parse metrics and timestamps
                measurement: Dict = metrics["measurement"]
                timestamps_raw = measurement.pop("timestamp")
                if not timestamps_raw:
                    raise ValueError("Missing timestamps")

                # Parse timestamps with error handling
                try:
                    timestamps = [datetime.fromisoformat(ts.replace("Z", "+00:00")) for ts in timestamps_raw]
                except (ValueError, AttributeError, TypeError) as e:
                    self.logger.exception("Invalid timestamp format: %s | topic: %s", e, message.topic)
                    MQTT_INVALID_MESSAGES.labels(reason="invalid_timestamp").inc()
                    raise ValueError(f"Invalid timestamp: {e}")

                point_count = len(timestamps_raw)
                all_point_metrics: Dict[int, Dict[str, Any]] = defaultdict(dict)
                for external_key, col in measurement.items():
                    metric_name = parse_metric_name(external_key)
                    if not col or len(col) != point_count:
                        self.logger.warning(f"Skipping metric {metric_name}: length mismatch ({len(col) if col else 0} vs {point_count})")
                        continue

                    for i, value in enumerate(col):
                        if value is not None and isinstance(value, (int, float)):
                            all_point_metrics[i][metric_name] = value

                # Build metrics dict for each measuring point
                # Create MeasuringPoint objects
                measuring_point_dicts = [
                    {
                        "application_id": db_application.id,
                        "cycle_id": cycle_data.id,
                        "timestamp": timestamps[i],
                        "act_current": point_metrics.get("act_current"),
                        "act_following_error": point_metrics.get("act_following_error"),
                        "act_position": point_metrics.get("act_position"),
                        "act_torque": point_metrics.get("act_torque"),
                        "act_velocity": point_metrics.get("act_velocity"),
                        "cmd_position": point_metrics.get("cmd_position"),
                        "cmd_velocity": point_metrics.get("cmd_velocity"),
                        "temp_motor": point_metrics.get("temp_motor"),
                        "temp_power_stage": point_metrics.get("temp_power_stage"),
                    }
                    for i in range(point_count)
                    if (point_metrics := all_point_metrics.get(i))
                ]

                bulk_insert_start = time.perf_counter()
                db.bulk_insert_mappings(MeasuringPoint.__mapper__, measuring_point_dicts)
                db.commit()
                duration = time.perf_counter() - bulk_insert_start
                self.entity_cache[cache_key] = (db_motor.id, db_application.id)
                record_count = len(measuring_point_dicts)
                DB_BULK_INSERT_ROW_DURATION_SECONDS.labels(table=MeasuringPoint.__tablename__).observe(duration / record_count)
                DB_BULK_INSERT_RECORDS.labels(table=MeasuringPoint.__tablename__).inc(record_count)
                # Track data
                APPLICATION_MESSAGE_POINTS.labels(application_id=db_application.id).inc(record_count)
                APPLICATION_CYCLES_CREATED.labels(application_id=db_application.id).inc()
                MQTT_MESSAGES_RECEIVED.labels(classification=cycle_data.classification.label, topic=message.topic).inc()
                ENTITY_CACHE_SIZE.set(len(self.entity_cache))

                # Record processing time
                processing_time = time.perf_counter() - start
                MQTT_MESSAGE_PROCESSING_TIME.observe(processing_time)
                self.logger.info("Received | app_id: %s | cycle_id: %s | class: %s | points: %d | processing_time: %.4f s",
                                 db_application.id, cycle_data.id, cycle_data.classification.label, point_count, processing_time)

                # Notify cycle batch collector - will trigger auto_action after idle period
                self.cycle_batch_collector.on_cycle_received(db_application.id, cycle_data.classification)
            except Exception:
                db.rollback()
                MQTT_DB_ERRORS.inc()
                self.logger.exception("Error processing message | client: %s | topic: %s | payload: %s", client, message.topic, message.payload[:500] if message.payload else None)

    def on_connect(self, client, userdata, flags, reason_code, properties):
        self.logger.info("MQTT Connected with result code %s", str(reason_code))
        MQTT_CONNECTIONS.inc()
        self.client.subscribe(setting.mqtt_topic, qos=2)  # Subscribe with QoS 2 (exactly once)

    def on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self.logger.warning("MQTT disconnected with result code %s", str(reason_code))
        MQTT_DISCONNECTIONS.inc()


class MqttValidator():
    """
    MQTT Message Validator
    """

    # Pattern for valid ISO 8601 timestamps
    # 2021-08-25T12:00:00.123456Z
    # 2021-08-25T12:00:00Z
    # 2021-08-25T12:00:00+02:00
    timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?(Z|[+-]\d{2}:\d{2})$')

    @staticmethod
    def validate_mqtt_json(message):
        validation_errors = []
        try:
            # check if payload is byte_string
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            # remove any nullbytes
            clean_message = message.replace('\x00', '')
            payload = json.loads(clean_message)
            if "timestamp" not in payload.keys():
                validation_errors.append("timestamp key is missing")
            if "metrics" not in payload.keys():
                validation_errors.append("metrics key is missing")
            else:
                metadata = payload.get("metrics").get("metadata")
                measurement = payload.get("metrics").get("measurement")
                if metadata is None:
                    validation_errors.append("metadata is not a dictionary")
                if measurement is None:
                    validation_errors.append("measurement is not a dictionary")
                if metadata and measurement:
                    # Normalize metadata keys once
                    metadata = {k.lower(): v for k, v in metadata.items()}

                    # cycles
                    if 'testcyclecounter' not in metadata:
                        validation_errors.append("testCycleCounter is missing")

                    # motor
                    if 'serialnumber' not in metadata:
                        validation_errors.append("serialNumber is missing")
                    if 'partnumber' not in metadata:
                        validation_errors.append("partNumber is missing")
                    if 'clientid' not in metadata:
                        validation_errors.append("partNumber is missing")

                    # application
                    if 'contextcode' not in metadata:
                        validation_errors.append("contextCode is missing")
                    if 'recipe' not in metadata:
                        validation_errors.append("recipe is missing")

                    # metrics
                    metrics = measurement
                    timestamps = metrics.pop("timestamp")
                    if timestamps is None or any(not timestamp or not MqttValidator.timestamp_pattern.match(timestamp) for timestamp in timestamps):
                        validation_errors.append("timestamp empty or missing or wrong pattern, any index of the array must not be empty")

                    actCurrents = metrics.pop("actCurrent")
                    if actCurrents is None or any(not actCurrent for actCurrent in actCurrents):
                        validation_errors.append("actCurrent empty or missing, any index of the array must not be empty")
                    if len(actCurrents) != len(timestamps):
                        validation_errors.append("actCurrent and timestamp arrays must have the same length")
            if validation_errors:
                return validation_errors
            return True
        except Exception as e:
            return e


# start mqtt client
if __name__ == "__main__":
    MqttClient()
