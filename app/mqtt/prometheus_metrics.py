"""
Prometheus Metrics for MQTT Client
Centralized metric definitions for monitoring MQTT message processing, database operations, and system health.
"""
import os
import shutil
from prometheus_client import Histogram, Counter, Gauge, CollectorRegistry, multiprocess, start_http_server

# Setup Prometheus Registry for multiprocess
prom_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus_metrics")
if prom_dir:
    shutil.rmtree(prom_dir, ignore_errors=True)
    os.makedirs(prom_dir, exist_ok=True)
else:
    raise ValueError("PROMETHEUS_MULTIPROC_DIR has no value!")

REGISTRY = CollectorRegistry()
multiprocess.MultiProcessCollector(REGISTRY)

# ==============================================================================
# Database Metrics
# ==============================================================================

DB_BULK_INSERT_ROW_DURATION_SECONDS = Histogram(
    "db_bulk_insert_row_duration_seconds",
    "Duration of insert operations per record",
    ["table"],
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5),
    registry=REGISTRY
)

DB_BULK_INSERT_RECORDS = Counter(
    "db_bulk_insert_records_total",
    "Total records inserted",
    ["table"],
    registry=REGISTRY
)

# ==============================================================================
# MQTT Message Metrics
# ==============================================================================

MQTT_MESSAGES_RECEIVED = Counter(
    "mqtt_messages_received_total",
    "Total MQTT messages received",
    ["classification", "topic"],
    registry=REGISTRY
)

MQTT_MESSAGE_PROCESSING_TIME = Histogram(
    "mqtt_message_processing_duration_seconds",
    "Time taken to process MQTT message",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY
)

MQTT_INVALID_MESSAGES = Counter(
    "mqtt_invalid_messages_total",
    "Total invalid MQTT messages",
    ["reason"],
    registry=REGISTRY
)

MQTT_DB_ERRORS = Counter(
    "mqtt_db_errors_total",
    "Total database errors during MQTT message processing",
    registry=REGISTRY
)

# ==============================================================================
# Application Metrics
# ==============================================================================

APPLICATION_CYCLES_CREATED = Counter(
    "application_cycles_created_total",
    "Total cycles created per application",
    ["application_id"],
    registry=REGISTRY
)

APPLICATION_MESSAGE_POINTS = Counter(
    "application_message_points_total",
    "Total data points received per application",
    ["application_id"],
    registry=REGISTRY
)

# ==============================================================================
# Connection Metrics
# ==============================================================================

MQTT_CONNECTIONS = Counter(
    "mqtt_connections_total",
    "Total MQTT connection attempts",
    registry=REGISTRY
)

MQTT_DISCONNECTIONS = Counter(
    "mqtt_disconnections_total",
    "Total MQTT disconnections",
    registry=REGISTRY
)

# ==============================================================================
# Cache Metrics
# ==============================================================================

ENTITY_CACHE_HITS = Counter(
    "entity_cache_hits_total",
    "Total entity cache hits",
    registry=REGISTRY
)

ENTITY_CACHE_MISSES = Counter(
    "entity_cache_misses_total",
    "Total entity cache misses",
    registry=REGISTRY
)

ENTITY_CACHE_SIZE = Gauge(
    "entity_cache_objects",
    "Current number of objects in the entity cache",
    registry=REGISTRY
)


def run_metrics_server():
    start_http_server(port=9095, registry=REGISTRY)
