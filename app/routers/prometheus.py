import ipaddress
import shutil
from fastapi import APIRouter, Response, HTTPException, Request
from prometheus_client import generate_latest, CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client import multiprocess
import psutil
from threading import Thread
import time
import gc
import logging
import os
from starlette.middleware.base import BaseHTTPMiddleware

ALLOWED_NETWORKS = [
    ipaddress.ip_network("172.16.0.0/12"),  # Docker default range
    ipaddress.ip_network("127.0.0.0/8"),     # Localhost
]

prom_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")

if prom_dir:
    shutil.rmtree(prom_dir, ignore_errors=True)  # Clear dir
    os.makedirs(prom_dir, exist_ok=True)  # Create dir
else:
    raise ValueError("PROMETHEUS_MULTIPROC_DIR has no value!")

pid = os.getpid()

# Overwrite prometheus default registry (REGISTRY case-sensitiv!)
REGISTRY = CollectorRegistry()

# Register MultiProcessCollector for multiple processes
multiprocess.MultiProcessCollector(REGISTRY)

router = APIRouter(tags=["Prometheus"])

# Set PID als label to have each worker separate
CPU_USAGE = Gauge(
    "cpu_usage_percent",
    "CPU usage percentage",
    ["pid"])
RAM_USAGE = Gauge(
    "ram_usage_bytes",
    "RAM usage in bytes",
    ["pid"])
GC_UNCOLLECTABLE = Gauge(
    'python_gc_objects_uncollectable_total_2',
    'Number of unreachable objects found by GC')
GC_OBJECTS_COLLECTED = Gauge(
    'python_gc_objects_collected_total_2',
    'Total objects collected by GC',
    ["generation"])

# Initialize FastAPI-specific metrics
REQUEST_COUNT = Counter(
    "fastapi_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "http_status", "pid"])
EXCEPTIONS = Counter(
    "fastapi_exceptions_total",
    "Total count of exceptions raised by path and exception type",
    ["method", "path", "exception_type"]
)
REQUESTS_IN_PROGRESS = Gauge(
    "fastapi_requests_in_progress",
    "Gauge of requests by method and path currently being processed",
    ["method", "path"]
)
REQUESTS_PROCESSING_TIME = Histogram(
    "fastapi_requests_duration_seconds",
    "Histogram of requests processing time by path (in seconds)",
    ["method", "path"]
)
RESPONSES = Counter(
    "fastapi_responses_total",
    "Total count of responses by method, path and status codes.",
    ["method", "path", "status_code"]
)

# Suppress logs for Prometheus calls on /metrics
logging.getLogger("uvicorn.access").addFilter(
    lambda record: "GET /metrics" not in record.getMessage()
)


@router.get("/metrics")
async def metrics(request: Request):
    """
    Define FastAPI-Route to get the metrics
    """
    if not is_allowed_ip(request.client.host):
        raise HTTPException(status_code=403, detail="Forbidden")
    return Response(generate_latest(), media_type="text/plain")


def is_allowed_ip(client_ip: str) -> bool:
    try:
        ip = ipaddress.ip_address(client_ip)
        return any(ip in network for network in ALLOWED_NETWORKS)
    except ValueError:
        return False


def collect_resource_metrics():
    while True:
        # Set metrics with current values
        CPU_USAGE.labels(pid=pid).set(psutil.cpu_percent(interval=1))  # 1-second sample interval
        RAM_USAGE.labels(pid=pid).set(psutil.virtual_memory().used)   # RAM used in bytes

        # Get GC stats
        gc_stats = gc.get_stats()
        uncollectable = len(gc.garbage)  # Count uncollectable objects
        GC_UNCOLLECTABLE.set(uncollectable)

        for gen, stats in enumerate(gc_stats):
            collected = stats.get("collected", 0)
            GC_OBJECTS_COLLECTED.labels(generation=gen).set(collected)

        time.sleep(7)  # Update every 7 seconds


resource_thread = Thread(target=collect_resource_metrics, daemon=True)
resource_thread.start()


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/metrics" or request.url.path.startswith("/htmx"):
            return await call_next(request)

        method = request.method
        path = request.url.path

        REQUESTS_IN_PROGRESS.labels(method=method, path=path).inc()

        start_time = time.time()
        response = await call_next(request)
        after_time = time.time()

        if response.status_code >= 400:
            EXCEPTIONS.labels(method=method, path=path, exception_type=response.status_code).inc()

        REQUESTS_PROCESSING_TIME.labels(method=method, path=path).observe(after_time - start_time)

        RESPONSES.labels(method=method, path=path, status_code=response.status_code).inc()
        REQUESTS_IN_PROGRESS.labels(method=method, path=path).dec()
        REQUEST_COUNT.labels(method=request.method, path=request.url.path, http_status=response.status_code, pid=pid).inc()

        return response
