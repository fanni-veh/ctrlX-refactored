# MIND — Motion Insights and Diagnostics

AI-driven predictive maintenance for industrial drive systems, developed by Maxon Motor AG.

MIND ingests motor sensor data, trains machine learning models to distinguish healthy from faulty operating cycles, and scores new data to predict failures before they occur.

**Version:** 1.1.0
**Python:** 3.11+
**Deployment target:** ctrlX CORE (ARM64, Snap package)

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Project structure](#project-structure)
3. [Running locally (development)](#running-locally-development)
4. [Deploying as a Snap](#deploying-as-a-snap)
5. [Configuration](#configuration)
6. [Database](#database)
7. [Key data flows](#key-data-flows)
8. [Tech stack](#tech-stack)

---

## What it does

| Feature | Description |
|---|---|
| **Import** | Upload ZIP archives of CSV motor recordings to populate the database |
| **Train** | Select an application → extract features → train HDBSCAN model → save as ONNX |
| **Predict** | Load a trained model → score new cycles as GOOD or BAD with a confidence value |
| **Live ingestion** | Optional MQTT client receives real-time sensor data from a broker |
| **Reports** | Per-run reports with confusion matrix, F1/AUC scores, and time-series plots |
| **Admin panel** | Manage users, motors, and the ML pipeline configuration |
| **Metrics** | Prometheus endpoint for operational monitoring |

---

## Project structure

```
.
├── app/                        # All application code (the only thing that runs)
│   ├── main.py                 # FastAPI app entry point — registers routes, middleware, startup/shutdown
│   ├── config.py               # All settings (reads from .env or environment variables)
│   ├── models.py               # SQLAlchemy database models (User, Motor, Application, CycleData, …)
│   ├── database.py             # Async engine and session factory
│   ├── __init__.py             # Re-exports common enums (RunState, RunTask, Classification, …)
│   ├── utils.py                # Shared helpers: template context, import/export logic, train/predict orchestration
│   ├── dto_models.py           # Pydantic request/response models
│   ├── dto_schema.py           # Additional Pydantic schemas
│   ├── update_message.py       # Progress message structure used during training/prediction
│   │
│   ├── routers/                # HTTP route handlers (all registered in main.py)
│   │   ├── api.py              # REST API: /api/* — train, predict, CRUD for motors/apps/cycles/runs
│   │   ├── htmx.py             # HTMX partials: /htmx/* — server-side HTML fragments for dynamic UI
│   │   ├── user.py             # User routes: /logout (auth is handled by ctrlX at OS level)
│   │   └── prometheus.py       # Metrics: /prometheus/metrics — scraped by Prometheus
│   │
│   ├── pipeline/               # ML pipeline
│   │   ├── pipeline.py         # Core classes: BaseClass, Trainer, Predictor
│   │   └── tools/
│   │       ├── hdbscan_clustering.py           # HDBSCAN classifier wrapper
│   │       ├── onnx_converter.py               # Convert sklearn model → ONNX; run ONNX inference
│   │       └── features/
│   │           ├── feature_set1.py             # Feature definitions (which features to extract)
│   │           ├── feature_extraction_functions.py  # The actual extraction math
│   │           ├── similarity_values.py        # Feature similarity matrix (subset)
│   │           └── similarity_values_full.py   # Feature similarity matrix (full)
│   │
│   ├── mqtt/                   # Real-time data ingestion
│   │   ├── mqtt_client.py      # Connects to MQTT broker, validates and routes incoming messages
│   │   ├── cycle_batch_collector.py  # Groups individual measurements into complete cycles
│   │   ├── sync_database.py    # Writes collected cycles to the database
│   │   ├── prometheus_metrics.py    # MQTT-specific Prometheus counters and histograms
│   │   └── rule_engine/
│   │       └── rule_engine.py  # Auto-actions triggered when a cycle arrives (e.g. alert on BAD)
│   │
│   ├── scripts/                # Business logic (called by routers and utils)
│   │   ├── auth.py             # Authentication: maps every request to the built-in admin user
│   │   ├── database_helper.py  # Optimised DB queries: bulk inserts, measurement loading
│   │   ├── report_service.py   # Assembles report data (scores, plots) for a completed run
│   │   ├── preprocessing_signal.py  # Signal preprocessing before feature extraction
│   │   ├── data_visualization.py    # Time-series and performance chart generation (Plotly)
│   │   ├── tsk_param.py        # Task parameter container passed between routers and the pipeline
│   │   ├── serialisation.py    # Makes arbitrary objects JSON-serialisable
│   │   ├── mail.py             # SMTP email notifications
│   │   ├── logging_summaries.py     # Log formatting helpers
│   │   ├── select_features_auto.py  # Automated feature selection utilities
│   │   └── tsa_logging.py      # Creates named loggers writing to rotating log files
│   │
│   ├── insight/                # Plotting utilities used by the pipeline
│   │   ├── ts_plotter.py       # Time-series plots
│   │   ├── feature_plotter.py  # Fisher score and feature similarity heatmaps
│   │   ├── model_plotter.py    # 2D/3D model geometry plots
│   │   └── df_explorer.py      # DataFrame inspection helpers
│   │
│   ├── middleware/
│   │   └── auth_middleware.py  # FastAPI middleware: validates auth on every request
│   │
│   ├── multiprocessing/
│   │   └── read_zip.py         # Reads and parses ZIP archives in a background process
│   │
│   ├── templates/              # Jinja2 HTML templates
│   │   ├── index.html          # Home / motor list
│   │   ├── train_predict.html  # Training and prediction UI (shared, switched by task_type)
│   │   ├── admin.html          # Admin panel (users, ML config)
│   │   ├── analytics.html      # Analytics dashboard
│   │   ├── report.html         # Run report page
│   │   ├── import.html         # ZIP import UI
│   │   ├── info.html           # App info / version page
│   │   └── partials/           # HTMX partial templates (loaded dynamically, no full page reload)
│   │       ├── train/
│   │       ├── predict/
│   │       └── rules/
│   │
│   ├── static/                 # Frontend assets served at /static
│   │   ├── scss/               # SCSS source (compiled to css/ at startup in dev mode)
│   │   ├── css/                # Compiled CSS (gitignored, regenerated at startup)
│   │   ├── bootstrap/          # Bootstrap CSS + JS
│   │   ├── bootstrap-icons/    # Icon font
│   │   ├── plotly.min.js       # Charting library
│   │   └── htmx.min.js         # HTMX library (drives partial HTML updates)
│   │
│   └── languages/              # i18n translation files
│       ├── en.json             # English
│       └── de.json             # German
│
├── alembic/                    # Database schema migrations
│   ├── env.py                  # Migration runner configuration
│   └── versions/               # One file per schema change, applied in order
│
├── snap/                       # Snap package (deployment to ctrlX CORE)
│   ├── snapcraft.yaml          # Build definition: ARM64, strict confinement, pre-built wheels
│   └── local/
│       ├── start.sh            # Entry point inside the snap: sets env vars, starts uvicorn on a Unix socket
│       └── package-manifest.json  # ctrlX CORE metadata for this snap
│
├── configs/
│   └── package-assets/
│       └── mind.package-manifest.json  # ctrlX CORE integration: proxy routing, menu links, scopes
│
├── wheels/                     # Pre-built Python wheels for ARM64 (used by snapcraft offline install)
│
├── model_config.yaml           # ML pipeline configuration (editable via the Admin panel at runtime)
├── pyproject.toml              # Python project metadata and dependencies (managed by Poetry)
├── requirements.txt            # Locked dependency list for standard installs
├── requirements-snap.txt       # Locked dependency list for the Snap build
├── alembic.ini                 # Alembic migration runner settings
└── package-manifest.json       # Root-level ctrlX CORE package manifest
```

---

## Running locally (development)

**Prerequisites:** Python 3.11+, Poetry

```bash
# 1. Install dependencies
poetry install

# 2. Configure the database (SQLite is easiest for local dev)
#    Create a .env file in the project root:
echo "DATABASE_URL=sqlite+aiosqlite:///./mind.db" > .env

# 3. Start the server (SASS is compiled automatically on startup)
poetry run uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

**Database migrations** (only needed if you are changing the schema):

```bash
# Apply all pending migrations
poetry run alembic upgrade head

# Create a new migration after changing app/models.py
poetry run alembic revision --autogenerate -m "describe your change"
```

---

## Deploying as a Snap

The snap targets ctrlX CORE devices (ARM64). It is cross-compiled from an amd64 machine.

```bash
# Build (requires snapcraft installed)
snapcraft

# Install on the ctrlX CORE device
snap install mind_1.1.0_arm64.snap --dangerous
```

**What happens at runtime:**

1. `snap/local/start.sh` is executed as a daemon
2. It sets environment variables (database path, log dir, socket path)
3. It starts `uvicorn app.main:app` listening on a Unix socket
4. The ctrlX CORE nginx reverse proxy forwards `https://device/mind/` to that socket

**Build artefacts** (`overlay/`, `prime/`, `stage/`, `parts/`) are gitignored — they are regenerated automatically by `snapcraft build` and should never be committed.

---

## Configuration

All settings are in [app/config.py](app/config.py) and are read from environment variables or a `.env` file.

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | Full DB connection string (SQLite or PostgreSQL) | — (required) |
| `DATABASE_HOSTNAME` / `_USERNAME` / `_PASSWORD` / `_NAME` | Alternative to DATABASE_URL for PostgreSQL | — |
| `JWT_SECRET` | Secret key for JWT tokens | `change-me-in-production` |
| `LOG_DIR` | Directory for rotating log files | `./logs` |
| `LOG_LEVEL` | Python logging level | `INFO` |
| `MQTT_BROKER` | MQTT broker hostname (optional — MQTT is disabled if not set) | — |
| `MQTT_TOPIC` | MQTT topic to subscribe to | — |
| `MQTT_PORT` | MQTT broker port | `1883` |
| `MQTT_USER` / `MQTT_PASSWORD` | MQTT credentials | — |
| `SMTP_SERVER` / `SMTP_PORT` | Email server for notifications | `smtp.maxonmotor.com:25` |

**ML pipeline settings** (also in `.env` or via the Admin panel in `model_config.yaml`):

| Variable | Description |
|---|---|
| `CORRELATION_THRESHOLD` | Max allowed correlation between features before one is dropped |
| `ENABLE_PCA_MODE` | Use PCA for dimensionality reduction (default: correlation-based) |
| `PCA_N_COMPONENTS` | Number of PCA components to keep |
| `SVD_N_COMPONENTS` | Number of SVD components to keep |

The `model_config.yaml` file contains additional pipeline knobs (validation split, plotting options, HDBSCAN hyperparameters) and can be edited live via the Admin panel without restarting the server.

---

## Database

The schema is managed with **Alembic**. The models are defined in [app/models.py](app/models.py).

### Model hierarchy

```
User  ←──(motor_user)──→  Motor
                            └── Application
                                  ├── CycleData  ←──(cycle_run)──→  Run
                                  │     └── MeasuringPoint               ├── Model
                                  └── MeasuringPoint                     └── Prediction
```

| Table | What it stores |
|---|---|
| `user` | Application users with roles (ADMIN, SERVICE, GUEST, EXHIBITION) |
| `api_key` | API keys owned by users for programmatic access |
| `motor` | Physical drive units, identified by (serial, part) |
| `application` | A specific use-case for a motor (context_code + recipe) |
| `cycledata` | One motor execution cycle with a GOOD/BAD/UNKNOWN label |
| `measuringpoint` | Raw timestamped sensor readings for a cycle (up to 9 metrics per row) |
| `run` | A training or prediction execution (IDLE → RUNNING → SUCCESS/ERROR) |
| `model` | A trained ONNX model binary + metadata, owned by a training Run |
| `prediction` | Per-cycle label + confidence values, owned by a prediction Run |
| `motor_user` | Association table linking Users to their accessible Motors |
| `cycle_run` | Association table linking CycleData to the Runs that used them |

**SQLite** is used in development and in the Snap (stored at `$SNAP_COMMON/mind.db`).
**PostgreSQL** is used for production deployments with multiple workers.

---

## Key data flows

### Training

```
POST /api/train (file or application_id)
  └── api.py: validate_and_prepare_task()
        └── Utils.tsk_train() [background task]
              └── pipeline.py: Trainer
                    1. Extract features from CycleData time-series
                    2. Dimensionality reduction (PCA / SVD / correlation)
                    3. Train HDBSCAN_Clustering
                    4. Evaluate (F1, AUC, Matthews, confusion matrix)
                    5. Convert to ONNX → save Model to DB
```

### Prediction

```
POST /api/predict (file or application_id)
  └── api.py: validate_and_prepare_task()
        └── Utils.tsk_predict() [background task]
              └── pipeline.py: Predictor
                    1. Extract same features as training
                    2. Apply same dimensionality reduction (from model metadata)
                    3. Load ONNX model → run onnxruntime inference
                    4. Save Prediction rows (label + confidence per cycle)
```

### Live MQTT ingestion

```
MQTT broker
  └── mqtt/mqtt_client.py: receive message
        └── cycle_batch_collector.py: buffer until full cycle arrives
              └── sync_database.py: insert CycleData + MeasuringPoints
                    └── rule_engine.py: trigger auto-actions if configured
```

### Export

```
GET /export/{application_id}  or  GET /export_run/{run_id}
  └── main.py: load Application + CycleData + MeasuringPoints from DB
        └── _create_zip(): one CSV file per cycle → return as ZIP download
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| HTML templating | Jinja2 + HTMX (no JavaScript framework) |
| Charts | Plotly.js |
| CSS | Bootstrap 5 + custom SCSS |
| Database ORM | SQLAlchemy 2.0 (async) |
| Migrations | Alembic |
| Databases supported | SQLite (dev/snap), PostgreSQL (production) |
| ML pipeline | scikit-learn, HDBSCAN |
| Model format | ONNX (via skl2onnx) + onnxruntime |
| Feature extraction | NumPy, SciPy, PyWavelets, statsmodels |
| MQTT client | paho-mqtt |
| Monitoring | Prometheus client |
| Settings | Pydantic Settings (reads `.env`) |
| Deployment | Snapcraft (ARM64, strict confinement) |
