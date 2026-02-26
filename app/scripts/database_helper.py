import datetime
import re
from typing import List, Optional
import pandas as pd
import time
import logging
from sqlalchemy import insert, select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from functools import lru_cache

from app import models


CAMEL_TO_SNAKE_PATTERN = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')


def _melt_measuring_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt measuring dataframe to long format.

    :param df: DataFrame with metrics per column
    :return: DataFrame in long format with columns ['timestamp', 'cycle_id', 'field', 'value']
    """
    value_vars = [m.name for m in models.MeasuringPoint.get_metrics() if m.name in df.columns]
    non_metrics = [m.name for m in models.MeasuringPoint.get_non_metrics() if m.name in df.columns]

    df = df.melt(
        id_vars=non_metrics,
        value_vars=value_vars,
        var_name="field",
        value_name="value",
    )
    # Convert field to category for memory efficiency
    df['field'] = df['field'].astype('category')
    # Ensure value column is numeric (float) - PostgreSQL REAL becomes float
    if 'value' in df.columns and not df.empty:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df


def _pivot_measuring_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot measuring dataframe to wide format.

    :param df: DataFrame in long format with columns ['timestamp', 'cycle_id', 'field', 'value']
    :return: DataFrame with metrics per column
    """

    table_columns = models.MeasuringPoint.__table__.columns
    cols = {c.name: c.info.get("metric") for c in table_columns if c.name in df.columns}
    non_metrics = [name for name, is_metric in cols.items() if not is_metric]

    return df.pivot_table(index=non_metrics, columns="field", values="value").reset_index()


@lru_cache(maxsize=20)
def parse_metric_name(external_name: str) -> str:
    """
    Convert external metric name (camelCase) to internal standardized name (snake_case).
    Args:
        external_name: External metric name in camelCase (e.g., "actCurrent")
    Returns:
        Internal metric name in snake_case (e.g., "act_current") or original if unknown.
    """
    # Convert camelCase to snake_case
    return CAMEL_TO_SNAKE_PATTERN.sub('_', external_name).lower()


def _convert_df_to_dict(application_id: int, cycle_id: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dataframe from csv to dict.

    :param df: DataFrame in long format with columns ['timestamp', 'field', 'value']
    :return: DataFrame with metrics per column
    """

    # Validate required columns
    required_cols = ['timestamp', 'field', 'value']
    if not all(col in df.columns for col in required_cols):
        missing = set(required_cols) - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Remove null values
    df_clean = df[df['value'].notna()].copy()

    records = [
        {
            'application_id': application_id,
            'cycle_id': cycle_id,
            'timestamp': ts,
            'metrics': dict(zip(group['field'], group['value']))
        }
        for ts, group in df_clean.groupby('timestamp', sort=False)  # group by timestamp
    ]

    return records


async def load_measurements_by_app(db: AsyncSession,
                                   application_id: int,
                                   start_time: Optional[datetime.datetime] = None,
                                   end_time: Optional[datetime.datetime] = None,
                                   preflight: bool = False,
                                   cycle_ids: Optional[set[int]] = None,
                                   metrics: Optional[List[models.MeasuringPoint.Metric]] = None,
                                   logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Load measurements for a given application.

    Args:
        db: Database session.
        application_id: Application ID to filter by
        start_time: Start timestamp
        end_time: End timestamp
        preflight: If True, return only first record per cycle
        cycle_ids: Optional set of specific cycle IDs to load
        metrics: List of metrics to load. Empty or None loads all available metrics.
        logger: Optional logger

    Returns:
        DataFrame with optimized dtypes:
        - cycle_id: int
        - timestamp: datetime64[ns, UTC]
        - field: category
        - value: float
    """
    if application_id is None:
        raise ValueError("application_id must be provided.")

    if not metrics:
        metrics = [m for m in models.MeasuringPoint.Metric]

    # Build SELECT clause with requested metrics
    select_columns = [
        models.MeasuringPoint.cycle_id.label('cycle_id'),
        models.MeasuringPoint.timestamp
    ]
    for metric in metrics if metrics else []:
        select_columns.append(metric.get_column())

    stmt = (
        select(*select_columns)
        .join(models.CycleData)
        .where(
            models.MeasuringPoint.application_id == application_id,
            ~models.CycleData.disabled
        )
    )

    if cycle_ids:
        stmt = stmt.where(models.MeasuringPoint.cycle_id.in_(cycle_ids))
    if start_time:
        stmt = stmt.where(models.MeasuringPoint.timestamp >= start_time)
    if end_time:
        stmt = stmt.where(models.MeasuringPoint.timestamp <= end_time)
    if preflight:
        stmt = stmt.limit(1)

    stmt = stmt.order_by(models.MeasuringPoint.timestamp.asc())

    start_perf = time.perf_counter()
    if preflight or (cycle_ids and len(cycle_ids) < 100):
        # Small query - fetch all at once
        result = await db.execute(stmt)
        rows = result.all()
        df = pd.DataFrame(rows)
    else:
        # Large query - use streaming with yield_per
        batch_size = 50000
        rows = []
        stream = await db.stream(stmt.execution_options(yield_per=batch_size))
        async for partition in stream.partitions(batch_size):
            rows.extend(partition)

        df = pd.DataFrame(rows) if rows else pd.DataFrame()

    df = _melt_measuring_df(df)

    # drop rows with null values in 'value' column
    df = df[df['value'].notna()]

    end_perf = time.perf_counter()
    if logger:
        logger.info(
            "Time taken for load_measurements_by_app(app-id:%s, metrics:%s, start-time:%s, preflight:%s, df-shape:%s): %.4f s",
            application_id, [m.value for m in metrics], start_time, preflight, df.shape, end_perf - start_perf
        )
    return df


async def add_cycles(
        db: AsyncSession,
        cycles: List[models.CycleData],
        max_batch_size: int = 2520,  # Postgres allow max 32768 parameters per query; with 13 metrics -> 2520 rows
        logger: Optional[logging.Logger] = None) -> int:
    """
    Add cycle data to database using SQLAlchemy bulk insert

    Args:
        db: AsyncSession for database operations
        cycles: List of CycleData models with dataframe attached
        max_batch_size: Maximum rows per bulk insert batch
        logger: Optional logger

    Returns:
        Total number of points written
    """
    start = time.perf_counter()
    total_points_written = 0

    # Prepare records from cycles
    measuringpoints = []
    for cycle in cycles:
        application_id = cycle.application_id
        cycle_id = cycle.id
        df_pivot = _pivot_measuring_df(cycle.dataframe)
        # Add constant columns
        df_pivot['application_id'] = application_id
        df_pivot['cycle_id'] = cycle_id
        # Convert to list of dicts directly
        measuringpoints.extend(df_pivot.to_dict(orient='records'))

    # Write in batches using SQLAlchemy Core bulk insert
    for i in range(0, len(measuringpoints), max_batch_size):
        batch = measuringpoints[i:i + max_batch_size]
        start_batch = time.perf_counter()
        stmt = insert(models.MeasuringPoint).values(batch)
        await db.execute(stmt)
        await db.commit()
        count = len(batch)
        total_points_written += count
        if logger:
            logger.debug("Wrote batch of %s points to TimescaleDB in %.4f s", count, time.perf_counter() - start_batch)

    end = time.perf_counter()
    if logger:
        logger.info("%s points written into TimescaleDB in %.4f s", total_points_written, end - start)

    return total_points_written


async def delete_cycles(db: AsyncSession,
                        application_id: Optional[int] = None,
                        cycledata_id: Optional[int] = None,
                        logger: Optional[logging.Logger] = None):
    """
    Delete cycle data from TimescaleDB using SQLAlchemy

    Args:
        db: AsyncSession for database operations
        application_id: Optional application ID to filter by
        cycledata_id: Optional cycledata (metadata) ID to filter by
        logger: Optional logger

    At least one of application_id or cycledata_id must be provided.
    """
    if not application_id and not cycledata_id:
        raise ValueError("At least one of application_id or cycledata_id must be provided.")

    # Build delete statement with conditions
    stmt = delete(models.MeasuringPoint)

    if application_id:
        stmt = stmt.where(models.MeasuringPoint.application_id == application_id)

    if cycledata_id:
        stmt = stmt.where(models.MeasuringPoint.cycle_id == cycledata_id)

    result = await db.execute(stmt)
    await db.commit()

    if logger:
        # Sanitize None values for logging
        application_id = "null" if application_id is None else str(int(application_id))
        cycledata_id = "null" if cycledata_id is None else str(int(cycledata_id))
        logger.info("Deleted %s cycles with application_id=%s and cycledata_id=%s",
                    result.rowcount, application_id, cycledata_id)
