import io
import json
import zipfile
import pandas as pd
from app import Classification
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import re
from app.scripts.database_helper import parse_metric_name
from app.scripts.tsa_logging import create_logger


logger = create_logger(__name__, output_file="mind_api")


def read_zip(data: bytes) -> list[tuple[dict, dict, dict, pd.DataFrame]]:
    """
    Read a ZIP file containing multiple CSV files and process them for training or testing. Any string key or values are converted to lowercase.

    :param data: bytes containing the received ZIP data.

    :return: List of tuples containing motor info, application info, cycle info, and corresponding dataframe for each CSV file.
    """

    threshold_for_multiprocess = 500  # amount of files to start parallel processing
    max_workers = min(4, mp.cpu_count())  # Limit to 4 workers or number of CPUs

    zip_file = zipfile.ZipFile(io.BytesIO(data))
    csv_files = [name for name in zip_file.namelist() if name.lower().endswith(".csv")]

    datas = []
    cycle_ids = set()

    # small ZIP → sequential
    if len(csv_files) < threshold_for_multiprocess:
        for name in csv_files:
            try:
                raw = zip_file.read(name)
                result = _read_csv(raw)
                if not result:
                    continue

                motor_info, app_info, cycle_info, df = result
                if df.empty:
                    continue

                cycle_id = df["cycle_id"].iloc[0] if "cycle_id" in df.columns else None
                if cycle_id is None or cycle_id in cycle_ids:
                    cycle_id = max(cycle_ids, default=0) + 1
                df["cycle_id"] = cycle_id
                cycle_ids.add(cycle_id)
                datas.append((motor_info, app_info, cycle_info, df))
            except Exception as e:
                logger.exception("Error processing file %s in ZIP: %s", name, e)
                continue

        zip_file.close()
        return datas

    # large ZIP → parallel
    # Extract all csv to memory first (for parallel processing)
    file_data = [(zfile, zip_file.read(zfile)) for zfile in csv_files]
    zip_file.close()

    # Process in parallel batches
    batch_size = max(1, len(file_data) // max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit batches
        futures = []
        for i in range(0, len(file_data), batch_size):
            batch = file_data[i:i + batch_size]
            futures.append(executor.submit(_process_file_batch, batch))

        # Collect results
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                for result in batch_results:
                    if result and not result[3].empty:
                        motor_info, app_info, cycle_info, df = result

                        # Handle cycle_id conflicts in main thread
                        cycle_id = df["cycle_id"].iloc[0] if "cycle_id" in df.columns else None
                        if cycle_id is None or cycle_id in cycle_ids:
                            cycle_id = max(cycle_ids, default=0) + 1
                        df["cycle_id"] = cycle_id
                        cycle_ids.add(cycle_id)
                        datas.append((motor_info, app_info, cycle_info, df))

            except Exception:
                logger.exception("Error processing file %s in ZIP", name)
                continue

    return datas


def _process_file_batch(file_batch):
    """
    Process a batch of files in parallel
    """

    results = []
    for zfile, csv_data in file_batch:
        try:
            result = _read_csv(csv_data)
            if result:
                results.append(result)
        except Exception:
            logger.exception("Error processing batch %s", zfile)
            continue
    return results


def _read_csv(file_content: bytes) -> tuple[dict, dict, dict, pd.DataFrame]:
    """
    Read a CSV file and process it for training or testing. Any string key or values are converted to lowercase.

    :param file_content: bytes containing the received CSV data.

    :return: Data including final time series dataframe, test set and train set (only if model creation), motor serial number
    """

    file_str = file_content.decode('utf-8')
    lines = file_str.split('\n')

    if len(lines) < 4:
        return None

    # Read metadata
    motor_info = lines[0].replace(" ", "").strip()
    motor_info_dict = _convert_string_to_dic(motor_info.split("Motor:")[1])
    motor_info_dict = dict(zip(motor_info_dict, map(str, motor_info_dict.values())))  # Convert all values to string

    application_info = lines[1].replace(" ", "").strip()
    application_info_dict = _convert_string_to_dic(application_info.split("Application:")[1])
    application_info_dict = dict(zip(application_info_dict, map(str, application_info_dict.values())))  # Convert all values to string

    cycle_data_info = lines[2].replace(" ", "").strip()
    cycle_data_info_dict: dict = _convert_string_to_dic(cycle_data_info.split("CycleData:")[1])
    cycle_data_info_dict['classification'] = Classification.from_str(cycle_data_info_dict['classification']).label

    column_names = [col.lstrip('_') for col in lines[3].replace(" ", "").strip().split(',')]

    csv_data = '\n'.join(lines[4:])
    if not csv_data.strip():
        return (motor_info_dict, application_info_dict, cycle_data_info_dict, pd.DataFrame())

    # Dynamic dtype: 'value' as float32, all others as string
    dtype_dict = {col: 'float32' if col == 'value' else 'str' for col in column_names}

    df = pd.read_csv(
        io.StringIO(csv_data),
        names=column_names,
        dtype=dtype_dict,
        header=None,
        engine='c',  # Use 'c' engine for better performance
        na_values=['', 'null', 'NULL', 'nan'],
        parse_dates=False
    )
    # Rename 'time' to 'timestamp'
    df.rename(columns={'time': 'timestamp'}, inplace=True)
    # Rename field-values to internal metric names
    df['field'] = df['field'].apply(parse_metric_name)

    if df.empty:
        return (motor_info_dict, application_info_dict, cycle_data_info_dict, df)

    # Reset index to convert time from index to column
    df.reset_index(inplace=True)

    # Convert 'timestamp' column
    df['timestamp'] = _parse_datetime_mixed_efficient(df['timestamp'])

    # Add cycle_id
    cycle_id_value = cycle_data_info_dict.get('id') or cycle_data_info_dict.get('testcyclecounter')
    if cycle_id_value is not None:
        df['cycle_id'] = int(cycle_id_value)

    # Add label
    label_value = Classification.from_str(cycle_data_info_dict['classification']).code
    df['label'] = label_value

    # Sort by time if not already sorted
    if not df['timestamp'].is_monotonic_increasing:
        df.sort_values(by='timestamp', inplace=True)

    df.reset_index(names=['point_id'], inplace=True)

    return (motor_info_dict, application_info_dict, cycle_data_info_dict, df)


def _convert_string_to_dic(content: str):
    """
    Convert a string to a dictionary.
    """

    def replace_match(match):
        """
        Replace the match with a valid JSON
        """
        key, value = match.groups()
        if value == '{' or value.isdigit() or (value.replace('.', '', 1).isdigit() and value.count('.') < 2):  # Check if the value is a number
            return f'"{key.lower()}":{value}'
        return f'"{key.lower()}":"{value.strip().lower()}"'

    # Add double quotes to keys and values if necessary
    return json.loads(re.sub(r'(\w+):([^,{}]*[{]?)', replace_match, content.replace("'", '')))


def _parse_datetime_mixed_efficient(time_series):
    """Efficient parsing by detecting format pattern first"""

    if time_series.empty:
        return time_series

    # Sample first non-null value to detect format
    sample_idx = time_series.first_valid_index()
    if sample_idx is None:
        return time_series

    sample = time_series.iloc[sample_idx]

    # Detect format based on pattern
    if 'T' in sample and sample.endswith('Z'):
        # ISO 8601 format with Z suffix
        format_strs = ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ']
    elif '+' in sample or sample.count(':') >= 3:
        # Format: 2025-06-02 07:13:20.351768+00:00
        format_strs = ['%Y-%m-%d %H:%M:%S.%f%z', '%Y-%m-%d %H:%M:%S%z']
    else:
        # Fallback to auto-detection
        logger.warning(f"Datetime '{sample}' has unknown format, falling back to auto-detection.")
        return pd.to_datetime(time_series, utc=True, errors='coerce')

    # Parse entire series with detected formats
    for fmt in format_strs:
        try:
            return pd.to_datetime(time_series, format=fmt, utc=True, errors='raise')
        except ValueError:
            continue

    logger.warning(f"Failed to parse datetime '{sample}' with known formats, falling back to auto-detection.")
    return pd.to_datetime(time_series, utc=True, errors='coerce')
