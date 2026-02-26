
import os
import tempfile
from typing import Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Setting(BaseSettings):
    """
    Configuration settings for the application.

    Attributes:
        database_url (str): Full database URL. Takes priority over individual params.
            SQLite example:     sqlite+aiosqlite:////path/to/mind.db
            PostgreSQL example: postgresql+asyncpg://user:pass@host:5432/dbname
        database_hostname (str): PostgreSQL hostname (used only if database_url is not set).
        database_port (str): PostgreSQL port (used only if database_url is not set).
        database_password (str): PostgreSQL password (used only if database_url is not set).
        database_name (str): PostgreSQL database name (used only if database_url is not set).
        database_username (str): PostgreSQL username (used only if database_url is not set).
        jwt_secret (str): The secret key for the application.
        algorithm (str): The algorithm used for encryption.
        log_dir (str): The directory where logs are stored.
        access_token_expire_minutes (int): The expiration time for access tokens.
        plot_features (bool): Whether to plot the features.
        correlation_threshold (float): The threshold for correlation.
        enable_DL (bool): Whether to enable deep learning.
        plot_corr_heat_map (bool): Whether to plot the correlation heat map.


    Notes:
        - The configuration values are loaded in the following order of priority:
          1. Environment variables
          2. Values from the `.env` file
          3. Default values specified in the class attributes

          `.env.prod` takes priority over `.env` when loading values from the environment.
    """

    # Single URL (preferred). If set, individual PostgreSQL params below are ignored.
    database_url: Optional[str] = None

    # Individual PostgreSQL params — used to build database_url when database_url is not set.
    database_hostname: Optional[str] = None
    database_port: str = '5432'
    database_password: Optional[str] = None
    database_name: Optional[str] = None
    database_username: Optional[str] = None

    @model_validator(mode='after')
    def build_database_url(self) -> 'Setting':
        if not self.database_url:
            if all([self.database_hostname, self.database_password,
                    self.database_name, self.database_username]):
                self.database_url = (
                    f"postgresql+asyncpg://{self.database_username}:{self.database_password}"
                    f"@{self.database_hostname}:{self.database_port}/{self.database_name}"
                )
            else:
                raise ValueError(
                    "Database not configured. Set DATABASE_URL (e.g. sqlite+aiosqlite:////path/to/mind.db), "
                    "or set DATABASE_HOSTNAME, DATABASE_USERNAME, DATABASE_PASSWORD, and DATABASE_NAME."
                )
        return self
    jwt_secret: str
    algorithm: str = 'HS256'
    predict_all_cycles: bool = False
    smtp_server: str = "smtp.maxonmotor.com"
    smtp_port: int = 25
    smtp_from: str = "MIND.Noreply@maxongroup.com"

    access_token_expire_minutes: int = 60

    log_dir: str = './logs'
    log_level: str = 'INFO'
    plot_features: bool = False
    plot_corr_heat_map: bool = False
    plot_corr_labels: bool = False
    enable_corr_labels: bool = False

    enable_DL: bool = False
    correlation_threshold: float = 0.9
    enable_pca_mode: bool = False
    pca_n_components: float = 10
    svd_n_components: int = 10
    svd_threshold: float = 0.3
    pca_feature_output_by_order: list[int] = [30, 10, 10]
    svd_feature_output_by_order: list[int] = [30, 5, 5]
    corr_feature_output_by_order: list[int] = [66, 66, 66]

    mqtt_broker: Optional[str] = None
    mqtt_topic: Optional[str] = None
    mqtt_port: Optional[int] = 1883
    mqtt_user: Optional[str] = None
    mqtt_password: Optional[str] = None
    mqtt_tls: Optional[bool] = False
    mqtt_service_username: Optional[str] = None
    mqtt_service_password: Optional[str] = None
    tsa_hostname: Optional[str] = None

    prometheus_multiproc_dir: str = os.path.join(tempfile.gettempdir(), "prometheus_metrics")
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings():
    return Setting()


setting = get_settings()
