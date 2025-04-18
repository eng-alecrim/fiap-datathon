# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import sys
from typing import Optional

from loguru import logger

from .utils import get_project_root
from .config import LoggerConfig, JsonConfigSettingsSource, BaseSettings

# =============================================================================
# FUNÇÕES
# =============================================================================

def configure_logging(
    project_name: str,
    log_to_file: Optional[bool] = False,
    log_level: str = "INFO",
) -> None:
    """
    Configure the logger for the project.

    Args:
        log_file: Name of the log file (if None, a default name will be used)
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate the log file
        retention: How long to keep the log files
    """
    # Remove default handlers
    logger.remove()

    # Add file handler if requested
    if log_to_file:
        log_config_file = (
            get_project_root(project_name=project_name) / "config/log_config.json"
        )
        logger_config = LoggerConfig(
            **JsonConfigSettingsSource(
                settings_cls=BaseSettings,
                json_file=log_config_file,
                json_file_encoding="utf-8",
            ).init_kwargs
        )
        logger.configure(**logger_config.model_dump())

    # Add stdout handler
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    logger.info(f"Logging configured at level {log_level}")
