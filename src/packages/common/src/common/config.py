# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

from typing import List, Tuple, Type

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


# =============================================================================
# CLASSES
# =============================================================================

# -----------------------------------------------------------------------------
# Variáveis de Ambiente
# -----------------------------------------------------------------------------


class EnvVariables(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    PROJECT_NAME: str


# -----------------------------------------------------------------------------
# Configuração Yaml
# -----------------------------------------------------------------------------


class SomeYamlConfigFile(BaseModel):
    parameter: str


class YamlSettings(BaseSettings):
    some_yaml_config: SomeYamlConfigFile
    model_config = SettingsConfigDict(yaml_file="some/path/to/file.yaml")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)

# -----------------------------------------------------------------------------
# Configuração JSON
# -----------------------------------------------------------------------------


class SomeJSONConfigFile(BaseModel):
    parameter: str


class JSONSettings(BaseSettings):
    some_json_config: SomeJSONConfigFile
    model_config = SettingsConfigDict(json_file="some/path/to/file.json")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (JsonConfigSettingsSource(settings_cls),)

# -----------------------------------------------------------------------------
# Configuração Logger
# -----------------------------------------------------------------------------


class LoggerHandlersBase(BaseModel):
    sink: str
    level: str
    format: str
    colorize: bool | None = None
    backtrace: bool | None = None
    diagnose: bool | None = None


class LoggerHandlersToDisk(LoggerHandlersBase):
    rotation: str | None = None
    retention: str | None = None


class LoggerConfig(BaseModel):
    handlers: List[LoggerHandlersBase | LoggerHandlersToDisk]
