# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import json
from typing import Any, Dict, Union
from pathlib import Path

# =============================================================================
# FUNÇÕES
# =============================================================================

# -----------------------------------------------------------------------------
#TODO
# -----------------------------------------------------------------------------


def ensure_directory(directory_path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory to create

    Returns:
        Path object pointing to the created/existing directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
#TODO
# -----------------------------------------------------------------------------


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing the JSON data
    """
    with open(file_path, "r") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Retorna o diretório do projeto
# -----------------------------------------------------------------------------


def save_json(
    data: Dict[str, Any], file_path: Union[str, Path], indent: int = 4
) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        file_path: Path to save the JSON file
        indent: JSON indentation level
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent)


# -----------------------------------------------------------------------------
# TODO
# -----------------------------------------------------------------------------


def get_project_root(project_name: str, current_dir: Path = Path.cwd()) -> Path:
    """
    Get the root directory of the project.

    Returns:
        Path object pointing to the project root
    """
    if current_dir.name == project_name:
        return current_dir
    return get_project_root(project_name, current_dir.parent)
