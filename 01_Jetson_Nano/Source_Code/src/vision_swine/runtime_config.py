"""Load and validate Vision Swine settings from vision_swine_v2.yaml.

The YAML file is the source of truth for detection, tracking, and counting.
Set VISION_SWINE_CONFIG to override automatic YAML discovery.
"""

# ======================================================
# Import Libraries
# ======================================================

import os
from functools import lru_cache
from pathlib import Path
from typing import Union


# ======================================================
# Configuration File Settings
# ======================================================

CONFIG_FILE_NAME = "vision_swine_v2.yaml"
CONFIG_ENV_VARIABLE = "VISION_SWINE_CONFIG"

REQUIRED_SECTIONS = (
    "model",
    "camera",
    "inference",
    "tracking",
    "counting",
    "stream"
)


# ======================================================
# Find YAML Configuration File
# ======================================================

def find_config_file() -> Path:
    """Find the YAML file, optionally using an environment override."""

    override = os.environ.get(CONFIG_ENV_VARIABLE)

    # Use the explicitly configured YAML path, if provided.
    if override:

        config_path = Path(override).expanduser().resolve()

        if not config_path.is_file():
            raise FileNotFoundError(
                f"{CONFIG_ENV_VARIABLE} not found: {config_path}"
            )

        return config_path

    # Search beside this module, parent folders, and the current directory.
    current_file = Path(__file__).resolve()
    search_folders = (
        current_file.parent,
        *current_file.parents,
        Path.cwd()
    )

    for folder in search_folders:

        candidate_paths = (
            folder / "config" / CONFIG_FILE_NAME,
            folder / CONFIG_FILE_NAME
        )

        for config_path in candidate_paths:

            if config_path.is_file():
                return config_path.resolve()

    raise FileNotFoundError(
        f"{CONFIG_FILE_NAME} not found. "
        "Place it in <project>/config/ or set "
        f"{CONFIG_ENV_VARIABLE} to its absolute path."
    )


# ======================================================
# Get Project Root Directory
# ======================================================

def project_root(config_path: Path) -> Path:
    """Resolve the project root from the YAML file location."""

    config_directory = config_path.parent

    if config_directory.name == "config":
        return config_directory.parent

    # Support the board layout: <project>/src/vision_swine/*.yaml.
    if (
        config_directory.name == "vision_swine"
        and config_directory.parent.name == "src"
    ):
        return config_directory.parent.parent

    return config_directory


# ======================================================
# Resolve Asset Path
# ======================================================

def resolve_asset(path: Union[str, Path], config_path: Path) -> Path:
    """Resolve an absolute path or a path relative to the project root."""

    candidate = Path(path).expanduser()

    if candidate.is_absolute():
        return candidate.resolve()

    return (
        project_root(config_path) / candidate
    ).resolve()


# ======================================================
# Read YAML Configuration
# ======================================================

def read_yaml_config(config_path: Path) -> dict:
    """Read YAML data and confirm that the top level is a mapping."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required: python -m pip install pyyaml"
        ) from exc

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a YAML mapping in {config_path}"
        )

    return config


# ======================================================
# Validate YAML Configuration
# ======================================================

def validate_runtime_config(config: dict) -> None:
    """Validate required sections and tracking/inference settings."""

    # Check all required configuration sections.
    for section in REQUIRED_SECTIONS:

        if not isinstance(config.get(section), dict):
            raise ValueError(
                f"Missing YAML configuration section: {section}"
            )

    # A TensorRT engine path must be specified in the YAML file.
    if not config["model"].get("path"):
        raise ValueError(
            "model.path is required in vision_swine_v2.yaml"
        )

    # Tracking is expected to use ByteTrack.
    tracking_method = str(
        config["tracking"].get("method", "")
    ).lower()

    if tracking_method != "bytetrack":
        raise ValueError(
            "tracking.method must be bytetrack"
        )

    # Process every received inference frame.
    process_every = int(
        config["inference"].get("process_every", 1)
    )

    if process_every != 1:
        raise ValueError(
            "inference.process_every must be 1 "
            "for continuous inference"
        )


# ======================================================
# Prepare Runtime Paths
# ======================================================

def prepare_runtime_paths(config: dict, config_path: Path) -> dict:
    """Attach resolved model and gate paths for the existing callers."""

    config["_config_path"] = config_path
    config["_project_root"] = project_root(config_path)

    config["model"]["resolved_path"] = resolve_asset(
        config["model"]["path"],
        config_path
    )

    config["counting"]["resolved_gate_config"] = resolve_asset(
        config["counting"].get(
            "gate_config",
            "config/gate.json"
        ),
        config_path
    )

    return config


# ======================================================
# Load Runtime Configuration
# ======================================================

@lru_cache(maxsize=1)
def load_runtime_config() -> dict:
    """Load, validate, and cache the Vision Swine runtime settings."""

    config_path = find_config_file()

    config = read_yaml_config(config_path)

    validate_runtime_config(config)

    return prepare_runtime_paths(
        config,
        config_path
    )
