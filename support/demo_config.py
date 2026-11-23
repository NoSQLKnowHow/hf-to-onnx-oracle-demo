"""Load shared configuration for the embedding demo and comparison."""
import os
import re
import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.toml"


def load_config():
    try:
        with CONFIG_PATH.open("rb") as source:
            config = tomllib.load(source)
    except FileNotFoundError:
        raise SystemExit(f"Configuration not found: {CONFIG_PATH}. Restore config/config.toml from the repository and edit it.")
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f"Invalid configuration in {CONFIG_PATH}: {exc}")
    try:
        database = config["database"]
        model = config["model"]
        for key in ("admin", "user", "dsn", "wallet_dir"):
            if not isinstance(database[key], str) or not database[key].strip():
                raise ValueError(f"database.{key} must be a nonempty string")
        for key in ("id", "name", "output_dir", "query"):
            if not isinstance(model[key], str) or not model[key].strip():
                raise ValueError(f"model.{key} must be a nonempty string")
        for key in ("quantize", "external_data", "do_lower_case", "use_float16"):
            if not isinstance(model[key], bool):
                raise ValueError(f"model.{key} must be true or false")
        for key in ("max_tokens", "dimensions"):
            if type(model[key]) is not int or model[key] <= 0:
                raise ValueError(f"model.{key} must be a positive integer")
        metrics = model["distance_metrics"]
        if not isinstance(metrics, list) or not metrics or not all(isinstance(x, str) and x for x in metrics):
            raise ValueError("model.distance_metrics must be a nonempty array of strings")
        if not isinstance(model["languages"], list) or not all(isinstance(x, str) and x for x in model["languages"]):
            raise ValueError("model.languages must be an array of nonempty strings")
        if not isinstance(model["post_processors"], list) or not all(isinstance(x, dict) and isinstance(x.get("name"), str) for x in model["post_processors"]):
            raise ValueError("model.post_processors must be an array of processor tables")
        if not isinstance(config["comparison"]["trust_remote_code"], bool):
            raise ValueError("comparison.trust_remote_code must be true or false")
        validate_model_name(model["name"] + ("_INT8" if model["quantize"] else ""))
    except KeyError as exc:
        raise SystemExit(f"Missing configuration setting: {exc}")
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"Invalid configuration: {exc}")
    return config


def validate_model_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Z][A-Z0-9_$#]{0,127}", name):
        raise ValueError("model names must be simple uppercase Oracle identifiers")
    return name


def configure_wallet(config, *, check_files=False):
    """Point Oracle Net to the configured wallet directory for this process."""
    path = Path(config["database"]["wallet_dir"]).expanduser()
    if not path.is_absolute():
        raise SystemExit("database.wallet_dir in config/config.toml must be an absolute path.")
    if check_files and not (path / "tnsnames.ora").is_file():
        raise SystemExit(f"Cannot find tnsnames.ora in wallet directory: {path}")
    os.environ["TNS_ADMIN"] = str(path)
