#!/usr/bin/env python3
"""Candidate Nomic ONNX conversion and Autonomous Database demo.

Run inside the full OML4Py 2.1 client container:
    python3 embedding_model_demo.py convert
    python3 embedding_model_demo.py import-db

Edit config/config.toml and set the database connection alias;
configure network access and the wallet if required by that connection. The password is prompted
without echoing. The database user needs CREATE MINING MODEL. Internet access to Hugging Face
is needed on the first conversion. This candidate needs an actual test with this model and the
installed OML4Py version before recording.
"""

import argparse
import getpass
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from support.demo_config import configure_wallet, load_config

# Load the configurations from config.toml
CONFIG = load_config()
configure_wallet(CONFIG)

from oml.utils import ONNXPipeline, ONNXPipelineConfig

DB_USER = CONFIG["database"]["user"]
DB_DSN = CONFIG["database"]["dsn"]
MODEL = CONFIG["model"]
MODEL_ID = MODEL["id"]
QUANTIZE_MODEL = MODEL["quantize"]
MODEL_NAME = MODEL["name"] + ("_INT8" if QUANTIZE_MODEL else "")
FILE_NAME = MODEL["name"] + "_ONNX" + ("_INT8" if QUANTIZE_MODEL else "")
OUTPUT_DIR = Path(os.environ.get("ONNX_OUTPUT_DIR", MODEL["output_dir"]))
MAX_TOKENS = MODEL["max_tokens"]
USE_EXTERNAL_DATA = MODEL["external_data"]
QUERY = MODEL["query"]
DIMENSIONS = MODEL["dimensions"]

@contextmanager
def database_export_directory():
    """Isolate files OML4Py may create in the process working directory."""
    original_dir = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=".onnx-db-export-", dir=OUTPUT_DIR.resolve()) as temp_dir:
        try:
            os.chdir(temp_dir)
            yield
        finally:
            os.chdir(original_dir)


def make_pipeline():
    config = ONNXPipelineConfig.from_template(
        "text",
        max_seq_length=MAX_TOKENS,
        distance_metrics=MODEL["distance_metrics"],
        quantize_model=QUANTIZE_MODEL,
        use_external_data=USE_EXTERNAL_DATA,
        do_lower_case=MODEL["do_lower_case"],
        post_processors=MODEL["post_processors"],
        languages=MODEL["languages"],
        use_float16=MODEL["use_float16"],
    )
    return ONNXPipeline(model_name=MODEL_ID, config=config)


def convert():
    suffix = ".zip" if USE_EXTERNAL_DATA else ".onnx"
    output_file = OUTPUT_DIR / f"{FILE_NAME}{suffix}"
    if not OUTPUT_DIR.is_dir():
        raise SystemExit(f"Output directory does not exist: {OUTPUT_DIR}")
    if output_file.exists():
        raise SystemExit(f"Output file already exists: {output_file}")
    print(f"Converting {MODEL_ID} to an ONNX pipeline...", flush=True)
    pipeline = make_pipeline()
    pipeline.export2file(FILE_NAME, output_dir=str(OUTPUT_DIR))
    if not output_file.is_file():
        raise RuntimeError(f"Expected output file not found: {output_file}")
    print(f"Saved: {output_file.resolve()} ({output_file.stat().st_size:,} bytes)")
    print("Pipeline summary:", pipeline.summary())


def import_db():
    configure_wallet(CONFIG, check_files=True)
    import oml

    user, dsn = DB_USER, DB_DSN
    if not user or not dsn or dsn == "database-name_low":
        raise SystemExit("Set database.dsn in config/config.toml before importing.")
    password = getpass.getpass("Enter your database password: ")
    oml.connect(user=user, password=password, dsn=dsn)
    try:
        cursor = oml.cursor()
        cursor.execute(
            "SELECT algorithm FROM user_mining_models WHERE model_name = :name",
            {"name": MODEL_NAME},
        )
        existing = cursor.fetchone()
        if existing and existing[0] != "ONNX":
            raise RuntimeError(
                f"Cannot replace {MODEL_NAME}: the existing model is not an ONNX model."
            )

        # export2file and export2db are separate exports. Stage the file first
        # so a failed file export leaves any existing file and model intact.
        if not OUTPUT_DIR.is_dir():
            raise RuntimeError(f"Output directory does not exist: {OUTPUT_DIR}")
        pipeline = make_pipeline()
        suffix = ".zip" if USE_EXTERNAL_DATA else ".onnx"
        output_file = OUTPUT_DIR / f"{FILE_NAME}{suffix}"
        print(f"Saving ONNX pipeline to {output_file}...", flush=True)
        with tempfile.TemporaryDirectory(prefix=".onnx-export-", dir=OUTPUT_DIR) as staging_dir:
            pipeline.export2file(FILE_NAME, output_dir=staging_dir)
            staged_file = Path(staging_dir) / output_file.name
            if not staged_file.is_file():
                raise RuntimeError(f"Expected output file not found: {staged_file}")
            os.replace(staged_file, output_file)
        print(f"Saved: {output_file.resolve()} ({output_file.stat().st_size:,} bytes)")

        # OML4Py may retain the staged path after export2file. Use a fresh
        # pipeline for export2db so it does not reuse a moved temporary file.
        # The direct database export prepares the model again; it does not load
        # the file above. The old model is dropped just before export.
        db_pipeline = make_pipeline()
        if existing:
            print(f"Replacing existing database model {MODEL_NAME}...", flush=True)
            cursor.execute(
                "BEGIN DBMS_VECTOR.DROP_ONNX_MODEL(model_name => :name); END;",
                {"name": MODEL_NAME},
            )
        print(f"Exporting {MODEL_ID} directly to the database...", flush=True)
        try:
            with database_export_directory():
                db_pipeline.export2db(MODEL_NAME)
        except Exception:
            if existing:
                print(
                    f"Export failed after dropping {MODEL_NAME}; the previous model was not restored.",
                    flush=True,
                )
            raise
        print("Pipeline summary:", db_pipeline.summary())

        cursor.execute(
            "SELECT model_name, algorithm, mining_function "
            "FROM user_mining_models WHERE model_name = :name",
            {"name": MODEL_NAME},
        )
        model = cursor.fetchone()
        if not model:
            raise RuntimeError("Export completed, but the model was not found.")
        print("Database model:", model)

        # MODEL_NAME is a fixed identifier; the input text is a bind variable.
        cursor.execute(
            f"SELECT VECTOR_EMBEDDING({MODEL_NAME} USING :text AS data) FROM dual",
            {"text": QUERY},
        )
        vector = cursor.fetchone()[0]
        if vector is None or len(vector) != DIMENSIONS:
            raise RuntimeError(f"Expected a {DIMENSIONS}-dimensional embedding.")
        print("Input:", QUERY)
        print(f"Embedding: {len(vector)} dimensions; first five values: {list(vector[:5])}")
        cursor.close()
    finally:
        oml.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("convert", "import-db"))
    args = parser.parse_args()
    (convert if args.step == "convert" else import_db)()


if __name__ == "__main__":
    main()
