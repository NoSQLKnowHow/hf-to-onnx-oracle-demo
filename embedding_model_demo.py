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
import hashlib
import json
import uuid
import zipfile
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

def artifact_path():
    suffix = ".zip" if USE_EXTERNAL_DATA else ".onnx"
    return OUTPUT_DIR / f"{FILE_NAME}{suffix}"


def upload_blob(connection, source):
    """Stream the bytes from the saved artifact into a temporary database BLOB."""
    import oracledb

    blob = connection.createlob(oracledb.DB_TYPE_BLOB)
    digest = hashlib.sha256()
    offset = 1
    while chunk := source.read(1024 * 1024):
        blob.write(chunk, offset)
        digest.update(chunk)
        offset += len(chunk)
    return blob, digest.hexdigest()


def import_artifact(cursor, path, candidate):
    """Import the ONNX graph and external data contained in the saved artifact."""
    from contextlib import ExitStack

    metadata = json.dumps({"function": "embedding", "embeddingOutput": "embedding",
                           "input": {"input": ["DATA"]}})
    with ExitStack() as stack:
        blobs = []
        def upload(name, stream):
            blob, digest = upload_blob(cursor.connection, stream)
            blobs.append(blob)
            print(f"  {name}: SHA-256 {digest}", flush=True)
            return blob

        binds = {"name": candidate, "metadata": metadata}
        if path.suffix == ".onnx":
            with path.open("rb") as source:
                binds["model"] = upload(path.name, source)
            sql = "BEGIN DBMS_DATA_MINING.IMPORT_ONNX_MODEL(:name, :model, JSON(:metadata)); END;"
        else:
            archive = stack.enter_context(zipfile.ZipFile(path))
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("The archive contains duplicate filenames.")
            graphs = [n for n in names if n.endswith(".onnx")]
            manifests = [n for n in names if n.endswith("_external_data.json")]
            if len(graphs) != 1 or len(manifests) != 1:
                raise ValueError("Expected one ONNX graph and one external-data JSON in the ZIP.")
            manifest_bytes = archive.read(manifests[0])
            manifest = json.loads(manifest_bytes)
            groups = manifest.get("external_data_groups")
            if not isinstance(groups, list) or not groups:
                raise ValueError("External-data metadata has no data groups.")
            group_names = [g.get("group_name") for g in groups]
            if any(not isinstance(n, str) or not n or n not in names for n in group_names) or len(set(group_names)) != len(group_names):
                raise ValueError("External-data metadata has missing or duplicate data files.")
            with archive.open(graphs[0]) as source:
                binds["model"] = upload(graphs[0], source)
            binds["external_metadata"] = manifest_bytes.decode("utf-8")
            statements = []
            for index, name in enumerate(group_names):
                with archive.open(name) as source:
                    binds[f"data{index}"] = upload(name, source)
                binds[f"filename{index}"] = name
                statements.append(f"groups(:filename{index}) := :data{index};")
            sql = ("DECLARE groups DBMS_DATA_MINING.EXTERNAL_DATA_GROUPS_T; BEGIN "
                   + " ".join(statements)
                   + " DBMS_DATA_MINING.IMPORT_ONNX_MODEL(:name, :model, JSON(:metadata), "
                     "JSON(:external_metadata), groups); END;")
        cursor.execute(sql, binds)


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
    output_file = artifact_path()
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

        output_file = artifact_path()
        if not output_file.is_file():
            raise RuntimeError(f"Saved ONNX artifact not found: {output_file}. Run convert first.")
        # Import under a temporary name first, so a failed upload or import does
        # not remove the existing model. The previous model is retained as a backup.
        candidate = "ONNX_STAGE_" + uuid.uuid4().hex[:20].upper()
        backup = "ONNX_BACKUP_" + uuid.uuid4().hex[:20].upper()
        print(f"Importing the saved artifact: {output_file.resolve()}", flush=True)
        print(f"Temporary database model name: {candidate}", flush=True)
        import_artifact(cursor, output_file, candidate)
        print(f"Imported as {candidate}; checking its embedding...", flush=True)
        cursor.execute(f"SELECT VECTOR_EMBEDDING({candidate} USING :text AS data) FROM dual", {"text": QUERY})
        candidate_vector = cursor.fetchone()[0]
        if candidate_vector is None or len(candidate_vector) != DIMENSIONS:
            raise RuntimeError(f"Expected a {DIMENSIONS}-dimensional embedding from {candidate}.")
        try:
            cursor.execute(
                "BEGIN DBMS_DATA_MINING.RENAME_MODEL(:candidate, :target, :backup); END;",
                {"candidate": candidate, "target": MODEL_NAME, "backup": backup if existing else None},
            )
        except Exception:
            print(f"Could not promote {candidate}; it may still exist in the database.", flush=True)
            raise
        if existing:
            print(f"Previous database model preserved as {backup}.")

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
