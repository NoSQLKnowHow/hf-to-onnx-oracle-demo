#!/usr/bin/env python3
"""Compare one or two imported Nomic models with Hugging Face reference vectors.

Copy config.example.toml to config.toml and configure the database, model names,
and example inputs. Run inside the OML4Py client container with TNS_ADMIN set.
This script only queries existing models; it does not import or delete them.
"""
import getpass
import os

import numpy as np
import oracledb
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from demo_config import load_config, validate_model_name

CONFIG = load_config()
DB_USER = CONFIG["database"]["user"]
DB_DSN = CONFIG["database"]["dsn"]
MODEL = CONFIG["model"]
COMPARISON = CONFIG["comparison"]
MAX_TOKENS = MODEL["max_tokens"]


def read_cases():
    cases = {}
    for case in CONFIG.get("cases", []):
        name = case.get("name")
        if not isinstance(name, str) or not name or name in cases:
            raise SystemExit("Each [[cases]] entry needs a unique nonempty name.")
        if "text" in case:
            text = case["text"]
        else:
            repeat = case.get("repeat", 1)
            if type(repeat) is not int or not 1 <= repeat <= 10000:
                raise SystemExit(f"Invalid repeat count for {name}.")
            text = case.get("prefix", "") + case.get("repeat_text", "") * repeat
        if not isinstance(text, str) or not text:
            raise SystemExit(f"Case {name} needs nonempty text.")
        cases[name] = text
    if not cases:
        raise SystemExit("Add at least one [[cases]] entry to config.toml.")
    return cases


def reference_embedding(text, tokenizer, model):
    encoded = tokenizer(text, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
        token_vectors = output[0]
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_vectors.size()).float()
        pooled = (token_vectors * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        result = F.normalize(pooled, p=2, dim=1)
    return result[0].cpu().numpy().astype(np.float64)


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else float("nan")


def main():
    if DB_DSN == "database-name_low":
        raise SystemExit("Set database.dsn in config.toml before running.")
    if not os.environ.get("TNS_ADMIN"):
        raise SystemExit("Set TNS_ADMIN to the wallet directory before running.")
    try:
        models = {kind: validate_model_name(name) for kind, name in (
            ("unquantized", COMPARISON["unquantized_model"]),
            ("quantized", COMPARISON["quantized_model"]),
        ) if name}
        if not models or len(set(models.values())) != len(models):
            raise ValueError("Set at least one database model name; enabled names must be distinct.")
        tokenizer_id = COMPARISON["tokenizer_id"]
        code_revision = COMPARISON["code_revision"]
        query_case = COMPARISON["query_case"]
        document_cases = COMPARISON["document_cases"]
    except (KeyError, ValueError, TypeError) as exc:
        raise SystemExit(f"Invalid comparison configuration: {exc}")
    cases = read_cases()
    if not isinstance(document_cases, list) or any(name not in cases for name in document_cases) or query_case not in cases:
        raise SystemExit("comparison.query_case and document_cases must name existing cases.")
    # This reference implements Nomic's mean-pooling and normalization recipe.
    if MODEL["id"] != "nomic-ai/nomic-embed-text-v1.5":
        raise SystemExit("The comparison reference currently supports Nomic Embed Text v1.5 only.")
    oracledb.init_oracle_client()
    password = getpass.getpass("Database password: ")
    with oracledb.connect(user=DB_USER, password=password, dsn=DB_DSN) as connection:
        with connection.cursor() as cursor:
            for kind, name in models.items():
                cursor.execute("SELECT model_name FROM user_mining_models WHERE model_name = :name", name=name)
                if not cursor.fetchone():
                    raise SystemExit(f"Model {name} was not found in the connected user's schema.")
            print("Database models:", models)
            print("Loading Nomic reference model. This may take a few minutes.", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            tokenizer.model_max_length = 1_000_000
            model = AutoModel.from_pretrained(MODEL["id"], trust_remote_code=True, code_revision=code_revision)
            model.eval()
            print("Reference model revision:", getattr(model.config, "_commit_hash", "unknown"))
            print("Custom code revision:", code_revision)
            print(f"Token count includes special tokens; reference truncates at {MAX_TOKENS}.")
            results = {}
            for label, text in cases.items():
                token_count = len(tokenizer(text, truncation=False)["input_ids"])
                if token_count > MAX_TOKENS:
                    print(f"Note: {label} has {token_count} tokens; the reference input is truncated to {MAX_TOKENS} tokens before inference. This is an example of why you need a data chunking strategy.")
                vectors = {"reference": reference_embedding(text, tokenizer, model)}
                for kind, name in models.items():
                    cursor.execute(f"SELECT VECTOR_EMBEDDING({name} USING :text AS data) FROM dual", text=text)
                    row = cursor.fetchone()
                    if not row or row[0] is None:
                        raise RuntimeError(f"{name} returned no vector for {label}")
                    vectors[kind] = np.asarray(row[0], dtype=np.float64)
                for kind, vector in vectors.items():
                    if vector.shape != (MODEL["dimensions"],) or not np.isfinite(vector).all():
                        raise RuntimeError(f"Unexpected dimensions or non-finite values for {label} / {kind}: {vector.shape}")
                results[label] = vectors
                print(f"\n{label} ({token_count} tokens):")
                names = list(vectors)
                for i, first in enumerate(names):
                    for second in names[i + 1:]:
                        x, y = vectors[first], vectors[second]
                        print(f"  {first} / {second}: cosine={cosine(x, y):.6f}, max abs diff={np.max(np.abs(x-y)):.6f}")
                print("  norms: " + ", ".join(f"{k}={np.linalg.norm(v):.6f}" for k, v in vectors.items()), flush=True)
            print("\nQuery-to-document cosine similarities (higher is closer):")
            for label in document_cases:
                scores = [f"{kind}={cosine(results[query_case][kind], results[label][kind]):.6f}" for kind in results[query_case]]
                print(f"  {label}: " + ", ".join(scores))
            print("\nNo automatic pass threshold: inspect differences and ranking with representative data.")


if __name__ == "__main__":
    main()
