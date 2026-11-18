#!/usr/bin/env python3
"""Compare quantized and unquantized Oracle models with Nomic reference vectors.

Run inside the OML4Py container with the wallet configured (TNS_ADMIN).
Set DB_DSN and the database model names below. This script only queries existing models.
"""

import getpass
import os
import re

import numpy as np
import oracledb
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DB_USER = "ADMIN"
DB_DSN = "database-name_low"
# Fixed SQL identifier, not a bind variable. Change only to a trusted model name.
QUANTIZE_MODEL = False
MODEL_NAME = "NOMIC_EMBED_TEXT_V15" + ("_INT8" if QUANTIZE_MODEL else "")
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
# The custom-code revision observed in the successful export log.
CODE_REVISION = "7710840340a098cfb869c4f65e87cf2b1b70caca"
MAX_TOKENS = 512

CASES = [
    ("query_short", "search_query: How do I find documents by meaning?"),
    ("document_short", "search_document: Vector search finds documents with related meanings."),
    ("document_unrelated", "search_document: A recipe for baking a loaf of bread."),
    ("query_case", "search_query: How does Oracle Database handle AI Vector Search?"),
    ("document_medium", "search_document: " + "Oracle Database can store embeddings and use vector search to find related documents. " * 20),
    ("document_near_limit", "search_document: " + "vector search " * 245),
    ("document_over_limit", "search_document: " + "vector search " * 270),
]


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
    if DB_DSN == "your_database_low":
        raise SystemExit("Set DB_DSN in this script before running.")
    if not os.environ.get("TNS_ADMIN"):
        raise SystemExit("Set TNS_ADMIN to the wallet directory before running.")
    # The OML4Py image includes Oracle Instant Client. Thick mode uses sqlnet.ora.
    oracledb.init_oracle_client()
    password = getpass.getpass("Database password: ")
    with oracledb.connect(user=DB_USER, password=password, dsn=DB_DSN) as connection:
        with connection.cursor() as cursor:
            models = {"unquantized": DB_MODEL_UNQUANTIZED, "quantized": DB_MODEL_QUANTIZED}
            if DB_MODEL_UNQUANTIZED == "YOUR_UNQUANTIZED_MODEL_NAME":
                raise SystemExit("Set DB_MODEL_UNQUANTIZED to the imported model name.")
            if len(set(models.values())) != 2:
                raise SystemExit("Set distinct model names for quantized and unquantized models.")
            for kind, name in models.items():
                if not re.fullmatch(r"[A-Z][A-Z0-9_$#]*", name):
                    raise SystemExit(f"Use a simple uppercase Oracle model name for {kind}: {name}")
                cursor.execute("SELECT model_name FROM user_mining_models WHERE model_name = :name", name=name)
                if not cursor.fetchone():
                    raise SystemExit(f"Model {name} was not found in the connected user's schema.")
            print("Database models:", models)
            print("Loading Nomic reference model. This may take a few minutes.", flush=True)
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            # Allow token counting without a misleading model-length warning.
            # Reference inference still explicitly truncates to MAX_TOKENS.
            tokenizer.model_max_length = 1_000_000
            model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, code_revision=CODE_REVISION)
            model.eval()
            print("Reference model revision:", getattr(model.config, "_commit_hash", "unknown"))
            print("Custom code revision:", CODE_REVISION)
            print("Token count includes special tokens; reference truncates at 512.")
            print("Pairwise cosine similarity (1.0 means identical direction):")
            print(f"{'case':22} {'tokens':>7} {'ref/unquant':>12} {'ref/quant':>12} {'unquant/quant':>13}")
            results = {}
            for label, text in CASES:
                token_count = len(tokenizer(text, truncation=False)["input_ids"])
                if token_count > MAX_TOKENS:
                    print(f"  Note: {label} has {token_count} tokens; the reference input is truncated to {MAX_TOKENS} tokens before inference. This is an example of why you need a data chunking strategy.", flush=True)
                vectors = {"reference": reference_embedding(text, tokenizer, model)}
                for kind, name in models.items():
                    cursor.execute(f"SELECT VECTOR_EMBEDDING({name} USING :text AS data) FROM dual", text=text)
                    row = cursor.fetchone()
                    if not row or row[0] is None:
                        raise RuntimeError(f"{name} returned no vector for {label}")
                    vectors[kind] = np.asarray(row[0], dtype=np.float64)
                for kind, vector in vectors.items():
                    if vector.shape != (768,) or not np.isfinite(vector).all():
                        raise RuntimeError(f"Unexpected dimensions or non-finite values for {label} / {kind}: {vector.shape}")
                results[label] = vectors
                ref, unquant, quant = (vectors[k] for k in ("reference", "unquantized", "quantized"))
                print(f"{label:22} {token_count:7d} {cosine(ref, unquant):12.6f} {cosine(ref, quant):12.6f} {cosine(unquant, quant):13.6f}", flush=True)
            print("\nMaximum absolute coordinate differences:")
            print(f"{'case':22} {'ref/unquant':>12} {'ref/quant':>12} {'unquant/quant':>13}")
            for label, vectors in results.items():
                ref, unquant, quant = (vectors[k] for k in ("reference", "unquantized", "quantized"))
                print(f"{label:22} {np.max(np.abs(ref-unquant)):12.6f} {np.max(np.abs(ref-quant)):12.6f} {np.max(np.abs(unquant-quant)):13.6f}")
            print("\nVector norms:")
            for label, vectors in results.items():
                print(f"{label:22} " + " ".join(f"{kind}={np.linalg.norm(vector):.6f}" for kind, vector in vectors.items()))
            print("\nQuery-to-document cosine similarities (higher is closer):")
            print(f"{'document':22} {'reference':>12} {'unquantized':>12} {'quantized':>12}")
            for label in ("document_short", "document_unrelated", "document_medium"):
                values = [cosine(results["query_short"][kind], results[label][kind]) for kind in ("reference", "unquantized", "quantized")]
                print(f"{label:22} " + " ".join(f"{value:12.6f}" for value in values))
            print("\nNo automatic pass threshold: inspect differences and ranking with representative data.")


if __name__ == "__main__":
    main()
