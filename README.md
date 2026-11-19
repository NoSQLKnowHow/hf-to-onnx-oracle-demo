# Converting an embedding model from HuggingFace.co to be used inside an Oracle AI Database

**Note:** The code and information in this repository are not endorsed, sanctioned, or otherwise approved by Oracle, Inc.

Demo scripts for converting `nomic-ai/nomic-embed-text-v1.5` with Oracle Machine Learning for Python (OML4Py), importing it into Oracle Autonomous AI Database, and comparing database embeddings with the Nomic reference implementation.

- `embedding_model_demo.py` creates an ONNX pipeline and exports a pipeline directly to the database.
- `compare_nomic_embeddings.py` compares an unquantized Nomic reference with quantized and unquantized database models.
- [Setup guide](nomic-embedding-demo-setup.md) covers the OML4Py container, `einops`, wallet setup, conversion, and database export.

Run the scripts inside the configured OML4Py client container. The scripts use the `ADMIN` database user. Set the service alias and model names in the scripts. Passwords are requested at runtime. Do not commit wallet files, credentials, model downloads, or exported ONNX files.

The database import and embedding generation have been exercised. The comparison was run on several sample inputs. The current external-data file export and other configurations should be verified in the target environment. These examples are not a substitute for retrieval-quality evaluation on application data.
