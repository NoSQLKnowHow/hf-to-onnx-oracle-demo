# Converting an embedding model from HuggingFace.co to be used inside an Oracle AI Database

**Note:** The code and information in this repository are for demonstration purposes online. They are not endorsed, sanctioned, or otherwise approved by Oracle, Inc.

Demo scripts for converting `nomic-ai/nomic-embed-text-v1.5`, or any other embedding model on HF.co, with Oracle Machine Learning for Python (OML4Py), importing it into Oracle Autonomous AI Database, and then comparing database embeddings with the Nomic reference implementation.

- `embedding_model_demo.py` Using the [OML4Py client container image](https://container-registry.oracle.com/ords/ocr/ba/database/oml4py-client), this script packages an embedding model from HuggingFace.co into ONNX format by creating an ONNX pipeline, and can export that embedding model directly to an Oracle AI Database, if desired.
- `compare_nomic_embeddings.py` compares an unquantized [Nomic](https://www.nomic.ai/) reference vector with an ONNX-formatted quantized and unquantized version of the same embedding model hosted in an Oracle AI Database.
- [Setup guide](nomic-embedding-demo-setup.md) covers the OML4Py container, `einops`, Oracle wallet setup, conversion, and export to an Oracle AI Database.

Run the scripts inside the configured OML4Py client container. The scripts use the `ADMIN` database user. Set the service alias and model names in the scripts. Passwords are requested at runtime. Do not commit wallet files, credentials, model downloads, or exported ONNX files.

The database import and embedding generation have been exercised. The comparison was run on several sample inputs. The current external-data file export and other configurations should be verified in the target environment. These examples are not a substitute for retrieval-quality evaluation on application data.
