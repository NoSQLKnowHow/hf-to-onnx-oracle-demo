# Running Hugging Face embedding models inside Oracle AI Database

**Note:** The code and information in this repository are for demonstration purposes online. They are not endorsed, sanctioned, or otherwise approved by Oracle, Inc.

Demo scripts for converting [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), or any other embedding model on [Hugging Face](https://huggingface.co/), with [Oracle Machine Learning for Python (OML4Py)](https://docs.oracle.com/en/database/oracle/machine-learning/oml4py/), importing it into Oracle Autonomous AI Database, and then comparing database embeddings with the Nomic reference implementation.

- [`embedding_model_demo.py`](embedding_model_demo.py) Using the [OML4Py client container image](https://container-registry.oracle.com/ords/ocr/ba/database/oml4py-client), this script packages an embedding model from [Hugging Face](https://huggingface.co/) into ONNX format by creating an ONNX pipeline, and can export that embedding model directly to an Oracle AI Database, if desired.
- [`compare_nomic_embeddings.py`](compare_nomic_embeddings.py) compares an unquantized [Nomic](https://www.nomic.ai/) reference vector with one or both imported database models. The reference comparison is specific to Nomic Embed Text v1.5.
- `config/config.example.toml` holds the shared settings and sample inputs. Copy it to `config/config.toml` and edit your copy.
- `support/demo_config.py` loads and checks those settings.
- [Setup guide](docs/nomic-embedding-demo-setup.md) covers the OML4Py container, `einops`, Oracle wallet setup, conversion, and export to an Oracle AI Database.

Run the scripts inside the configured OML4Py client container. Copy `config/config.example.toml` to `config/config.toml`, then set your service alias and base model name in the configuration file. The quantized comparison name is derived by adding `_INT8`. The example uses `ADMIN`. Passwords are requested at runtime. Do not commit wallet files, credentials, model downloads, or exported ONNX files.

The database import and embedding generation have been exercised. The comparison was run on several sample inputs. The current external-data file export and other configurations should be verified in the target environment. These examples are not a substitute for retrieval-quality evaluation on application data.
