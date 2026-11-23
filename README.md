# Running Hugging Face embedding models inside Oracle AI Database

**Note:** The code and information in this repository are for demonstration purposes online. The code is not endorsed, sanctioned, or otherwise approved by Oracle, Inc. There is no support, and the code is provided as-is. Use at your own discretion.

Demo scripts for converting [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), or any other embedding model on [Hugging Face](https://huggingface.co/) with a supported tokenizer, with [Oracle Machine Learning for Python (OML4Py)](https://docs.oracle.com/en/database/oracle/machine-learning/oml4py/), importing it into Oracle Autonomous AI Database, and then comparing database embeddings with the Nomic reference implementation.

- [`embedding_model_demo.py`](embedding_model_demo.py) Using the [OML4Py client container image](https://container-registry.oracle.com/ords/ocr/ba/database/oml4py-client), this script packages an embedding model from [Hugging Face](https://huggingface.co/) into ONNX format by creating an ONNX pipeline, and can import the saved ONNX artifact into an Oracle AI Database.
- [`compare_nomic_embeddings.py`](compare_nomic_embeddings.py) compares an unquantized [Nomic](https://www.nomic.ai/) reference vector with one or both imported database models. The reference comparison is specific to Nomic Embed Text v1.5.
- `config/config.toml` holds the shared settings and sample inputs. Edit it for your environment.
- `support/demo_config.py` loads and checks those settings.
- `support/create_demo_user.py` connects as `database.admin` to create a separate database user for the demo.
- [Setup guide](docs/nomic-embedding-demo-setup.md) covers the OML4Py container, `einops`, Oracle wallet setup, conversion, and export to an Oracle AI Database.

Run the scripts inside the configured OML4Py client container. Edit `config/config.toml` for your environment; `support/demo_config.py` reads and validates the shared settings and configures the wallet. Passwords are requested at runtime. Do not commit wallet files, credentials, model downloads, or exported ONNX files.

## Configure the database, wallet, and model

Review these settings in `config/config.toml` before running the scripts:

| When changing | Setting | What to enter or check |
| --- | --- | --- |
| Demo database user | `database.user` | The account that imports and compares models, such as `EMBEDDING_DEMO`. It must exist and have the required privileges. Models are stored and looked up in this user's schema, so import them again when changing users. |
| Setup administrator | `database.admin` | The account used by `support/create_demo_user.py` to create the demo user and run setup SQL. The default is `ADMIN`. |
| Database service | `database.dsn` | A service alias from the wallet's `tnsnames.ora`, such as `database-name_low`, replaced with your actual alias. |
| Wallet location | `database.wallet_dir` | The absolute path **inside the container** to the directory containing `tnsnames.ora` and the extracted wallet files, for example `/u01/oml_home/wallet`. |
| Hugging Face model | `model.id` | The Hugging Face repository ID, such as `nomic-ai/nomic-embed-text-v1.5`. Check that OML4Py supports the model's tokenizer. |
| Model name | `model.name` | The base Oracle model name. The scripts add `_INT8` to a quantized database model name, and `_ONNX` plus an optional `_INT8` to exported file names. |
| Model dimensions | `model.dimensions` | The expected output dimensions for validation and the demo table's vector column. Set it to match the model's actual output; this setting does not resize the embedding. |
| Input length | `model.max_tokens` | The maximum token length used for conversion and reference inference. Choose a limit supported by the model. |

When changing models, also review `model.do_lower_case`, `model.post_processors`, `model.distance_metrics`, `model.languages`, `model.use_float16`, and `model.query` against the model's requirements. A supported tokenizer alone does not guarantee the entire model is compatible. The comparison script's reference implementation is specific to Nomic Embed Text v1.5 and cannot validate a different model without changes.

Use `model.quantize` to choose a quantized or unquantized export and `model.external_data` to choose a ZIP with external data or a standalone ONNX file. `model.output_dir` controls where the artifact is saved; `ONNX_OUTPUT_DIR`, if set, overrides it. Run `convert` with the desired settings before `import-db`, and keep the relevant name, quantization, external-data, and output settings consistent so import finds the intended artifact. For comparison, enable the database model variants you imported with `comparison.include_unquantized` and `comparison.include_quantized`.

The database import and embedding generation have been exercised. The comparison was run on several sample inputs. The current external-data file export and other configurations should be verified in the target environment. These examples are not a substitute for retrieval-quality evaluation on application data.

Run `convert` first. `import-db` reads that saved ONNX file or ZIP, uploads its graph and external weight files, and imports those bytes into the database. It does not download or convert the model again. It tests the new model under a temporary name before promoting it; if the target name exists, the previous model is retained under a printed backup name. A failed validation can leave a temporary model to inspect or remove. This direct BLOB import path requires validation in the target Autonomous Database. SHA-256 values identify the bytes read from the artifact; they do not verify a database readback. Do not use this replacement path for models managed by `DBMS_VECTOR_DATABASE`.

To create a separate database user, set the wallet directory and service alias in the configuration, then run `python3 support/create_demo_user.py` from the repository root. The script prompts for the configured administrator password and a new password, creates the `database.user` account (default `EMBEDDING_DEMO`) with a 2 GB DATA quota, and grants `CREATE SESSION`, `CREATE TABLE`, and `CREATE MINING MODEL`. Use `--quota 4G` to change the quota. `--user NAME` overrides the configured user for this run; update `database.user` to match. It does not modify an existing user. Database DDL commits independently, so if a later step fails the account may already exist.

The script also runs SQL as the configured administrator to create `EMBEDDING_DEMO.DEMO_DOCUMENTS` with an ID, document text, and a vector column sized from `model.dimensions`. To run a different single statement instead, pass `--sql 'SELECT USER FROM dual'` without a trailing semicolon. The statement executes with the configured administrator's privileges. Set `database.user` in the configuration to the new user and import models again into that user's schema. Database Actions can enable OML for an existing user if additional OML features require it. The original setup script was tested end to end; the configurable administrator change still needs validation.
