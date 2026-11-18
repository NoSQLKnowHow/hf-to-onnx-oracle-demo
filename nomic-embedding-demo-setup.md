# Nomic embedding model: demo setup

This guide prepares the OML4Py client container to convert `nomic-ai/nomic-embed-text-v1.5` into an ONNX pipeline and export it to Oracle Autonomous AI Database in OCI.

**Status:** The unquantized conversion created a 547,912,283-byte ONNX file with OML4Py 2.1.1. The quantized, external-data pipeline was exported directly to Autonomous AI Database and produced a 768-dimensional embedding. Comparison with Nomic reference embeddings remains outstanding. The export emitted tracing warnings and the OML4Py disconnect reported an insufficient INHERIT PRIVILEGES warning during temporary-object cleanup.

## 1. Open the existing container

On the OCI compute instance, list containers:

```bash
podman ps -a
```

If the container is stopped, start it with `podman start CONTAINER_NAME`. Open a shell, using its actual name:

```bash
podman exec -it -e TNS_ADMIN=/u01/oml_home/wallet CONTAINER_NAME /bin/bash
```

The remaining commands run inside the container unless otherwise stated.

## 2. Install the Nomic model dependency

The Nomic model's custom code requires `einops`. The OML4Py container in this demo was configured to use an Oracle internal package index that the OCI compute instance could not resolve. Install `einops` from the public Python package index:

```bash
python3 -m pip install --user --index-url https://pypi.org/simple einops
python3 -c "import einops; print('einops version:', einops.__version__)"
```

Install this before converting. If a new container is created, install the package again unless it has been included in that container's environment. Record the installed version after the demo has been successfully validated.

## 3. Configure the database wallet

In OCI, open the Autonomous AI Database's **Database connection** page and download an **Instance wallet**. Transfer its ZIP file securely to the compute instance. On the compute instance, outside the container, replace the example ZIP path and container name:

```bash
mkdir -p ~/adb-wallet
chmod 700 ~/adb-wallet
unzip ~/Wallet_mydb.zip -d ~/adb-wallet
chmod -R go-rwx ~/adb-wallet
podman exec CONTAINER_NAME mkdir -p /u01/oml_home/wallet
podman cp ~/adb-wallet/. CONTAINER_NAME:/u01/oml_home/wallet
```

Inside the container, configure the wallet location and verify it:

```bash
sed -i.bak 's|?/network/admin|/u01/oml_home/wallet|g' /u01/oml_home/wallet/sqlnet.ora
cat /u01/oml_home/wallet/sqlnet.ora
```

The `DIRECTORY` entry should point to `/u01/oml_home/wallet`. The shell must have `TNS_ADMIN=/u01/oml_home/wallet`, as set by the `podman exec` command above. Copying the wallet into the existing container does not make it available to a replacement container. For a persistent setup, mount a protected wallet directory when creating the container.

## 4. Copy the demo script

Transfer `embedding_model_demo.py` to the compute instance. On the compute instance, outside the container, substitute the path and container name:

```bash
podman cp /path/to/embedding_model_demo.py CONTAINER_NAME:/u01/oml_home/embedding_model_demo.py
```

## 5. Convert to ONNX

Inside the container:

```bash
cd /u01/oml_home
python3 embedding_model_demo.py convert
```

The installed Transformers version asks for permission to run custom model code. Review the model and referenced code before answering `y` to each prompt. The prompts can time out. The script requests INT8 quantization and external model data. Oracle documents that `export2file` produces a ZIP archive containing the ONNX graph and its external data files. The script expects `NOMIC_EMBED_TEXT_V15_INT8.zip` in the current directory and displays a pipeline summary if conversion succeeds. This configuration and archive name have not yet been verified in the container.

The demo configures a 512-token input limit and full 768-dimensional embeddings. The observed conversion recommended quantization for a model over 400 MB and emitted tracing warnings about sequence-length conditions. Successful conversion alone does not establish numerical equivalence or correct behavior for all input lengths. Compare outputs with a Nomic reference for varied short, long, query, and document inputs before recording. Quantization is optional and needs its own quality checks.

## 6. Export to Autonomous AI Database

The script uses `ADMIN` for `DB_USER`. Set `DB_DSN` in the configuration section of the Python script. Use a TNS alias from the OCI Database connection page or the wallet's `tnsnames.ora`. The user needs `CREATE MINING MODEL`. The script prompts for the database user's password, which is separate from the wallet password.

```bash
python3 embedding_model_demo.py import-db
```

This step exports the pipeline directly to the database. It does not read the ONNX file from step 5. It checks that the mining model exists and generates a 768-dimensional embedding for a `search_query:` input. Use the `search_document:` prefix for documents. The demo uses the database model name `NOMIC_EMBED_TEXT_V15_INT8` and does not replace an existing database model or an existing ONNX file.

## References

- [Oracle OML4Py client container](https://docs.oracle.com/en/database/oracle/machine-learning/oml4py/2-23ai/mlpug/install-oml4py-client-container.html)
- [Download Autonomous AI Database client credentials](https://docs.oracle.com/en-us/iaas/autonomous-database-serverless/doc/connect-download-wallet.html)
- [Nomic model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Nomic custom model code](https://huggingface.co/nomic-ai/nomic-bert-2048)

## Observed database cleanup warning

After a successful export and embedding query, the OML4Py session reported ORA-06598 in `PYQSYS.PYQ$DELETEOREREFDBOBJECT` while cleaning up temporary tables. This is distinct from the model import. Verify the required privileges and cleanup with the database administrator or Oracle Support before changing grants or deleting objects. Do not assume the objects were removed.
