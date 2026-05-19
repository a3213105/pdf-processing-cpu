# PDF Processing CPU

## 0. Directory Layout

Core runtime files are under the `script/` directory:

- `script/main.py`: Skill entrypoint (recommended)
- `script/app.py`: 保留的服务实现（含 `PDF_Instance`）
- `script/entry_args.py`: argument definitions
- `script/pdf_runtime.py`: runtime/benchmark helpers
- `script/http_server.py`: HTTP route wrapper
- `script/client.py`: HTTP client
- `script/prepare_models.sh`: model download script
- `script/mineru.json`: MinerU config

## 1. Installation

### 1.1 Prerequisites

- Python 3.10+
- Linux (recommended)

### 1.2 Install dependencies

Run from the repository root:

```bash
pip install -r requirements.txt
```

Or use the install script:

```bash
bash install.sh
```

### 1.3 Prepare models

```bash
bash script/prepare_models.sh /PATH/TO/MODEL_OUTPUT modelscope
```

or:

```bash
bash script/prepare_models.sh /PATH/TO/MODEL_OUTPUT huggingface
```

## 2. Local Batch Mode (Recommended)

Use the skill entrypoint:

```bash
python script/main.py -i /PATH/TO/PDF/FILE -o /PATH/TO/OUTPUT_DIR
```

Batch directory:

```bash
python script/main.py -i /PATH/TO/PDF_DIR -o /PATH/TO/OUTPUT_DIR
```

Verify installation:

```bash
python script/main.py -v
```

## 3. Key Arguments

The following arguments are for `script/main.py`:

- `-i, --input`: input PDF file or directory
- `-o, --output_dir`: output directory
- `-n, --nstreams`: number of OpenVINO streams (default: `8`)
- `-c, --disable-cache`: disable cache to reduce memory usage
- `-j, --return_json`: output JSON
- `-m, --return_md`: output Markdown (enabled by default)
- `-v, --verify`: verify installation

Cache behavior:

- Cache is enabled by default.
- If total system memory is detected as `<4GB` at startup, cache is automatically disabled.
- You can force cache off with `--disable-cache`.

## 4. HTTP Service Mode (Optional)

Start service:

```bash
python script/app.py
```

Request example:

```bash
curl --noproxy "*" -X POST http://127.0.0.1:5000/ \
	-F "file=@/PATH/TO/PDF/FILE;type=application/pdf"
```

Or use client:

```bash
python script/client.py --url http://127.0.0.1:5000/ --pdf /PATH/TO/PDF/FILE
```

## 5. Troubleshooting

- `script/main.py` reports config errors: check whether `script/mineru.json` is correct.
- First run is slow: model loading and warmup are expected.
- High memory usage on large PDFs: use `--disable-cache` first.