# PDF Processing CPU

## 0. Directory Layout

Core runtime files are under the `script/` directory:

- `script/main.py`: Skill entrypoint (recommended)
- `script/app.py`: retained service implementation (includes `PDF_Instance`)
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
python script/main.py --app -i /PATH/TO/PDF/FILE --config script/mineru.json
```

Batch directory:

```bash
python script/main.py --app -i /PATH/TO/PDF_DIR --config script/mineru.json
```

Verify installation:

```bash
python script/main.py -v
```

## 3. Key Arguments

The following arguments are for `script/main.py`:

- `-i, --input`: input PDF file or directory
- `-p, --app`: run local parsing mode (without this flag, runs HTTP service mode)
- `--config`: path to `mineru.json`
- `-n, --nstreams`: number of OpenVINO streams (default: `8`)
- `-o, --disable-ov`: disable OpenVINO
- `-c, --disable-cache`: disable cache to reduce memory usage
- `--enable-json`: enable middle JSON output (disabled by default)
- `-j, --disable-json`: keep JSON output disabled (default behavior)
- `-v, --verify`: verify installation

## 4. Output Schema

Local parse mode (`--app`) prints one JSON object per input PDF:

```json
{
	"success": true,
	"message": "Processed /abs/path/file.pdf, json_raw=0, md_raw=49778, latency=212.364162 seconds",
	"outputs": [
		{
			"output_path": "/tmp/pdf_ocr_output/file.pdf",
			"md_path": "/tmp/pdf_ocr_output/file.pdf/file.pdf.md",
			"images_md_dir": "/tmp/pdf_ocr_output/file.pdf/images_md",
			"md_raw": "...full markdown text..."
		}
	]
}
```

Failure output keeps the same schema, with `success=false`, `output_path=null`, and error details appended to `message`.

Cache behavior:

- Cache is enabled by default.
- If total system memory is detected as `<4GB` at startup, cache is automatically disabled.
- You can force cache off with `--disable-cache`.

## 5. HTTP Service Mode (Optional)

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

## 6. Troubleshooting

- `script/main.py` reports config errors: check whether `script/mineru.json` is correct.
- First run is slow: model loading and warmup are expected.
- High memory usage on large PDFs: use `--disable-cache` first.