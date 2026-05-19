from flask import Flask, request, jsonify
import os
import uuid
import time
import argparse
import json
import statistics
import gc
import ctypes
from typing import Any, Optional
from pathlib import Path
from mineru.cli.common import do_parse, init_BatchAnalyze, read_fn, set_env
import requests
import psutil
import urllib.parse
import sys
from loguru import logger

app = Flask(__name__)

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--disable-ov', '-o', action='store_true', default=False, help='disable_ov')
    parser.add_argument('--layout-type', type=str, default="bf16", help='layout detection infer type')
    parser.add_argument('--mfd-type', type=str, default="bf16", help='formula detection infer type')
    parser.add_argument('--mfr-enc-type', type=str, default="bf16", help='formula recognition enc infer type')
    parser.add_argument('--mfr-dec-type', type=str, default="bf16", help='formula recognition dec infer type')
    parser.add_argument('--ocr-det-type', type=str, default="bf16", help='ocr detection infer type')
    parser.add_argument('--ocr-rec-type', type=str, default="bf16", help='ocr recognition infer type')
    parser.add_argument('--wired-table-type', type=str, default="bf16", help='wired table infer type')
    parser.add_argument('--wireless-table-type', type=str, default="bf16", help='wireless table infer type')
    parser.add_argument('--table-cls-type', type=str, default="bf16", help='page layout infer type')
    parser.add_argument('--img-cls-type', type=str, default="bf16", help='image orientation classification infer type')
    parser.add_argument('--layoutreader-type', type=str, default="bf16", help='page layout infer type')
    parser.add_argument('--all', '-a', type=str, default=None, help='set all infer type')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input pdfs')
    parser.add_argument('--nstreams', '-n', type=int, default=1, help='Number of ov streams')
    parser.add_argument('--app', '-p', action='store_true', default=False, help='True for app, False for serving')
    parser.add_argument('--benchmark', action='store_true', default=False, help='Enable benchmark in app mode only')
    parser.add_argument('--repeat', '-r', type=int, default=20, help='Number of measured benchmark rounds in app mode')
    parser.add_argument('--warmup', '-w', type=int, default=3, help='Number of warmup rounds in app mode benchmark')
    parser.add_argument('--benchmark-json', type=str, default=None, help='Optional benchmark summary output path (json)')
    parser.add_argument('--disable-json', '-j', action='store_true', default=False, help='disable json output')
    parser.add_argument("--disable-cache", "-c", dest="enable_cache", default=True, action="store_false", help="disable caching")
    parser.add_argument("--config", type=str, default="./mineru.json", help="Path to mineru.json configuration file",)

    return parser.parse_args()

args = parse_args()
if args.all is not None :
    args.all = args.all.lower()
    args.layout_type = args.all
    args.mfd_type = args.all
    args.mfr_enc_type = args.all
    args.mfr_dec_type = args.all
    args.ocr_det_type = args.all
    args.ocr_rec_type = args.all
    args.wired_table_type = args.all
    args.wireless_table_type = args.all
    args.img_cls_type = args.all
    args.table_cls_type = args.all
    args.layoutreader_type = args.all
else :
    args.layout_type = args.layout_type.lower()
    args.mfd_type = args.mfd_type.lower()
    args.mfr_enc_type = args.mfr_enc_type.lower()
    args.mfr_dec_type = args.mfr_dec_type.lower()
    args.ocr_det_type = args.ocr_det_type.lower()
    args.ocr_rec_type = args.ocr_rec_type.lower()
    args.wired_table_type = args.wired_table_type.lower()
    args.wireless_table_type = args.wireless_table_type.lower()
    args.img_cls_type = args.img_cls_type.lower()
    args.table_cls_type = args.table_cls_type.lower()
    args.layoutreader_type = args.layoutreader_type.lower()

logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add new handler

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

def _release_request_memory() -> None:
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass
 
class PDF_Instance :
    def __init__(self, args) :
        if args.all is not None :
            args.layout_type = args.all
            args.mfd_type = args.all
            args.mfr_enc_type = args.all
            args.mfr_dec_type = args.all
            args.ocr_det_type = args.all
            args.ocr_rec_type = args.all
            args.wired_table_type = args.all
            args.wireless_table_type = args.all
            args.img_cls_type = args.all
            args.table_cls_type = args.all
            args.layoutreader_type = args.all

        mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        if args.enable_cache and mem_gb < 4:
            print(f"System memory is {mem_gb:.2f}GB (<4GB). Disabling cache to reduce memory usage.")
            args.enable_cache = False
        if args.nstreams > 1 and mem_gb <= 32:  # Check if system memory is greater than 32GB for multiple streams
            print(f"System memory is {mem_gb:.2f}GB, which is not enough to enable multiple streams (requires >32GB). Disabling multiple streams.")
            args.nstreams = 1

        set_env(args.enable_cache, args.config)

        self.enable_ov = not args.disable_ov
        self.layout_type = args.layout_type
        self.mfd_type = args.mfd_type
        self.mfr_enc_type = args.mfr_enc_type
        self.mfr_dec_type = args.mfr_dec_type
        self.ocr_det_type = args.ocr_det_type
        self.ocr_rec_type = args.ocr_rec_type
        self.wired_table_type = args.wired_table_type
        self.wireless_table_type = args.wireless_table_type
        self.img_cls_type = args.img_cls_type
        self.table_cls_type = args.table_cls_type
        self.layoutreader_type = args.layoutreader_type
        self.nstreams = args.nstreams
        self.return_md = True
        self.return_json = (not args.disable_json) and args.app
        self.enable_cache = args.enable_cache
        self.release_model_per_request = _env_bool(
            "MINERU_RELEASE_MODEL_PER_REQUEST",
            default=(not self.enable_cache),
        )
        logger.info(
            f"MINERU_RELEASE_MODEL_PER_REQUEST={self.release_model_per_request} "
            f"(enable_cache={self.enable_cache})"
        )
        self.pdf_model = init_BatchAnalyze(enable_cache=self.enable_cache, enable_ov=self.enable_ov, 
                                    Layout_infer_type=self.layout_type, MFD_infer_type=self.mfd_type, MFR_enc_infer_type=self.mfr_enc_type,
                                    MFR_dec_infer_type=self.mfr_dec_type, OCR_det_infer_type=self.ocr_det_type, OCR_rec_infer_type=self.ocr_rec_type,
                                    wired_table_type=self.wired_table_type, WirelessTable_type=self.wireless_table_type,
                                    img_orientation_cls_type=self.img_cls_type, table_cls_type=self.table_cls_type,
                                    layoutreader_type=self.layoutreader_type, nstreams=self.nstreams)        
        self.output_dir = "/tmp/pdf_ocr_output"

    def process_pdf(self, pdf_bytes: bytes, file_name: str = "input.pdf") -> Any:
        return do_parse(
                output_dir=self.output_dir,
                pdf_file_names=[file_name],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=["ch"],
                BatchAnalyze = self.pdf_model,
                backend="pipeline",
                parse_method="auto",
                formula_enable=True,
                table_enable=True,
                server_url=None,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=self.return_md,
                f_dump_middle_json=self.return_json,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
                # f_make_md_mode=MakeMode.MM_MD,
                f_draw_line_sort_bbox=False,
                start_page_id=0,
                end_page_id=None,
            )

    def release_request_cache(self) -> None:
        if not self.release_model_per_request:
            return

        try:
            if hasattr(self.pdf_model, "model") and hasattr(self.pdf_model.model, "atom_model_manager"):
                manager = self.pdf_model.model.atom_model_manager
                if hasattr(manager, "clear_cache"):
                    manager.clear_cache()
        except Exception as exc:
            logger.warning(f"release atom model cache failed: {exc}")

        try:
            if hasattr(self.pdf_model, "model_manager") and hasattr(self.pdf_model.model_manager, "clear_cache"):
                self.pdf_model.model_manager.clear_cache()
        except Exception as exc:
            logger.warning(f"release model manager cache failed: {exc}")

pdf_instance = PDF_Instance(args)

def download_file(
    url: str,
    filename: Optional[str] = None,
    directory: Optional[str] = None,
    show_progress: bool = True,
) -> Path:
    """
    Download a file from a url and save it to the local filesystem. The file is saved to the
    current directory by default, or to `directory` if specified. If a filename is not given,
    the filename of the URL will be used.

    :param url: URL that points to the file to download
    :param filename: Name of the local file to save. Should point to the name of the file only,
                     not the full path. If None the filename from the url will be used
    :param directory: Directory to save the file to. Will be created if it doesn't exist
                      If None the file will be saved to the current working directory
    :param show_progress: If True, show an TQDM ProgressBar
    :param silent: If True, do not print a message if the file already exists
    :param timeout: Number of seconds before cancelling the connection attempt
    :return: path to downloaded file
    """
    # from tqdm import tqdm

    filename = filename or Path(urllib.parse.urlparse(url).path).name
    chunk_size = 16384  # make chunks bigger so that not too many updates are triggered for Jupyter front-end

    filename_path = Path(filename)
    if len(filename_path.parts) > 1:
        raise ValueError(
            "`filename` should refer to the name of the file, excluding the directory. "
            "Use the `directory` parameter to specify a target directory for the downloaded file."
        )

    filepath = Path(directory) / filename_path if directory is not None else filename_path
    if filepath.exists():
        return filepath.resolve()

    # create the directory if it does not exist, and add the directory to the filename
    if directory is not None:
        Path(directory).mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url=url, headers={"User-agent": "Mozilla/5.0"}, stream=True)
        response.raise_for_status()
    except (
        requests.exceptions.HTTPError
    ) as error:  # For error associated with not-200 codes. Will output something like: "404 Client Error: Not Found for url: {url}"
        raise Exception(error) from None
    except requests.exceptions.Timeout:
        raise Exception(
            "Connection timed out. If you access the internet through a proxy server, please "
            "make sure the proxy is set in the shell from where you launched Jupyter."
        ) from None
    except requests.exceptions.RequestException as error:
        raise Exception(f"File downloading failed with error: {error}") from None

    # download the file if it does not exist
    filesize = int(response.headers.get("Content-length", 0))
    if not filepath.exists():
        with open(filepath, "wb") as file_object:
            for chunk in response.iter_content(chunk_size):
                file_object.write(chunk)
    else:
        print(f"'{filepath}' already exists.")

    response.close()

    return filepath.resolve()

def load_pdf_file(file_path):
    return read_fn(file_path)
    # with open(file_path, 'rb') as f:
    #     f.seek(0)
    #     return f.read()
            
def percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = rank - lower
    return float(sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac)

def collect_pdf_inputs(input_paths):
    pdf_files = []
    for input_name in input_paths:
        if os.path.isdir(input_name):
            for root, dirs, files in os.walk(input_name):
                for f in files:
                    if f.lower().endswith("pdf"):
                        pdf_files.append(os.path.join(root, f))
        elif os.path.isfile(input_name) and input_name.lower().endswith("pdf"):
            pdf_files.append(input_name)
    return pdf_files

def run_app_benchmark(pdf_files, repeat, warmup, output_json=None):
    repeat = max(1, repeat)
    warmup = max(0, warmup)

    for _ in range(warmup):
        for pdf_file in pdf_files:
            pdf_raw = load_pdf_file(pdf_file)
            pdf_instance.process_pdf(pdf_raw, file_name=os.path.basename(pdf_file))

    latencies = []
    benchmark_start = time.perf_counter()
    for _ in range(repeat):
        for pdf_file in pdf_files:
            pdf_raw = load_pdf_file(pdf_file)
            one_start = time.perf_counter()
            pdf_instance.process_pdf(pdf_raw, file_name=os.path.basename(pdf_file))
            one_end = time.perf_counter()
            latencies.append(one_end - one_start)
    benchmark_end = time.perf_counter()

    sorted_latencies = sorted(latencies)
    total_requests = len(latencies)
    total_time = benchmark_end - benchmark_start
    throughput = total_requests / total_time if total_time > 0 else 0.0

    summary = {
        "mode": "app",
        "pdf_count": len(pdf_files),
        "repeat": repeat,
        "warmup": warmup,
        "total_requests": total_requests,
        "total_time_sec": round(total_time, 6),
        "throughput_req_per_sec": round(throughput, 6),
        "latency_sec": {
            "min": round(min(sorted_latencies), 6),
            "max": round(max(sorted_latencies), 6),
            "mean": round(statistics.mean(sorted_latencies), 6),
            "median": round(statistics.median(sorted_latencies), 6),
            "p90": round(percentile(sorted_latencies, 0.90), 6),
            "p95": round(percentile(sorted_latencies, 0.95), 6),
            "p99": round(percentile(sorted_latencies, 0.99), 6),
        }
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if output_json is not None:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved benchmark summary to: {output_json}")

@app.route('/', methods=['POST'])
def pdf_process():
    pdf_raw = None
    md_raw = None
    json_raw = None
    parse_file_name = "input.pdf"
    if request.is_json:
        json_data = request.get_json()
        if json_data:
            if 'url' in json_data:
                random_uuid = uuid.uuid4()
                filename = f"{random_uuid}.pdf"
                parse_file_name = filename
                filename = download_file(
                    url=json_data['url'],
                    filename=filename,
                    directory='/tmp'
                )
                pdf_raw = load_pdf_file(filename)
                os.remove(filename)
            elif 'pdf_raw' in json_data:
                pdf_raw = json_data['pdf_raw']
            else :
                return jsonify({'error': 'Unsupported JSON format. Expected {"url": "address"} or {"pdf_raw": "data"}'}), 400
        else:
            return jsonify({'error': 'Invalid JSON format. Expected {"url": "address"} or {"pdf_raw": "data"}'}), 400
    elif 'filename' in request.form:
        if os.path.exists(request.form['filename']):
            parse_file_name = os.path.basename(request.form['filename'])
            # load pdf file
            pdf_raw = load_pdf_file(request.form['filename'])
        else:
            return jsonify({'error': 'Failed to open image file'}), 400
    elif 'file' in request.files:       
        pdf_file = request.files['file']
        if pdf_file.filename:
            parse_file_name = pdf_file.filename
        pdf_raw = pdf_file.read()
    elif 'url' in request.form:
        random_uuid = uuid.uuid4()
        filename = f"{random_uuid}.pdf"
        parse_file_name = filename
        filename = download_file(
            url=request.form['url'],
            filename=filename,
            directory='/tmp'
        )
        if os.path.exists(filename):
            pdf_raw = load_pdf_file(filename)
            os.remove(filename)
        else:
            return jsonify({'error': 'Failed to download PDF from URL'}), 400
    else :
        return jsonify({'error': 'No PDF uploaded or filename provided'}), 400
    if pdf_raw is None :
        return jsonify({'error': 'PDF data is invalid'}), 400
    try:
        start_time = time.perf_counter()
        (md_raw, json_raw) = pdf_instance.process_pdf(pdf_raw, file_name=parse_file_name)
        end_time = time.perf_counter()
        latency = end_time - start_time
        print(f"Processed {parse_file_name} in {latency:.6f} seconds")
        return jsonify({'json_raw': json_raw, 'md_raw': md_raw, 'latency': latency})
    finally:
        if md_raw is not None:
            del md_raw
        if json_raw is not None:
            del json_raw
        if pdf_raw is not None:
            del pdf_raw
        pdf_instance.release_request_cache()
        _release_request_memory()

if __name__ == '__main__':
    if args.app:
        if args.input is None :
            print(f"app mode need set input")
            exit(0)
        elif isinstance(args.input, str) :
            args.input = [args.input]
        pdf_files = collect_pdf_inputs(args.input)
        if len(pdf_files) == 0:
            print("app mode need set valid pdf input")
            exit(0)

        if args.benchmark:
            run_app_benchmark(pdf_files, args.repeat, args.warmup, args.benchmark_json)
        else:
            for full_path in pdf_files:
                pdf_raw = load_pdf_file(full_path)
                start_time = time.perf_counter()
                (md_raw, json_raw) = pdf_instance.process_pdf(pdf_raw, file_name=os.path.basename(full_path))
                end_time = time.perf_counter()
                latency = end_time - start_time
                print(f"Processed {full_path}, json_raw={len(json_raw) if json_raw is not None else 'None'}, md_raw={len(md_raw) if md_raw is not None else 'None'}, latency={latency:.6f} seconds")
    else :
        if args.benchmark:
            print("--benchmark works only in app mode (--app). Use client.py to benchmark serving mode.")
        app.run(host='0.0.0.0', port=5000)