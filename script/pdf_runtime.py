import ctypes
import gc
import json
from multiprocessing import process
import os
import statistics
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Optional

import psutil
import requests
from loguru import logger
from mineru.cli.common import do_parse, init_BatchAnalyze, read_fn, set_env


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def release_request_memory() -> None:
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


class PDF_Instance:
    def __init__(self, args):
        mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        if args.enable_cache and mem_gb < 4:
            print(f"System memory is {mem_gb:.2f}GB (<4GB). Disabling cache to reduce memory usage.")
            args.enable_cache = False
        if args.nstreams > 1 and mem_gb <= 32:
            print(f"System memory is {mem_gb:.2f}GB, disabling multiple streams.")
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
        self.return_json = bool(getattr(args, "enable_json", False)) and args.app
        self.enable_cache = args.enable_cache
        self.release_model_per_request = _env_bool(
            "MINERU_RELEASE_MODEL_PER_REQUEST",
            default=(not self.enable_cache),
        )
        logger.info(
            f"MINERU_RELEASE_MODEL_PER_REQUEST={self.release_model_per_request} "
            f"(enable_cache={self.enable_cache})"
        )

        self.pdf_model = init_BatchAnalyze(
            enable_cache=self.enable_cache,
            enable_ov=self.enable_ov,
            Layout_infer_type=self.layout_type,
            MFD_infer_type=self.mfd_type,
            MFR_enc_infer_type=self.mfr_enc_type,
            MFR_dec_infer_type=self.mfr_dec_type,
            OCR_det_infer_type=self.ocr_det_type,
            OCR_rec_infer_type=self.ocr_rec_type,
            wired_table_type=self.wired_table_type,
            WirelessTable_type=self.wireless_table_type,
            img_orientation_cls_type=self.img_cls_type,
            table_cls_type=self.table_cls_type,
            layoutreader_type=self.layoutreader_type,
            nstreams=self.nstreams,
        )
        self.output_dir = "/tmp/pdf_ocr_output"

    def process_pdf(self, pdf_bytes: bytes, file_name: str = "input.pdf", return_output_meta: bool = False) -> Any:
        result = do_parse(
            output_dir=self.output_dir,
            pdf_file_names=[file_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["ch"],
            BatchAnalyze=self.pdf_model,
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
            f_draw_line_sort_bbox=False,
            start_page_id=0,
            end_page_id=None,
            return_output_meta=return_output_meta,
        )
        if return_output_meta:
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError("Invalid parse result when return_output_meta=True")
            md_raw, json_raw, output_metas = result
            output_meta = output_metas[0] if output_metas else None
            return md_raw, json_raw, output_meta
        return result

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


def download_file(url: str, filename: Optional[str] = None, directory: Optional[str] = None) -> Path:
    filename = filename or Path(urllib.parse.urlparse(url).path).name
    filepath = Path(directory) / filename if directory is not None else Path(filename)
    if filepath.exists():
        return filepath.resolve()
    if directory is not None:
        Path(directory).mkdir(parents=True, exist_ok=True)

    response = requests.get(url=url, headers={"User-agent": "Mozilla/5.0"}, stream=True)
    response.raise_for_status()
    with open(filepath, "wb") as file_object:
        for chunk in response.iter_content(16384):
            file_object.write(chunk)
    response.close()
    return filepath.resolve()


def load_pdf_file(file_path: str) -> bytes:
    return read_fn(file_path)


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
            for root, _dirs, files in os.walk(input_name):
                for f in files:
                    if f.lower().endswith("pdf"):
                        pdf_files.append(os.path.join(root, f))
        elif os.path.isfile(input_name) and input_name.lower().endswith("pdf"):
            pdf_files.append(input_name)
    return pdf_files


def run_app_benchmark(pdf_instance: PDF_Instance, pdf_files, repeat, warmup, output_json=None):
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
        },
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if output_json is not None:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved benchmark summary to: {output_json}")

def print_processing_info(input_name, output_path, json_raw, md_raw, latency, output_meta=None, error=None, log_mode="full"):
    message = f"Processed {input_name}, json_raw={len(json_raw) if json_raw is not None else 'None'}, md_raw={len(md_raw) if md_raw is not None else 'None'}, latency={latency:.6f} seconds"
    if error:
        message = f"{message}, error={error}"
    output_item = {
        "output_path": output_path,
        "md_path": None,
        "images_md_dir": None,
        "md_raw": md_raw,
    }
    if isinstance(output_meta, dict):
        for key in ["md_path", "images_md_dir"]:
            if key in output_meta and output_meta.get(key) is not None:
                output_item[key] = output_meta.get(key)

    if output_path is None:
         info = {
            "success": False,
            "message" : message,
            "outputs": [
                output_item
            ]
        }
    else :
        info = {
            "success": True,
            "message": message,
            "outputs": [
                output_item
            ]
        }
    if log_mode == "serving":
        file_name = os.path.basename(str(input_name))
        print(f"Processed {file_name}, latency={latency:.6f} seconds, output_path={output_item.get('output_path')}")
    else:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    return info

def run_local_files(pdf_instance: PDF_Instance, pdf_files):
    for full_path in pdf_files:
        try :
            pdf_raw = load_pdf_file(full_path)
            start_time = time.perf_counter()
            md_raw, json_raw, output_meta = pdf_instance.process_pdf(
                pdf_raw,
                file_name=os.path.basename(full_path),
                return_output_meta=True,
            )
            end_time = time.perf_counter()
            latency = end_time - start_time
            output_path = output_meta.get("output_dir") if isinstance(output_meta, dict) else None
            print_processing_info(full_path, output_path, json_raw, md_raw, latency, output_meta=output_meta)
        except Exception as exc:
            print_processing_info(full_path, None, None, None, 0.0, error=str(exc))

def verify_output() -> str:
    return json.dumps(
        [
            {
                "success": True,
                "message": "Installation Verified Successfully",
                "outputs": [],
            }
        ],
        ensure_ascii=False,
    )
