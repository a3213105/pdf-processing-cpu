
import requests
from pathlib import Path
import argparse
import json
import time
import statistics
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--url', '-u', type=str, default="http://127.0.0.1:5000/", help='serving url')
    parser.add_argument('--pdf', '-p', type=str, default="/home/sgui/BD/pdf_ocr/MinerU/demo/pdfs/demo1.pdf",
                        help='Filenames of input pdf')
    parser.add_argument('--benchmark', action='store_true', default=False, help='run benchmark mode')
    parser.add_argument('--allow_remote_benchmark', action='store_true', default=False,
                        help='allow benchmark against non-local URLs (disabled by default)')
    parser.add_argument('--repeat', '-r', type=int, default=20, help='number of measured requests')
    parser.add_argument('--warmup', '-w', type=int, default=3, help='number of warmup requests')
    parser.add_argument('--concurrency', '-c', type=int, default=1, help='concurrent workers for benchmark')
    parser.add_argument('--timeout', '-t', type=int, default=300, help='request timeout in seconds')
    parser.add_argument('--save-json', type=str, default=None, help='save benchmark summary to a json file')
    return parser.parse_args()

args = parse_args()


def is_local_url(url):
    parsed = urlparse(url)
    host = parsed.hostname
    if host is None:
        return False
    return host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def _percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = rank - lower
    return float(sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac)


def print_single_response(data):
    print(f"JSON response:{data.keys()} {len(data['json_raw'])} {data['latency']:.6f} seconds")
    for line in data['json_raw']:
        if line['type'] == 'text' :
            print(f"{line['page_idx']}, {line['type']}, "
                    f"{line['text'] if len(line['text']) < 5 else len(line['text'])}")
        elif line['type'] == 'image' :
            print(f"{line['page_idx']}, {line['type']}, "
                    f"{line['img_caption'][0] if len(line['img_caption'][0]) < 5 else len(line['img_caption'][0])}, "
                    f"{line['img_footnote'] if len(line['img_footnote']) == 0 else len(line['img_footnote'][0])}")
        elif line['type'] == 'table' :
            print(f"{line['page_idx']}, {line['type']}, "
                    f"{line['table_caption'][0] if len(line['table_caption'][0]) < 5 else len(line['table_caption'][0])}, "
                    f"{line['table_footnote'] if len(line['table_footnote']) == 0 else len(line['table_footnote'][0])}")
        elif line['type'] == 'equation' :
            print(f"{line['page_idx']}, {line['type']}, "
                    f"{line['text'] if len(line['text']) == 0 else len(line['text'][0])}, "
                    f"{line['text_format']}")
        else :
            print(f"### {line['page_idx']}, {line['type']}, {line.keys()}")


def send_one_request(url, pdf_name, pdf_bytes, timeout):
    session = requests.Session()
    session.trust_env = False
    files = {
        "file": (
            pdf_name,
            pdf_bytes,
            "application/pdf",
        )
    }
    start = time.perf_counter()
    try:
        resp = session.post(url, files=files, timeout=timeout)
        resp.raise_for_status()
        _ = resp.json()
        end = time.perf_counter()
        return True, end - start, None
    except requests.RequestException as exc:
        end = time.perf_counter()
        return False, end - start, str(exc)
    except ValueError as exc:
        end = time.perf_counter()
        return False, end - start, f"Invalid JSON response: {exc}"
    finally:
        session.close()


def run_benchmark(url, pdf_name, pdf_bytes, repeat, warmup, concurrency, timeout):
    repeat = max(1, repeat)
    warmup = max(0, warmup)
    concurrency = max(1, concurrency)

    for _ in range(warmup):
        send_one_request(url, pdf_name, pdf_bytes, timeout)

    latencies = []
    failures = []
    benchmark_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_one_request, url, pdf_name, pdf_bytes, timeout)
            for _ in range(repeat)
        ]
        for future in as_completed(futures):
            success, latency, error_message = future.result()
            latencies.append(latency)
            if not success:
                failures.append(error_message)
    benchmark_end = time.perf_counter()

    sorted_latencies = sorted(latencies)
    success_count = repeat - len(failures)
    total_time = benchmark_end - benchmark_start
    throughput = repeat / total_time if total_time > 0 else 0.0

    summary = {
        "url": url,
        "pdf": pdf_name,
        "repeat": repeat,
        "warmup": warmup,
        "concurrency": concurrency,
        "timeout_sec": timeout,
        "total_time_sec": round(total_time, 6),
        "throughput_req_per_sec": round(throughput, 6),
        "success": success_count,
        "fail": len(failures),
        "success_rate": round(success_count / repeat, 6),
        "latency_sec": {
            "min": round(min(sorted_latencies), 6),
            "max": round(max(sorted_latencies), 6),
            "mean": round(statistics.mean(sorted_latencies), 6),
            "median": round(statistics.median(sorted_latencies), 6),
            "p90": round(_percentile(sorted_latencies, 0.90), 6),
            "p95": round(_percentile(sorted_latencies, 0.95), 6),
            "p99": round(_percentile(sorted_latencies, 0.99), 6),
        },
        "errors": failures[:10],
    }
    return summary


def main():
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pdf_bytes = pdf_path.read_bytes()

    if args.benchmark:
        if (not args.allow_remote_benchmark) and (not is_local_url(args.url)):
            raise ValueError(
                f"Benchmark only supports local service URL. Current: {args.url}. "
                "Use localhost/127.0.0.1/0.0.0.0/::1, or set --allow_remote_benchmark."
            )
        summary = run_benchmark(
            url=args.url,
            pdf_name=pdf_path.name,
            pdf_bytes=pdf_bytes,
            repeat=args.repeat,
            warmup=args.warmup,
            concurrency=args.concurrency,
            timeout=args.timeout,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if args.save_json is not None:
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"Saved benchmark report to: {args.save_json}")
        return

    session = requests.Session()
    session.trust_env = False
    files = {
        "file": (
            pdf_path.name,
            pdf_bytes,
            "application/pdf",
        )
    }

    try:
        resp = session.post(args.url, files=files, timeout=args.timeout)
        resp.raise_for_status()
        try:
            data = resp.json()
            print_single_response(data)
        except ValueError:
            print("Text response:", resp.text)
    except requests.RequestException as e:
        print(e)
    finally:
        session.close()


if __name__ == '__main__':
    main()