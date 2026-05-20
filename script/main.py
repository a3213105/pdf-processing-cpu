import os
os.environ["YOLO_VERBOSE"] = 'False'

from entry_args import parse_args
from entry_args import normalize_infer_types
from http_server import create_app
from pdf_runtime import (
    PDF_Instance,
    collect_pdf_inputs,
    run_app_benchmark,
    run_local_files,
    verify_output,
)


def run():
    args = parse_args()
    args = normalize_infer_types(args)

    if args.verify:
        print(verify_output())
        return

    pdf_instance = PDF_Instance(args)

    if args.app:
        if args.input is None:
            print("app mode need set input")
            return

        pdf_files = collect_pdf_inputs(args.input)
        if len(pdf_files) == 0:
            print("app mode need set valid pdf input")
            return

        if args.benchmark:
            run_app_benchmark(pdf_instance, pdf_files, args.repeat, args.warmup, args.benchmark_json)
        else:
            run_local_files(pdf_instance, pdf_files)
    else:
        if args.benchmark:
            print("--benchmark works only in app mode (--app). Use client.py to benchmark serving mode.")
        app = create_app(pdf_instance)
        app.run(host='0.0.0.0', port=5000, threaded=False)


if __name__ == '__main__':
    run()