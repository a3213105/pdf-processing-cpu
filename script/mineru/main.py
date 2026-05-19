# Copyright (c) Opendatalab. All rights reserved.
import os
import argparse
from pathlib import Path
os.environ["YOLO_VERBOSE"] = 'False'
from loguru import logger
from mineru.cli.common import read_fn, do_parse, init_BatchAnalyze, set_env
from mineru.utils.enum_class import MakeMode
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path
        
def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference pipeline with optional OpenVINO acceleration"
    )

    parser.add_argument("--input", "-i", type=str, help="Input file or directory")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    parser.add_argument("--disable_ov", "-d", dest="enable_ov", action="store_false", default=True, help="Disable OpenVINO inference")
    parser.add_argument("--enable_cache", "-c", dest="enable_cache", default=False, action="store_true", help="Enable caching")
    parser.add_argument("--config", dest="config_path", type=str, default="./mineru.json", help="Path to mineru.json configuration file",)
    parser.add_argument("--return_layout", "-l", action="store_true", default=False, help="Return layout information")
    parser.add_argument("--return_span", "-p", action="store_true", default=False, help="Return span information")
    parser.add_argument("--return_json", "-j", action="store_true", default=False, help="Return result in JSON format")
    parser.add_argument("--return_line", "-n", action="store_true", default=False, help="Return line information")
    parser.add_argument("--init", "-I", action="store_true", default=False, help="Initialize the system")
    parser.add_argument("--start_page_id", "-s", type=int, default=0, help="Start page ID for parsing (default: 0)")
    parser.add_argument("--end_page_id", "-e", type=int, default=None, help="End page ID for parsing (default: None, parse to end)")
    args = parser.parse_args()

    return args

from typing import List


def collect_input_files(input_path: Path) -> List[Path]:
    from mineru.cli.common import pdf_suffixes, image_suffixes

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    input_files = []
    if input_path.is_file():
        if input_path.suffix.lower().replace('.', '') not in pdf_suffixes + image_suffixes:
            raise ValueError(f"Input file is not a PDF or supported image: {input_path}")
        input_files.append(input_path)
    elif input_path.is_dir():
        for doc_path in Path(input_path).glob('*'):
            if guess_suffix_by_path(doc_path) in pdf_suffixes + image_suffixes:
                input_files.append(doc_path)
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    return input_files

def parse_doc(path_list: list[Path], output_dir, BatchAnalyze,
              return_layout, return_span, return_json, return_line,
              lang="ch", method="auto", start_page_id=0, end_page_id=None):

    """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka', 'th', 'el',
                       'latin', 'arabic', 'east_slavic', 'cyrillic', 'devanagari']。
            Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
            Adapted only for the case where the backend is set to 'pipeline' and 'hybrid-*'
        backend: the backend for parsing pdf:
            pipeline: More general.
            vlm-auto-engine: High accuracy via local computing power.
            vlm-http-client: High accuracy via remote computing power(client suitable for openai-compatible servers).
            hybrid-auto-engine: Next-generation high accuracy solution via local computing power.
            hybrid-http-client: High accuracy but requires a little local computing power(client suitable for openai-compatible servers).
            Without method specified, hybrid-auto-engine will be used by default.
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
            Adapted only for the case where the backend is set to 'pipeline' and 'hybrid-*'.
        server_url: When the backend is `http-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
        start_page_id: Start page ID for parsing, default is 0
        end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)
    """
    try:
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            do_parse(
                output_dir=output_dir,
                pdf_file_names=[file_name],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[lang],
                BatchAnalyze = BatchAnalyze,
                backend="pipeline",
                parse_method="auto",
                formula_enable=True,
                table_enable=True,
                server_url=None,
                f_draw_layout_bbox=return_layout,
                f_draw_span_bbox=return_span,
                f_dump_md=True,
                f_dump_middle_json=return_json,
                f_dump_model_output=return_json,
                f_dump_orig_pdf=return_json,
                f_dump_content_list=return_json,
                f_make_md_mode=MakeMode.MM_MD,
                f_draw_line_sort_bbox=return_line,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
            )
            del pdf_bytes
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    """If you are unable to download the model due to network problems, you can set the environment variable MINERU_MODEL_SOURCE to use the agent-free warehouse to download the model for modelscope."""
    # os.environ['MINERU_MODEL_SOURCE'] = "modelscope"

    args = parse_args()

    set_env(args.enable_cache, args.config_path)
    
    Layout_infer_type = "bf16"
    MFD_infer_type = "bf16"
    MFR_enc_infer_type = "bf16"
    MFR_dec_infer_type = "bf16"
    OCR_det_infer_type = "bf16"
    OCR_rec_infer_type = "bf16"
    wired_table_type = "bf16"
    WirelessTable_type = "bf16"
    img_orientation_cls_type = "bf16"
    table_cls_type = "bf16"
    layoutreader_type = "f16"
    if args.enable_cache:
        cpu_count = os.cpu_count() or 1
        nstreams = max((cpu_count // 2), 1)
    else:
        nstreams = 1

    try :
        BatchAnalyze =init_BatchAnalyze(True, True, Layout_infer_type, MFD_infer_type, MFR_enc_infer_type, MFR_dec_infer_type,
                          OCR_det_infer_type, OCR_rec_infer_type, wired_table_type, WirelessTable_type, img_orientation_cls_type,
                          table_cls_type, layoutreader_type, nstreams, remove_unused_weight=True)
    except Exception as e:
        logger.exception(f"Initialization failed: {str(e)}")
        exit(-1)
    if args.init:
        logger.info("Initialization completed.")
    else :
        doc_path_list = collect_input_files(Path(args.input))

        """Use hybrid mode and local computing power to parse documents"""
        parse_doc(doc_path_list, args.output, BatchAnalyze,
                  return_layout=args.return_layout, return_span=args.return_span,
                  return_json=args.return_json, return_line=args.return_line,
                  start_page_id=args.start_page_id, end_page_id=args.end_page_id)

