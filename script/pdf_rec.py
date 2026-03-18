from PIL import Image
import io
import sys
import os
import uuid
import time
import argparse
import numpy as np
from os import PathLike
from pathlib import Path
from utils import load_pdf_file
from magic_pdf.model.doc_analyze_by_custom_model import init_models, doc_analyze_direct
SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent
os.environ['MINERU_TOOLS_CONFIG_JSON'] = f'{SCRIPT_DIR}/magic-pdf.json'

class PDF_Instance :
    def __init__(self, args) :
        if args.all is not None :
            args.layout_type = args.all
            args.mfd_type = args.all
            args.mfr_enc_type = args.all
            args.mfr_dec_type = args.all
            args.ocr_det_type = args.all
            args.ocr_rec_type = args.all
            args.table_type = args.all
            args.lang_type = args.all
            args.page_type = args.all
        self.enable_ov = not args.disable_ov
        self.layout_type = args.layout_type
        self.mfd_type = args.mfd_type
        self.mfr_enc_type = args.mfr_enc_type
        self.mfr_dec_type = args.mfr_dec_type
        self.ocr_det_type = args.ocr_det_type
        self.ocr_rec_type = args.ocr_rec_type
        self.table_type = args.table_type
        self.lang_type = args.lang_type
        self.page_type = args.page_type
        self.nstreams = args.nstreams
        self.pdf_model = init_models(self.enable_ov, self.layout_type, self.mfd_type, self.mfr_enc_type,
                        self.mfr_dec_type, self.ocr_det_type, self.ocr_rec_type, self.table_type,
                        self.lang_type, self.page_type, self.nstreams, True)

    def process_pdf(self, pdf_raw: bytes, return_md, return_json, output_dir: str, input_name: str = None) :
        return doc_analyze_direct(pdf_raw, self.pdf_model, self.enable_ov, self.layout_type, self.mfd_type,
                                  self.mfr_enc_type, self.mfr_dec_type, self.ocr_det_type,
                                  self.ocr_rec_type, self.table_type, self.lang_type, self.page_type,
                                  self.nstreams, return_md, return_json, output_dir, input_name)

def process_pdf_file(args, pdf_instance) :
    if args.input is None :
        print(f"app mode need set input")
        exit(0)
    elif isinstance(args.input, str) :
        args.input = [args.input]
    output_md_list = []
    for input_name in args.input:
        if os.path.isdir(input_name) :  
            for root, dirs, files in os.walk(input_name):
                for input_name in files:
                    if input_name.lower().endswith("pdf"):
                        full_path = os.path.join(root, input_name)
                        pdf_raw = load_pdf_file(full_path)
                        (md_raw, json_raw, page_info, output_md_filename) = pdf_instance.process_pdf(pdf_raw, args.return_md, args.return_json, args.output_dir, Path(input_name).stem)
                        output_md_list.append((input_name, output_md_filename, md_raw, json_raw, page_info))
        elif os.path.isfile(input_name):
            pdf_raw = load_pdf_file(input_name)
            (md_raw, json_raw, page_info, output_md_filename) = pdf_instance.process_pdf(pdf_raw, args.return_md, args.return_json, args.output_dir, Path(input_name).stem)
            # print(f"### process file {input_name} done")
            # print(f"### markdown output ({args.return_md}):\n{md_raw}")
            # print(f"### json output ({args.return_json}):\n{json_raw}")
            # print(f"### page info:\n{page_info}")
            # print(f"### markdown file saved at: {output_md_filename}")
            output_md_list.append((input_name, output_md_filename, md_raw, json_raw, page_info))
        else :
            print(f"app mode need set input")
    return output_md_list