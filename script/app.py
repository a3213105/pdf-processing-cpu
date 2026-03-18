import os
os.environ["YOLO_VERBOSE"] = 'False'
from PIL import Image
import sys
import uuid
import time
import argparse
import numpy as np
from os import PathLike
from pathlib import Path
from flask import Flask, request, jsonify
from pdf_rec import PDF_Instance
from utils import parse_args, download_file, load_pdf_file

app = Flask(__name__)
args = parse_args()
pdf_instance = PDF_Instance(args)

@app.route('/', methods=['POST'])
def pdf_process():
    pdf_raw = None
    return_md = args.return_md
    return_json = args.return_json
    output_dir = args.output_dir
    if 'disable_md' in request.form:
        return_md = not request.form['disable_md']
    if 'disable_json' in request.form:
        return_json = not request.form['disable_json']
    if 'output_dir' in request.form:
        output_dir = request.form['output_dir']
    if request.is_json:
        json_data = request.get_json()
        if json_data:
            if 'output_dir' in json_data:
                output_dir = json_data['output_dir']
            if output_dir is None :
                output_dir = "./outputs"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)               
            if 'url' in json_data:
                random_uuid = uuid.uuid4()
                filename = f"{random_uuid}.pdf"
                filename = download_file(
                    url=json_data['url'],
                    filename=filename,
                    directory=output_dir
                )
                pdf_raw = load_pdf_file(filename)
                os.remove(filename)
            elif 'pdf_raw' in json_data:
                pdf_raw = json_data['pdf_raw'],
            else :
                 return jsonify({'error': 'Unsupported JSON format. Expected {"url": "address"} or {"pdf_raw": "data"}'}), 400
            if 'disable_md' in json_data:
                return_md = not json_data['disable_md']
            if 'disable_json' in json_data:
                return_json = not json_data['disable_json']
        else:
            return jsonify({'error': 'Invalid JSON format. Expected {"url": "address"} or {"pdf_raw": "data"}'}), 400
    elif 'filename' in request.form:
        if os.path.exists(request.form['filename']):
           # load pdf file
            pdf_raw = load_pdf_file(request.form['filename'])
        else:
            return jsonify({'error': 'Failed to open image file'}), 400
    elif 'file' in request.files:       
        pdf_file = request.files['file']
        pdf_raw = pdf_file.read()
    elif 'url' in request.form:
        random_uuid = uuid.uuid4()
        filename = f"{random_uuid}.pdf"
        if output_dir is None :
            output_dir = "./outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)               
        filename = download_file(
           url=request.form['url'],
           filename=filename,
           directory=output_dir
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

    start_time = time.perf_counter()
    (md_raw, json_raw, page_info, output_md_filename) = pdf_instance.process_pdf(pdf_raw, False, return_json, local_dir)
    end_time = time.perf_counter()
    print(f"Processing:  {end_time-start_time:.3f}")
    return jsonify({'json_raw': json_raw})

app.run(host='0.0.0.0', port=5000)