import json
import os
import uuid
import time

from flask import Flask, request, jsonify

from pdf_runtime import download_file, load_pdf_file, release_request_memory, print_processing_info


def create_app(pdf_instance):
    app = Flask(__name__)

    @app.route('/', methods=['POST'])
    def pdf_process():
        pdf_raw = None
        md_raw = None
        json_raw = None
        output_meta = None
        parse_file_name = 'input.pdf'

        if request.is_json:
            json_data = request.get_json()
            if json_data:
                if 'url' in json_data:
                    filename = f"{uuid.uuid4()}.pdf"
                    parse_file_name = filename
                    filename = download_file(url=json_data['url'], filename=filename, directory='/tmp')
                    pdf_raw = load_pdf_file(str(filename))
                    os.remove(filename)
                elif 'pdf_raw' in json_data:
                    pdf_raw = json_data['pdf_raw']
                else:
                    return jsonify({'error': 'Unsupported JSON format. Expected {"url": "address"} or {"pdf_raw": "data"}'}), 400
            else:
                return jsonify({'error': 'Invalid JSON format. Expected {"url": "address"} or {"pdf_raw": "data"}'}), 400
        elif 'filename' in request.form:
            filename = request.form['filename']
            if os.path.exists(filename):
                parse_file_name = os.path.basename(filename)
                pdf_raw = load_pdf_file(filename)
            else:
                return jsonify({'error': 'Failed to open image file'}), 400
        elif 'file' in request.files:
            pdf_file = request.files['file']
            if pdf_file.filename:
                parse_file_name = pdf_file.filename
            pdf_raw = pdf_file.read()
        elif 'url' in request.form:
            filename = f"{uuid.uuid4()}.pdf"
            parse_file_name = filename
            filename = download_file(url=request.form['url'], filename=filename, directory='/tmp')
            if os.path.exists(filename):
                pdf_raw = load_pdf_file(str(filename))
                os.remove(filename)
            else:
                return jsonify({'error': 'Failed to download PDF from URL'}), 400
        else:
            return jsonify({'error': 'No PDF uploaded or filename provided'}), 400

        if pdf_raw is None:
            return jsonify({'error': 'PDF data is invalid'}), 400

        try:
            start_time = time.perf_counter()
            md_raw, json_raw, output_meta = pdf_instance.process_pdf(
                pdf_raw,
                file_name=parse_file_name,
                return_output_meta=True,
            )
            end_time = time.perf_counter()
            latency = end_time - start_time
            output_path = output_meta.get("output_dir") if isinstance(output_meta, dict) else pdf_instance.output_dir
            process_info = print_processing_info(
                parse_file_name,
                output_path,
                json_raw,
                md_raw,
                latency,
                output_meta=output_meta,
                log_mode="serving",
            )
            return jsonify(process_info)
            # return jsonify({'json_raw': json_raw, 'md_raw': md_raw, 'latency': latency})
        finally:
            if md_raw is not None:
                del md_raw
            if json_raw is not None:
                del json_raw
            if pdf_raw is not None:
                del pdf_raw
            pdf_instance.release_request_cache()
            release_request_memory()

    return app
