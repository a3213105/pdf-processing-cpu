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
from magic_pdf.model.doc_analyze_by_custom_model import init_models, doc_analyze_direct
SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent
os.environ['MINERU_TOOLS_CONFIG_JSON'] = f'{SCRIPT_DIR}/magic-pdf.json'


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--disable_ov', '-d', action='store_true', default=False, help='disable_ov')
    parser.add_argument('--layout_type', type=str, default="bf16", help='layout detection infer type')
    parser.add_argument('--mfd_type', type=str, default="bf16", help='formula detection infer type')
    parser.add_argument('--mfr_enc_type', type=str, default="bf16", help='formula recognition enc infer type')
    parser.add_argument('--mfr_dec_type', type=str, default="bf16", help='formula recognition dec infer type')
    parser.add_argument('--ocr_det_type', type=str, default="bf16", help='ocr detection infer type')
    parser.add_argument('--ocr_rec_type', type=str, default="bf16", help='ocr recognition infer type')
    parser.add_argument('--table_type', type=str, default="bf16", help='table infer type')
    parser.add_argument('--lang_type', type=str, default="bf16", help='language detection infer type')
    parser.add_argument('--page_type', type=str, default="bf16", help='page layout infer type')
    parser.add_argument('--all', '-a', type=str, default=None, help='set all infer type')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', default="demo/pdfs/demo1.pdf",
                        help='Filenames of input pdfs')
    parser.add_argument('--output_dir', '-o', type=str, default="./outputs",
                        help='outputs directory for markdown and images')
    parser.add_argument('--nstreams', '-n', type=int, default=8, help='Number of ov streams')
    parser.add_argument('--verify', '-v', action='store_true', default=False, help='Verify Installation')
    parser.add_argument('--return_json', '-j', action='store_true', default=False, 
                        help='enable json output (set return_json=True)')
    parser.add_argument('--return_md', '-m', action='store_true', default=False,
                        help='enable markdown output (set return_md=True)')
    parser.add_argument('--return_layout', '-l', action='store_true', default=False,
                        help='enable layout output (set return_layout=True)')
    parser.add_argument('--return_span', '-s', action='store_true', default=False,
                        help='enable span output (set return_span=True)')
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
    args.table_type = args.all
    args.lang_type = args.all
    args.page_type = args.all
else :
    args.layout_type = args.layout_type.lower()
    args.mfd_type = args.mfd_type.lower()
    args.mfr_enc_type = args.mfr_enc_type.lower()
    args.mfr_dec_type = args.mfr_dec_type.lower()
    args.ocr_det_type = args.ocr_det_type.lower()
    args.ocr_rec_type = args.ocr_rec_type.lower()
    args.table_type = args.table_type.lower()
    args.lang_type = args.lang_type.lower()
    args.page_type = args.page_type.lower()

def download_file(
    url: PathLike,
    filename: PathLike = None,
    directory: PathLike = None,
    show_progress: bool = True,
) -> PathLike:
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
    import requests
    import urllib.parse

    filename = filename or Path(urllib.parse.urlparse(url).path).name
    chunk_size = 16384  # make chunks bigger so that not too many updates are triggered for Jupyter front-end

    filename = Path(filename)
    if len(filename.parts) > 1:
        raise ValueError(
            "`filename` should refer to the name of the file, excluding the directory. "
            "Use the `directory` parameter to specify a target directory for the downloaded file."
        )

    filepath = Path(directory) / filename if directory is not None else filename
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
    with open(file_path, 'rb') as f:
        f.seek(0)
        return f.read()
