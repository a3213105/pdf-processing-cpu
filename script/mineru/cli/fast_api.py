import sys
import uuid
import os
import re
import tempfile
import asyncio
import uvicorn
import click
import zipfile
import shutil
from pathlib import Path
import glob
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
from loguru import logger

log_level = os.getenv("MINERU_LOG_LEVEL", "INFO").upper()
logger.remove()  # Remove default handler
logger.add(sys.stderr, level=log_level)  # Add new handler

from base64 import b64encode

from mineru.cli.common import aio_do_parse, read_fn, pdf_suffixes, image_suffixes
from mineru.utils.cli_parser import arg_parse
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path
from mineru.version import __version__

# Concurrency controller
_request_semaphore: Optional[asyncio.Semaphore] = None


# Concurrency control dependency function
async def limit_concurrency():
    if _request_semaphore is not None:
        # Check if the semaphore has been exhausted and reject the request if so
        if _request_semaphore._value == 0:
            raise HTTPException(
                status_code=503,
                detail=f"Server is at maximum capacity: {os.getenv('MINERU_API_MAX_CONCURRENT_REQUESTS', 'unset')}. Please try again later.",
            )
        async with _request_semaphore:
            yield
    else:
        yield


def create_app():
    # By default, the OpenAPI documentation endpoints (openapi_url, docs_url, redoc_url) are enabled.
    # To disable the FastAPI docs and schema endpoints, set the environment variable MINERU_API_ENABLE_FASTAPI_DOCS=0.
    enable_docs = str(os.getenv("MINERU_API_ENABLE_FASTAPI_DOCS", "1")).lower() in (
        "1",
        "true",
        "yes",
    )
    app = FastAPI(
        openapi_url="/openapi.json" if enable_docs else None,
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
    )

    # Initialize the concurrency controller: read from the environment variable MINERU_API_MAX_CONCURRENT_REQUESTS
    global _request_semaphore
    try:
        max_concurrent_requests = int(
            os.getenv("MINERU_API_MAX_CONCURRENT_REQUESTS", "0")
        )
    except ValueError:
        max_concurrent_requests = 0

    if max_concurrent_requests > 0:
        _request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        logger.info(f"Request concurrency limited to {max_concurrent_requests}")

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    return app


app = create_app()


def sanitize_filename(filename: str) -> str:
    """
    Format the file name of the compressed file
    Remove path traversal characters, reserve Unicode letters, numbers,._-
    Disable hidden files
    """
    sanitized = re.sub(r"[/\\.]{2,}|[/\\]", "", filename)
    sanitized = re.sub(r"[^\w.-]", "_", sanitized, flags=re.UNICODE)
    if sanitized.startswith("."):
        sanitized = "_" + sanitized[1:]
    return sanitized or "unnamed"


def cleanup_file(file_path: str) -> None:
    """Clean temporary files or directories"""
    try:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    except Exception as e:
        logger.warning(f"fail clean file {file_path}: {e}")


def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


def get_infer_result(
    file_suffix_identifier: str, pdf_name: str, parse_dir: str
) -> Optional[str]:
    """Read inference results from results file"""
    result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
    if os.path.exists(result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as fp:
            return fp.read()
    return None


@app.post(path="/file_parse", dependencies=[Depends(limit_concurrency)])
async def parse_pdf(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(
        ..., description="Upload pdf or image files for parsing"
    ),
    output_dir: str = Form("./output", description="Output local directory"),
    lang_list: List[str] = Form(
        ["ch"],
        description="""(Adapted only for pipeline and hybrid backend)Input the languages in the pdf to improve OCR accuracy.Options:
- ch: Chinese, English, Chinese Traditional.
- ch_lite: Chinese, English, Chinese Traditional, Japanese.
- ch_server: Chinese, English, Chinese Traditional, Japanese.
- en: English.
- korean: Korean, English.
- japan: Chinese, English, Chinese Traditional, Japanese.
- chinese_cht: Chinese, English, Chinese Traditional, Japanese.
- ta: Tamil, English.
- te: Telugu, English.
- ka: Kannada.
- th: Thai, English.
- el: Greek, English.
- latin: French, German, Afrikaans, Italian, Spanish, Bosnian, Portuguese, Czech, Welsh, Danish, Estonian, Irish, Croatian, Uzbek, Hungarian, Serbian (Latin), Indonesian, Occitan, Icelandic, Lithuanian, Maori, Malay, Dutch, Norwegian, Polish, Slovak, Slovenian, Albanian, Swedish, Swahili, Tagalog, Turkish, Latin, Azerbaijani, Kurdish, Latvian, Maltese, Pali, Romanian, Vietnamese, Finnish, Basque, Galician, Luxembourgish, Romansh, Catalan, Quechua.
- arabic: Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi, English.
- east_slavic: Russian, Belarusian, Ukrainian, English.
- cyrillic: Russian, Belarusian, Ukrainian, Serbian (Cyrillic), Bulgarian, Mongolian, Abkhazian, Adyghe, Kabardian, Avar, Dargin, Ingush, Chechen, Lak, Lezgin, Tabasaran, Kazakh, Kyrgyz, Tajik, Macedonian, Tatar, Chuvash, Bashkir, Malian, Moldovan, Udmurt, Komi, Ossetian, Buryat, Kalmyk, Tuvan, Sakha, Karakalpak, English.
- devanagari: Hindi, Marathi, Nepali, Bihari, Maithili, Angika, Bhojpuri, Magahi, Santali, Newari, Konkani, Sanskrit, Haryanvi, English.
""",
    ),
    backend: str = Form(
        "hybrid-auto-engine",
        description="""The backend for parsing:
- pipeline: More general, supports multiple languages, hallucination-free.
- vlm-auto-engine: High accuracy via local computing power, supports Chinese and English documents only.
- vlm-http-client: High accuracy via remote computing power(client suitable for openai-compatible servers), supports Chinese and English documents only.
- hybrid-auto-engine: Next-generation high accuracy solution via local computing power, supports multiple languages.
- hybrid-http-client: High accuracy via remote computing power but requires a little local computing power(client suitable for openai-compatible servers), supports multiple languages.""",
    ),
    parse_method: str = Form(
        "auto",
        description="""(Adapted only for pipeline and hybrid backend)The method for parsing PDF:
- auto: Automatically determine the method based on the file type
- txt: Use text extraction method
- ocr: Use OCR method for image-based PDFs
""",
    ),
    formula_enable: bool = Form(True, description="Enable formula parsing."),
    table_enable: bool = Form(True, description="Enable table parsing."),
    server_url: Optional[str] = Form(
        None,
        description="(Adapted only for <vlm/hybrid>-http-client backend)openai compatible server url, e.g., http://127.0.0.1:30000",
    ),
    return_md: bool = Form(True, description="Return markdown content in response"),
    return_middle_json: bool = Form(
        False, description="Return middle JSON in response"
    ),
    return_model_output: bool = Form(
        False, description="Return model output JSON in response"
    ),
    return_content_list: bool = Form(
        False, description="Return content list JSON in response"
    ),
    return_images: bool = Form(
        False, description="Return extracted images in response"
    ),
    response_format_zip: bool = Form(
        False, description="Return results as a ZIP file instead of JSON"
    ),
    start_page_id: int = Form(
        0, description="The starting page for PDF parsing, beginning from 0"
    ),
    end_page_id: int = Form(
        99999, description="The ending page for PDF parsing, beginning from 0"
    ),
):
    # Get command line configuration parameters
    config = getattr(app.state, "config", {})

    try:
        # Create a unique output directory
        unique_dir = os.path.join(output_dir, str(uuid.uuid4()))
        os.makedirs(unique_dir, exist_ok=True)
        background_tasks.add_task(cleanup_file, unique_dir)

        # Process uploaded PDF files
        pdf_file_names = []

        for file in files:
            content = await file.read()
            file_path = Path(file.filename)

            # Create temporary files
            temp_path = Path(unique_dir) / file_path.name
            with open(temp_path, "wb") as f:
                f.write(content)

            # If it is an image file or PDF, use read_fn to process it
            file_suffix = guess_suffix_by_path(temp_path)
            if file_suffix in pdf_suffixes + image_suffixes:
                try:
                    pdf_bytes = read_fn(temp_path)
                    pdf_name = file_path.stem
                    pdf_file_names.append(pdf_name)

                    # Select a language for the current file, falling back to the first language or default if the length does not match ch
                    if len(lang_list) > len(pdf_file_names) - 1:
                        cur_lang = lang_list[len(pdf_file_names) - 1]
                    else:
                        cur_lang = lang_list[0] if lang_list else "ch"

                    await aio_do_parse(
                        output_dir=unique_dir,
                        pdf_file_names=[pdf_name],
                        pdf_bytes_list=[pdf_bytes],
                        p_lang_list=[cur_lang],
                        backend=backend,
                        parse_method=parse_method,
                        formula_enable=formula_enable,
                        table_enable=table_enable,
                        server_url=server_url,
                        f_draw_layout_bbox=False,
                        f_draw_span_bbox=False,
                        f_dump_md=return_md,
                        f_dump_middle_json=return_middle_json,
                        f_dump_model_output=return_model_output,
                        f_dump_orig_pdf=False,
                        f_dump_content_list=return_content_list,
                        start_page_id=start_page_id,
                        end_page_id=end_page_id,
                        **config,
                    )

                    del pdf_bytes
                    os.remove(temp_path)  # Delete temporary files
                except Exception as e:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Failed to load file: {str(e)}"},
                    )
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unsupported file type: {file_suffix}"},
                )

        if not pdf_file_names:
            return JSONResponse(status_code=400, content={"error": "No valid files to parse"})

        # according to response_format_zip Determine return type
        if response_format_zip:
            zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="mineru_results_")
            os.close(zip_fd)
            background_tasks.add_task(cleanup_file, zip_path)

            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for pdf_name in pdf_file_names:
                    safe_pdf_name = sanitize_filename(pdf_name)

                    if backend.startswith("pipeline"):
                        parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
                    elif backend.startswith("vlm"):
                        parse_dir = os.path.join(unique_dir, pdf_name, "vlm")
                    elif backend.startswith("hybrid"):
                        parse_dir = os.path.join(
                            unique_dir, pdf_name, f"hybrid_{parse_method}"
                        )
                    else:
                        # unknown backend，Skip this file
                        logger.warning(f"Unknown backend type: {backend}, skipping {pdf_name}")
                        continue

                    if not os.path.exists(parse_dir):
                        continue

                    # Write text result
                    if return_md:
                        path = os.path.join(parse_dir, f"{pdf_name}.md")
                        if os.path.exists(path):
                            zf.write(
                                path,
                                arcname=os.path.join(
                                    safe_pdf_name, f"{safe_pdf_name}.md"
                                ),
                            )

                    if return_middle_json:
                        path = os.path.join(parse_dir, f"{pdf_name}_middle.json")
                        if os.path.exists(path):
                            zf.write(
                                path,
                                arcname=os.path.join(
                                    safe_pdf_name, f"{safe_pdf_name}_middle.json"
                                ),
                            )

                    if return_model_output:
                        path = os.path.join(parse_dir, f"{pdf_name}_model.json")
                        if os.path.exists(path):
                            zf.write(
                                path,
                                arcname=os.path.join(
                                    safe_pdf_name, f"{safe_pdf_name}_model.json"
                                ),
                            )

                    if return_content_list:
                        path = os.path.join(parse_dir, f"{pdf_name}_content_list.json")
                        if os.path.exists(path):
                            zf.write(
                                path,
                                arcname=os.path.join(
                                    safe_pdf_name, f"{safe_pdf_name}_content_list.json"
                                ),
                            )

                    # Write picture
                    if return_images:
                        images_dir = os.path.join(parse_dir, "images")
                        image_paths = glob.glob(
                            os.path.join(glob.escape(images_dir), "*.jpg")
                        )
                        for image_path in image_paths:
                            zf.write(
                                image_path,
                                arcname=os.path.join(
                                    safe_pdf_name,
                                    "images",
                                    os.path.basename(image_path),
                                ),
                            )

            return FileResponse(
                path=zip_path,
                media_type="application/zip",
                filename="results.zip",
            )
        else:
            # build JSON result
            result_dict = {}
            for pdf_name in pdf_file_names:
                result_dict[pdf_name] = {}
                data = result_dict[pdf_name]

                if backend.startswith("pipeline"):
                    parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
                elif backend.startswith("vlm"):
                    parse_dir = os.path.join(unique_dir, pdf_name, "vlm")
                elif backend.startswith("hybrid"):
                    parse_dir = os.path.join(
                        unique_dir, pdf_name, f"hybrid_{parse_method}"
                    )
                else:
                    # unknown backend，Skip this file
                    logger.warning(f"Unknown backend type: {backend}, skipping {pdf_name}")
                    continue

                if os.path.exists(parse_dir):
                    if return_md:
                        data["md_content"] = get_infer_result(
                            ".md", pdf_name, parse_dir
                        )
                    if return_middle_json:
                        data["middle_json"] = get_infer_result(
                            "_middle.json", pdf_name, parse_dir
                        )
                    if return_model_output:
                        data["model_output"] = get_infer_result(
                            "_model.json", pdf_name, parse_dir
                        )
                    if return_content_list:
                        data["content_list"] = get_infer_result(
                            "_content_list.json", pdf_name, parse_dir
                        )
                    if return_images:
                        images_dir = os.path.join(parse_dir, "images")
                        safe_pattern = os.path.join(glob.escape(images_dir), "*.jpg")
                        image_paths = glob.glob(safe_pattern)
                        data["images"] = {
                            os.path.basename(
                                image_path
                            ): f"data:image/jpeg;base64,{encode_image(image_path)}"
                            for image_path in image_paths
                        }

            return JSONResponse(
                status_code=200,
                content={
                    "backend": backend,
                    "version": __version__,
                    "results": result_dict,
                },
            )
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            status_code=500, content={"error": f"Failed to process file: {str(e)}"}
        )


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.pass_context
@click.option("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
@click.option("--port", default=8000, type=int, help="Server port (default: 8000)")
@click.option("--reload", is_flag=True, help="Enable auto-reload (development mode)")
@click.option(
    "--config-path",
    "--config",
    "config_path",
    type=click.Path(),
    default="./mineru.json",
    help="Path to mineru.json configuration file",
)
def main(ctx, host, port, reload, config_path, **kwargs):
    if config_path:
        os.environ['MINERU_TOOLS_CONFIG_JSON'] = str(Path(config_path).expanduser().resolve())

    kwargs.update(arg_parse(ctx))

    # Store configuration parameters into application state
    app.state.config = kwargs

    # Will CLI The concurrency parameters are synchronized to environment variables to ensure uvicorn Overloaded child processes are visible
    try:
        mcr = int(kwargs.get("mineru_api_max_concurrent_requests", 0) or 0)
    except ValueError:
        mcr = 0
    os.environ["MINERU_API_MAX_CONCURRENT_REQUESTS"] = str(mcr)

    """Start MinerU FastAPIServer command line entry"""
    print(f"Start MinerU FastAPI Service: http://{host}:{port}")
    print(f"API documentation: http://{host}:{port}/docs")

    uvicorn.run("mineru.cli.fast_api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
