# Copyright (c) Opendatalab. All rights reserved.

import base64
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import click
import gradio as gr
from gradio_pdf import PDF
from loguru import logger

log_level = os.getenv("MINERU_LOG_LEVEL", "INFO").upper()
logger.remove()  # Remove default handler
logger.add(sys.stderr, level=log_level)  # Add new handler

from mineru.cli.common import prepare_env, read_fn, aio_do_parse, pdf_suffixes, image_suffixes
from mineru.utils.cli_parser import arg_parse
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.hash_utils import str_sha256


async def parse_pdf(doc_path, output_dir, end_page_id, is_ocr, formula_enable, table_enable, language, backend, url):
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_name = f'{safe_stem(Path(doc_path).stem)}_{time.strftime("%y%m%d_%H%M%S")}'
        pdf_data = read_fn(doc_path)
        # according to backend Sure parse_method
        if backend.startswith("vlm"):
            parse_method = "vlm"
        else:
            parse_method = 'ocr' if is_ocr else 'auto'

        # according to backend Type preparation environment directory
        if backend.startswith("hybrid"):
            env_name = f"hybrid_{parse_method}"
        else:
            env_name = parse_method

        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, env_name)

        await aio_do_parse(
            output_dir=output_dir,
            pdf_file_names=[file_name],
            pdf_bytes_list=[pdf_data],
            p_lang_list=[language],
            parse_method=parse_method,
            end_page_id=end_page_id,
            formula_enable=formula_enable,
            table_enable=table_enable,
            backend=backend,
            server_url=url,
        )
        return local_md_dir, file_name
    except Exception as e:
        logger.exception(e)
        return None


def compress_directory_to_zip(directory_path, output_zip_path):
    """Compress the specified directory into a ZIP document.

    :param directory_path: Directory path to compress
    :param output_zip_path: Output ZIP file path
    """
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

            # Iterate through all files and subdirectories in a directory
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    # Build full file path
                    file_path = os.path.join(root, file)
                    # Calculate relative paths
                    arcname = os.path.relpath(file_path, directory_path)
                    # Add files to ZIP document
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def replace_image_with_base64(markdown_text, image_dir_path):
    # Match image tags in Markdown
    pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'

    # Replace image link
    def replace(match):
        relative_path = match.group(1)
        # Only process with.jpgEnding picture
        if relative_path.endswith('.jpg'):
            full_path = os.path.join(image_dir_path, relative_path)
            base64_image = image_to_base64(full_path)
            return f'![{relative_path}](data:image/jpeg;base64,{base64_image})'
        else:
            # Images in other formats remain as is
            return match.group(0)
    # Apply replacement
    return re.sub(pattern, replace, markdown_text)


async def to_markdown(file_path, end_pages=10, is_ocr=False, formula_enable=True, table_enable=True, language="ch", backend="pipeline", url=None):
    # If language contains (), extract the content before the brackets as the actual language
    if '(' in language and ')' in language:
        language = language.split('(')[0].strip()
    file_path = to_pdf(file_path)
    # Get the recognized md file and compressed package file path
    local_md_dir, file_name = await parse_pdf(file_path, './output', end_pages - 1, is_ocr, formula_enable, table_enable, language, backend, url)
    archive_zip_path = os.path.join('./output', str_sha256(local_md_dir) + '.zip')
    zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
    if zip_archive_success == 0:
        logger.info('Compression successful')
    else:
        logger.error('Compression failed')
    md_path = os.path.join(local_md_dir, file_name + '.md')
    with open(md_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    md_content = replace_image_with_base64(txt_content, local_md_dir)
    # Returns the converted PDF path
    new_pdf_path = os.path.join(local_md_dir, file_name + '_layout.pdf')

    return md_content, txt_content, archive_zip_path, new_pdf_path


latex_delimiters_type_a = [
    {'left': '$$', 'right': '$$', 'display': True},
    {'left': '$', 'right': '$', 'display': False},
]
latex_delimiters_type_b = [
    {'left': '\\(', 'right': '\\)', 'display': False},
    {'left': '\\[', 'right': '\\]', 'display': True},
]
latex_delimiters_type_all = latex_delimiters_type_a + latex_delimiters_type_b

header_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'header.html')
with open(header_path, mode='r', encoding='utf-8') as header_file:
    header = header_file.read()

other_lang = [
    'ch (Chinese, English, Chinese Traditional)',
    'ch_lite (Chinese, English, Chinese Traditional, Japanese)',
    'ch_server (Chinese, English, Chinese Traditional, Japanese)',
    'en (English)',
    'korean (Korean, English)',
    'japan (Chinese, English, Chinese Traditional, Japanese)',
    'chinese_cht (Chinese, English, Chinese Traditional, Japanese)',
    'ta (Tamil, English)',
    'te (Telugu, English)',
    'ka (Kannada)',
    'el (Greek, English)',
    'th (Thai, English)'
]
add_lang = [
    'latin (French, German, Afrikaans, Italian, Spanish, Bosnian, Portuguese, Czech, Welsh, Danish, Estonian, Irish, Croatian, Uzbek, Hungarian, Serbian (Latin), Indonesian, Occitan, Icelandic, Lithuanian, Maori, Malay, Dutch, Norwegian, Polish, Slovak, Slovenian, Albanian, Swedish, Swahili, Tagalog, Turkish, Latin, Azerbaijani, Kurdish, Latvian, Maltese, Pali, Romanian, Vietnamese, Finnish, Basque, Galician, Luxembourgish, Romansh, Catalan, Quechua)',
    'arabic (Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi, English)',
    'east_slavic (Russian, Belarusian, Ukrainian, English)',
    'cyrillic (Russian, Belarusian, Ukrainian, Serbian (Cyrillic), Bulgarian, Mongolian, Abkhazian, Adyghe, Kabardian, Avar, Dargin, Ingush, Chechen, Lak, Lezgin, Tabasaran, Kazakh, Kyrgyz, Tajik, Macedonian, Tatar, Chuvash, Bashkir, Malian, Moldovan, Udmurt, Komi, Ossetian, Buryat, Kalmyk, Tuvan, Sakha, Karakalpak, English)',
    'devanagari (Hindi, Marathi, Nepali, Bihari, Maithili, Angika, Bhojpuri, Magahi, Santali, Newari, Konkani, Sanskrit, Haryanvi, English)'
]
all_lang = [*other_lang, *add_lang]


def safe_stem(file_path):
    stem = Path(file_path).stem
    # Only letters, numbers, underscores, and dots are retained, and other characters are replaced with underscores.
    return re.sub(r'[^\w.]', '_', stem)


def to_pdf(file_path):

    if file_path is None:
        return None

    pdf_bytes = read_fn(file_path)

    # unique_filename = f'{uuid.uuid4()}.pdf'
    unique_filename = f'{safe_stem(file_path)}.pdf'

    # Build full file path
    tmp_file_path = os.path.join(os.path.dirname(file_path), unique_filename)

    # Write byte data to file
    with open(tmp_file_path, 'wb') as tmp_pdf_file:
        tmp_pdf_file.write(pdf_bytes)

    return tmp_file_path


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option(
    '--enable-example',
    'example_enable',
    type=bool,
    help="Enable example files for input."
         "The example files to be input need to be placed in the `example` folder within the directory where the command is currently executed.",
    default=True,
)
@click.option(
    '--enable-http-client',
    'http_client_enable',
    type=bool,
    help="Enable http-client backend to link openai-compatible servers.",
    default=False,
)
@click.option(
    '--enable-api',
    'api_enable',
    type=bool,
    help="Enable gradio API for serving the application.",
    default=True,
)
@click.option(
    '--max-convert-pages',
    'max_convert_pages',
    type=int,
    help="Set the maximum number of pages to convert from PDF to Markdown.",
    default=1000,
)
@click.option(
    '--server-name',
    'server_name',
    type=str,
    help="Set the server name for the Gradio app.",
    default=None,
)
@click.option(
    '--server-port',
    'server_port',
    type=int,
    help="Set the server port for the Gradio app.",
    default=None,
)
@click.option(
    '--latex-delimiters-type',
    'latex_delimiters_type',
    type=click.Choice(['a', 'b', 'all']),
    help="Set the type of LaTeX delimiters to use in Markdown rendering:"
         "'a' for type '$', 'b' for type '()[]', 'all' for both types.",
    default='all',
)
@click.option(
    '--config-path',
    '--config',
    'config_path',
    type=click.Path(),
    help='Path to mineru.json configuration file.',
    default='./mineru.json',
)
def main(ctx,
        example_enable,
        http_client_enable,
        api_enable, max_convert_pages,
        server_name, server_port, latex_delimiters_type, config_path, **kwargs
):

    if config_path:
        os.environ['MINERU_TOOLS_CONFIG_JSON'] = str(Path(config_path).expanduser().resolve())

    # create i18n Example, supports Chinese and English
    i18n = gr.I18n(
        en={
            "upload_file": "Please upload a PDF or image",
            "max_pages": "Max convert pages",
            "backend": "Backend",
            "server_url": "Server URL",
            "server_url_info": "OpenAI-compatible server URL for http-client backend.",
            "recognition_options": "**Recognition Options:**",
            "table_enable": "Enable table recognition",
            "table_info": "If disabled, tables will be shown as images.",
            "formula_label_vlm": "Enable display formula recognition",
            "formula_label_pipeline": "Enable formula recognition",
            "formula_label_hybrid": "Enable inline formula recognition",
            "formula_info_vlm": "If disabled, display formulas will be shown as images.",
            "formula_info_pipeline": "If disabled, display formulas will be shown as images, and inline formulas will not be detected or parsed.",
            "formula_info_hybrid": "If disabled, inline formulas will not be detected or parsed.",
            "ocr_language": "OCR Language",
            "ocr_language_info": "Select the OCR language for image-based PDFs and images.",
            "force_ocr": "Force enable OCR",
            "force_ocr_info": "Enable only if the result is extremely poor. Requires correct OCR language.",
            "convert": "Convert",
            "clear": "Clear",
            "pdf_preview": "PDF preview",
            "examples": "Examples:",
            "convert_result": "Convert result",
            "md_rendering": "Markdown rendering",
            "md_text": "Markdown text",
            "backend_info_vlm": "High-precision parsing via VLM, supports Chinese and English documents only.",
            "backend_info_pipeline": "Traditional Multi-model pipeline parsing, supports multiple languages, hallucination-free.",
            "backend_info_hybrid": "High-precision hybrid parsing, supports multiple languages.",
            "backend_info_default": "Select the backend engine for document parsing.",
        },
        zh={
            "upload_file": "Please upload PDF or picture",
            "max_pages": "Maximum number of conversion pages",
            "backend": "parsing backend",
            "server_url": "Server address",
            "server_url_info": "http-client backend OpenAI Compatible server address.",
            "recognition_options": "**Identification options:**",
            "table_enable": "Enable table recognition",
            "table_info": "When disabled, the table will be displayed as a picture.",
            "formula_label_vlm": "Enable interline formula recognition",
            "formula_label_pipeline": "Enable formula recognition",
            "formula_label_hybrid": "Enable inline formula recognition",
            "formula_info_vlm": "When disabled, inline formulas will appear as pictures.",
            "formula_info_pipeline": "When disabled, inline formulas will be displayed as images and inline formulas will not be detected or parsed.",
            "formula_info_hybrid": "When disabled, inline formulas will not be detected or parsed.",
            "ocr_language": "OCR language",
            "ocr_language_info": "for scanned version PDF and picture selection OCR language.",
            "force_ocr": "Force enable OCR",
            "force_ocr_info": "Only enable it when the recognition effect is extremely poor. You need to select the correct OCR language.",
            "convert": "Convert",
            "clear": "Clear",
            "pdf_preview": "PDF Preview",
            "examples": "Example:",
            "convert_result": "Conversion result",
            "md_rendering": "Markdown rendering",
            "md_text": "Markdown text",
            "backend_info_vlm": "High-precision analysis of multi-modal large models, only supports Chinese and English documents.",
            "backend_info_pipeline": "Traditional multi-model pipeline parsing supports multiple languages ​​and has no illusions.",
            "backend_info_hybrid": "High-precision mixed analysis, supporting multiple languages.",
            "backend_info_default": "Select the backend engine for document parsing.",
        },
    )

    # Get the formula identification label based on the backend type (closure function to support i18n）
    def get_formula_label(backend_choice):
        if backend_choice.startswith("vlm"):
            return i18n("formula_label_vlm")
        elif backend_choice == "pipeline":
            return i18n("formula_label_pipeline")
        elif backend_choice.startswith("hybrid"):
            return i18n("formula_label_hybrid")
        else:
            return i18n("formula_label_pipeline")

    def get_formula_info(backend_choice):
        if backend_choice.startswith("vlm"):
            return i18n("formula_info_vlm")
        elif backend_choice == "pipeline":
            return i18n("formula_info_pipeline")
        elif backend_choice.startswith("hybrid"):
            return i18n("formula_info_hybrid")
        else:
            return ""

    def get_backend_info(backend_choice):
        if backend_choice.startswith("vlm"):
            return i18n("backend_info_vlm")
        elif backend_choice == "pipeline":
            return i18n("backend_info_pipeline")
        elif backend_choice.startswith("hybrid"):
            return i18n("backend_info_hybrid")
        else:
            return i18n("backend_info_default")

    # Update interface function
    def update_interface(backend_choice):
        formula_label_update = gr.update(label=get_formula_label(backend_choice), info=get_formula_info(backend_choice))
        backend_info_update = gr.update(info=get_backend_info(backend_choice))
        if "http-client" in backend_choice:
            client_options_update = gr.update(visible=True)
        else:
            client_options_update = gr.update(visible=False)
        if "vlm" in backend_choice:
            ocr_options_update = gr.update(visible=False)
        else:
            ocr_options_update = gr.update(visible=True)

        return client_options_update, ocr_options_update, formula_label_update, backend_info_update


    kwargs.update(arg_parse(ctx))

    if latex_delimiters_type == 'a':
        latex_delimiters = latex_delimiters_type_a
    elif latex_delimiters_type == 'b':
        latex_delimiters = latex_delimiters_type_b
    elif latex_delimiters_type == 'all':
        latex_delimiters = latex_delimiters_type_all
    else:
        raise ValueError(f"Invalid latex delimiters type: {latex_delimiters_type}.")

    vlm_engine = get_vlm_engine("auto", is_async=True)
    if vlm_engine in ["transformers", "mlx-engine"]:
        http_client_enable = True
    else:
        try:
            logger.info(f"Start init {vlm_engine}...")
            from mineru.backend.vlm.vlm_analyze import ModelSingleton
            model_singleton = ModelSingleton()
            predictor = model_singleton.get_model(
                vlm_engine,
                None,
                None,
                **kwargs
            )
            logger.info(f"{vlm_engine} init successfully.")
        except Exception as e:
            logger.exception(e)

    suffixes = [f".{suffix}" for suffix in pdf_suffixes + image_suffixes]
    with gr.Blocks() as demo:
        gr.HTML(header)
        with gr.Row():
            with gr.Column(variant='panel', scale=5):
                with gr.Row():
                    input_file = gr.File(label=i18n("upload_file"), file_types=suffixes)
                with gr.Row():
                    max_pages = gr.Slider(1, max_convert_pages, max_convert_pages, step=1, label=i18n("max_pages"))
                with gr.Row():
                    drop_list = ["pipeline", "vlm-auto-engine", "hybrid-auto-engine"]
                    preferred_option = "hybrid-auto-engine"
                    if http_client_enable:
                        drop_list.extend(["vlm-http-client", "hybrid-http-client"])
                    backend = gr.Dropdown(drop_list, label=i18n("backend"), value=preferred_option, info=get_backend_info(preferred_option))
                with gr.Row(visible=False) as client_options:
                    url = gr.Textbox(label=i18n("server_url"), value='http://localhost:30000', placeholder='http://localhost:30000', info=i18n("server_url_info"))
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown(i18n("recognition_options"))
                        table_enable = gr.Checkbox(label=i18n("table_enable"), value=True, info=i18n("table_info"))
                        formula_enable = gr.Checkbox(label=get_formula_label(preferred_option), value=True, info=get_formula_info(preferred_option))
                    with gr.Column(visible=False) as ocr_options:
                        language = gr.Dropdown(all_lang, label=i18n("ocr_language"), value='ch (Chinese, English, Chinese Traditional)', info=i18n("ocr_language_info"))
                        is_ocr = gr.Checkbox(label=i18n("force_ocr"), value=False, info=i18n("force_ocr_info"))
                with gr.Row():
                    change_bu = gr.Button(i18n("convert"))
                    clear_bu = gr.ClearButton(value=i18n("clear"))
                pdf_show = PDF(label=i18n("pdf_preview"), interactive=False, visible=True, height=800)
                if example_enable:
                    example_root = os.path.join(os.getcwd(), 'examples')
                    if os.path.exists(example_root):
                        gr.Examples(
                            label=i18n("examples"),
                            examples=[os.path.join(example_root, _) for _ in os.listdir(example_root) if
                                      _.endswith(tuple(suffixes))],
                            inputs=input_file
                        )

            with gr.Column(variant='panel', scale=5):
                output_file = gr.File(label=i18n("convert_result"), interactive=False)
                with gr.Tabs():
                    with gr.Tab(i18n("md_rendering")):
                        md = gr.Markdown(
                            label=i18n("md_rendering"),
                            height=1200,
                            # buttons=["copy"],  # gradio 6 Use the above version
                            show_copy_button=True,  # gradio 6 The following versions are used
                            latex_delimiters=latex_delimiters,
                            line_breaks=True
                        )
                    with gr.Tab(i18n("md_text")):
                        md_text = gr.TextArea(
                            lines=45,
                            # buttons=["copy"],  # gradio 6 Use the above version
                            show_copy_button=True,  # gradio 6 The following versions are used
                            label=i18n("md_text")
                        )

        # Add event handling
        backend.change(
            fn=update_interface,
            inputs=[backend],
            outputs=[client_options, ocr_options, formula_enable, backend],
            # api_visibility="private"  # gradio 6 Use the above version
            api_name=False  # gradio 6 The following versions are used
        )
        # Add demo.loadEvent, triggers an interface update when the page is loaded
        demo.load(
            fn=update_interface,
            inputs=[backend],
            outputs=[client_options, ocr_options, formula_enable, backend],
            # api_visibility="private"  # gradio 6 Use the above version
            api_name=False  # gradio 6 The following versions are used
        )
        clear_bu.add([input_file, md, pdf_show, md_text, output_file, is_ocr])

        input_file.change(
            fn=to_pdf,
            inputs=input_file,
            outputs=pdf_show,
            api_name="to_pdf" if api_enable else False,  # gradio 6 The following versions are used
            # api_visibility="public" if api_enable else "private"  # gradio 6 Use the above version
        )
        change_bu.click(
            fn=to_markdown,
            inputs=[input_file, max_pages, is_ocr, formula_enable, table_enable, language, backend, url],
            outputs=[md, md_text, output_file, pdf_show],
            api_name="to_markdown" if api_enable else False,  # gradio 6 The following versions are used
            # api_visibility="public" if api_enable else "private"  # gradio 6 Use the above version
        )

    footer_links = ["gradio", "settings"]
    if api_enable:
        footer_links.append("api")
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        # footer_links=footer_links,  # gradio 6 Use the above version
        show_api=api_enable,  # gradio 6 The following versions are used
        i18n=i18n
    )


if __name__ == '__main__':
    main()
