# Copyright (c) Opendatalab. All rights reserved.
import io
import json
import os
import copy
import gc
import re
import shutil
from pathlib import Path
from typing import Any

from loguru import logger
import pypdfium2 as pdfium

from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox, draw_line_sort_bbox
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.enum_class import MakeMode
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_bytes
from mineru.utils.pdf_image_tools import images_bytes_to_pdf_bytes
from mineru.utils.pdf_page_id import get_end_page_id


pdf_suffixes = ["pdf"]
image_suffixes = ["png", "jpeg", "jp2", "webp", "gif", "bmp", "jpg", "tiff"]

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _trace_stage(stage: str, **kwargs):
    return


def _release_image_list(images_list):
    if not images_list:
        return
    for image_dict in images_list:
        if not isinstance(image_dict, dict):
            continue
        img_pil = image_dict.get("img_pil")
        if img_pil is not None:
            try:
                img_pil.close()
            except Exception:
                pass
    images_list.clear()


def _stream_json_array_start(fp, meta_obj: dict, array_key: str) -> None:
    meta_json = json.dumps(meta_obj, ensure_ascii=False)
    if meta_json == "{}":
        fp.write(f'{{"{array_key}":[')
    else:
        fp.write(meta_json[:-1])
        fp.write(f',"{array_key}":[')


def _append_text_file(path: str, text: Any) -> None:
    if text is None:
        return
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def _count_trailing_newlines(text: str) -> int:
    count = 0
    for ch in reversed(text):
        if ch == "\n":
            count += 1
        else:
            break
    return count

def set_env(enable_cache, config_json_path=None, ov_cache_size=None):
    os.environ['MINERU_MODEL_SOURCE'] = 'local'
    os.environ['YOLO_VERBOSE'] = 'False'
    os.environ.pop('MINERU_DONOT_CLEAN_MEM', None)

    if config_json_path:
        os.environ['MINERU_TOOLS_CONFIG_JSON'] = str(Path(config_json_path).expanduser().resolve())
    if ov_cache_size is not None:
        os.environ['CPU_RUNTIME_CACHE_CAPACITY'] = str(ov_cache_size)

    if enable_cache:
        os.environ['MINERU_PDF_RENDER_THREADS'] = '16'
        os.environ['MINERU_MIN_BATCH_INFERENCE_SIZE'] = '512'
        os.environ['MINERU_OCR_REC_CHUNK_SIZE'] = '256'
        os.environ['MINERU_FORCE_DISABLE_OCR_DET_BATCH'] = '0'
        os.environ['MINERU_TABLE_CONSUME_CHUNK_SIZE'] = '16'
        os.environ['MINERU_DONOT_CLEAN_MEM'] = '1'
    else:
        os.environ['MINERU_PDF_RENDER_THREADS'] = '1'
        os.environ['MINERU_MIN_BATCH_INFERENCE_SIZE'] = '1'
        os.environ['MINERU_OCR_REC_CHUNK_SIZE'] = '1'
        os.environ['MINERU_FORCE_DISABLE_OCR_DET_BATCH'] = '1'
        os.environ['MINERU_TABLE_CONSUME_CHUNK_SIZE'] = '1'
        os.environ['CPU_RUNTIME_CACHE_CAPACITY'] = '0'
        os.environ['MINERU_PAGE_CHUNK_SIZE'] = '1'

def read_fn(path):
    if not isinstance(path, Path):
        path = Path(path)
    with open(str(path), "rb") as input_file:
        file_bytes = input_file.read()
        file_suffix = guess_suffix_by_bytes(file_bytes, path)
        if file_suffix in image_suffixes:
            return images_bytes_to_pdf_bytes(file_bytes)
        elif file_suffix in pdf_suffixes:
            return file_bytes
        else:
            raise Exception(f"Unknown file suffix: {file_suffix}")


def prepare_env(output_dir, pdf_file_name, parse_method):
    # local_md_dir = str(os.path.join(output_dir, pdf_file_name, parse_method))
    local_md_dir = str(os.path.join(output_dir, pdf_file_name))
    local_image_dir = os.path.join(str(local_md_dir), "images")
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir


def convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id=0, end_page_id=None):
    pdf = pdfium.PdfDocument(pdf_bytes)
    output_pdf = pdfium.PdfDocument.new()
    try:
        end_page_id = get_end_page_id(end_page_id, len(pdf))

        # Import page by page,Skip if failed
        output_index = 0
        for page_index in range(start_page_id, end_page_id + 1):
            try:
                output_pdf.import_pages(pdf, pages=[page_index])
                output_index += 1
            except Exception as page_error:
                output_pdf.del_page(output_index)
                logger.warning(f"Failed to import page {page_index}: {page_error}, skipping this page.")
                continue

        # Save new PDF to memory buffer
        output_buffer = io.BytesIO()
        output_pdf.save(output_buffer)

        # Get byte data
        output_bytes = output_buffer.getvalue()
    except Exception as e:
        logger.warning(f"Error in converting PDF bytes: {e}, Using original PDF bytes.")
        output_bytes = pdf_bytes
    pdf.close()
    output_pdf.close()
    return output_bytes


def _prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id):
    """Prepare to process PDF byte data"""
    if start_page_id == 0 and end_page_id is None:
        _trace_stage("prepare_pdf.skip")
        return list(pdf_bytes_list)
    result = []
    _trace_stage("prepare_pdf.begin", file_count=len(pdf_bytes_list), start=start_page_id, end=end_page_id)
    for pdf_bytes in pdf_bytes_list:
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
        result.append(new_pdf_bytes)
        _trace_stage("prepare_pdf.one_done", current_count=len(result))
    _trace_stage("prepare_pdf.end", file_count=len(result))
    return result


def _process_output(
        pdf_info,
        pdf_bytes,
        pdf_file_name,
        local_md_dir,
        local_image_dir,
        md_writer,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_orig_pdf,
        f_dump_md,
        f_dump_content_list,
        f_dump_middle_json,
        f_dump_model_output,
        f_make_md_mode,
        f_draw_line_sort_bbox,
        middle_json,
        model_output=None,
        is_pipeline=True
):
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
    """Process output files"""
    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

    if f_dump_orig_pdf:
        md_writer.write(
            f"{pdf_file_name}_origin.pdf",
            pdf_bytes,
        )

    if f_draw_line_sort_bbox:
        draw_line_sort_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_line_sort.pdf")

    image_dir = str(os.path.basename(local_image_dir))

    md_output = ''
    if f_dump_md:
        make_func = pipeline_union_make
        md_output = make_func(pdf_info, f_make_md_mode, image_dir)
        md_output = _split_images_by_md_references(local_md_dir, pdf_file_name, str(md_output) if md_output is not None else "")
        md_writer.write_string(f"{pdf_file_name}.md", md_output,)

    if f_dump_content_list:
        make_func = pipeline_union_make
        content_list = make_func(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )
        if not is_pipeline:
            content_list_v2 = make_func(pdf_info, MakeMode.CONTENT_LIST_V2, image_dir)
            md_writer.write_string(
                f"{pdf_file_name}_content_list_v2.json",
                json.dumps(content_list_v2, ensure_ascii=False, indent=4),
            )


    middle_json_str = ""
    if f_dump_middle_json:
        middle_json_str = json.dumps(middle_json, ensure_ascii=False, indent=4)
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            middle_json_str,
        )

    json_output = ''
    if f_dump_model_output:
        json_output = json.dumps(model_output, ensure_ascii=False, indent=4)
        md_writer.write_string(f"{pdf_file_name}_model.json", json_output,)
    elif f_dump_middle_json:
        json_output = middle_json_str

    logger.info(f"local output dir is {local_md_dir}")
    return md_output, json_output


def _split_images_by_md_references(local_md_dir: str, pdf_file_name: str, md_text: str | None = None) -> str:
    md_path = Path(local_md_dir) / f"{pdf_file_name}.md"
    images_dir = Path(local_md_dir) / "images"
    if not images_dir.exists():
        return md_text if md_text is not None else ""

    if md_text is None:
        if not md_path.exists():
            return ""
        md_text = md_path.read_text(encoding="utf-8", errors="ignore")

    image_refs = re.findall(r"!\[\]\(images/([^)]+)\)", md_text)
    used_image_files = set(image_refs)

    images_md_dir = Path(local_md_dir) / "images_md"
    images_intermediate_dir = Path(local_md_dir) / "images_intermediate"
    images_md_dir.mkdir(parents=True, exist_ok=True)
    images_intermediate_dir.mkdir(parents=True, exist_ok=True)

    for image_file in images_dir.iterdir():
        if not image_file.is_file():
            continue
        target_dir = images_md_dir if image_file.name in used_image_files else images_intermediate_dir
        target_path = target_dir / image_file.name
        if target_path.exists():
            target_path.unlink()
        shutil.move(str(image_file), str(target_path))

    md_text = md_text.replace("](images/", "](images_md/")
    md_path.write_text(md_text, encoding="utf-8")

    try:
        images_dir.rmdir()
    except OSError:
        pass

    return md_text


def init_BatchAnalyze(enable_cache, enable_ov, Layout_infer_type, MFD_infer_type, MFR_enc_infer_type, MFR_dec_infer_type,
                      OCR_det_infer_type, OCR_rec_infer_type, wired_table_type, WirelessTable_type, img_orientation_cls_type,
                      table_cls_type, layoutreader_type, nstreams, p_formula_enable=True, p_table_enable=True, remove_unused_weight=False):
    from mineru.backend.pipeline.batch_analyze import BatchAnalyze
    from mineru.backend.pipeline.pipeline_analyze import ModelSingleton, get_batch_info
    batch_ratio, enable_ocr_det_batch = get_batch_info()
    model_manager = ModelSingleton()
    batch_model = BatchAnalyze(model_manager, enable_cache, enable_ov, Layout_infer_type, MFD_infer_type, MFR_enc_infer_type,
                               MFR_dec_infer_type, OCR_det_infer_type, OCR_rec_infer_type, wired_table_type, WirelessTable_type,
                               img_orientation_cls_type, table_cls_type, layoutreader_type, nstreams, batch_ratio, p_formula_enable,
                               p_table_enable, enable_ocr_det_batch)
    if remove_unused_weight :
        batch_model.remove_unused_weight()
    return batch_model

def _process_pipeline_cache(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        p_lang_list,
        parse_method,
        p_formula_enable,
        p_table_enable,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_make_md_mode,
        f_draw_line_sort_bbox,
        batch_model,
        start_page_id=0,
        end_page_id=None,
        tqdm_enable = False
):
    """Handle pipeline back-end logic"""
    from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
   
    infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
        pipeline_doc_analyze(batch_model, pdf_bytes_list, p_lang_list,  parse_method=parse_method, tqdm_enable=tqdm_enable)
    )
    _trace_stage("pipeline_cache.analyze_done", file_count=len(infer_results))

    md_outputs = ""
    json_outputs = ""
    output_metas = []
    for idx, model_list in enumerate(infer_results):
        pdf_doc = None
        images_list = None
        model_json = None
        middle_json = None
        try:
            _trace_stage("pipeline_cache.doc_begin", index=idx, page_count=len(model_list))
            model_json = copy.deepcopy(model_list) if f_dump_model_output else None
            original_pdf_file_name = pdf_file_names[idx]
            pdf_file_name = original_pdf_file_name
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

            images_list = all_image_lists[idx]
            pdf_doc = all_pdf_docs[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]

            middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_doc, image_writer,
                                                         batch_model, _lang, _ocr_enable, p_formula_enable)

            pdf_info = middle_json["pdf_info"]
            pdf_bytes = pdf_bytes_list[idx]
            output_file_name = pdf_file_name
            if start_page_id != 0 or end_page_id is not None:
                if end_page_id is None:
                    end_page_id = len(pdf_doc) - 1 + start_page_id
                output_file_name = f"{pdf_file_name}_{start_page_id}_{end_page_id}"

            md_output, json_output = _process_output(pdf_info, pdf_bytes, output_file_name, local_md_dir, local_image_dir,
                            md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
                            f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
                            f_make_md_mode, f_draw_line_sort_bbox, middle_json, model_json, is_pipeline=True)
            if md_output:
                md_outputs += str(md_output)
            if json_output:
                json_outputs += str(json_output)

            output_metas.append(
                {
                    "input_name": original_pdf_file_name,
                    "output_dir": local_md_dir,
                    "output_file_name": output_file_name,
                    "md_path": os.path.join(local_md_dir, f"{output_file_name}.md"),
                    "images_md_dir": os.path.join(local_md_dir, "images_md"),
                    "middle_json_path": os.path.join(local_md_dir, f"{output_file_name}_middle.json"),
                    "model_json_path": os.path.join(local_md_dir, f"{output_file_name}_model.json"),
                }
            )
            _trace_stage("pipeline_cache.doc_output_done", index=idx)
        finally:
            infer_results[idx] = []
            _release_image_list(images_list)
            all_image_lists[idx] = []
            if pdf_doc is not None:
                try:
                    pdf_doc.close()
                except Exception:
                    pass
                all_pdf_docs[idx] = None
            del model_json
            del middle_json
            _trace_stage("pipeline_cache.doc_cleanup_done", index=idx)

    infer_results.clear()
    all_image_lists.clear()
    all_pdf_docs.clear()
    ocr_enabled_list.clear()
    gc.collect()
    _trace_stage("pipeline_cache.done")
    return md_outputs, json_outputs, output_metas

def _process_pipeline_nocache(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        p_lang_list,
        parse_method,
        p_formula_enable,
        p_table_enable,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_make_md_mode,
        f_draw_line_sort_bbox,
        batch_model,
        start_page_id=0,
        end_page_id=None,
        tqdm_enable = False
):
    """Handle pipeline back-end logic"""
    from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze_1by1 as pipeline_doc_analyze
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make

    md_outputs = ""
    json_outputs = ""
    output_metas = []
    page_chunk_size = max(1, int(os.getenv("MINERU_PAGE_CHUNK_SIZE", "1") or 1))
    emit_outputs = any([
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_draw_line_sort_bbox,
    ])
    _trace_stage("pipeline_nocache.begin", file_count=len(pdf_file_names), chunk=page_chunk_size)

    for pdf_bytes, p_lang, pdf_file_name in zip(pdf_bytes_list, p_lang_list, pdf_file_names):
        model_list = None
        images_list = None
        pdf_doc = None
        middle_fp = None
        model_fp = None
        middle_array_started = False
        middle_first_page = True
        model_array_started = False
        model_first_item = True
        md_has_content = False
        md_tail = ""
        try:
            parse_start = start_page_id
            src_doc = pdfium.PdfDocument(pdf_bytes)
            try:
                parse_end = get_end_page_id(end_page_id, len(src_doc))
            finally:
                src_doc.close()
            _trace_stage("pipeline_nocache.doc_begin", file=pdf_file_name, start=parse_start, end=parse_end)

            output_file_name = pdf_file_name
            if parse_start != 0 or end_page_id is not None:
                output_file_name = f"{pdf_file_name}_{parse_start}_{parse_end}"

            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
            image_dir = str(os.path.basename(local_image_dir))
            middle_json_path = os.path.join(local_md_dir, f"{output_file_name}_middle.json")
            model_json_path = os.path.join(local_md_dir, f"{output_file_name}_model.json")
            md_output_path = os.path.join(local_md_dir, f"{output_file_name}.md")

            if f_dump_md and os.path.exists(md_output_path):
                os.remove(md_output_path)

            if f_dump_orig_pdf:
                md_writer.write(f"{output_file_name}_origin.pdf", pdf_bytes)

            for chunk_start in range(parse_start, parse_end + 1, page_chunk_size):
                chunk_end = min(chunk_start + page_chunk_size - 1, parse_end)
                _trace_stage("pipeline_nocache.chunk_begin", file=pdf_file_name, chunk_start=chunk_start, chunk_end=chunk_end)
                chunk_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, chunk_start, chunk_end)
                _trace_stage("pipeline_nocache.chunk_pdf_ready", file=pdf_file_name, chunk_start=chunk_start, chunk_end=chunk_end, bytes=len(chunk_pdf_bytes))

                model_list, images_list, pdf_doc, _lang, _ocr_enable = (
                    pipeline_doc_analyze(batch_model, chunk_pdf_bytes, p_lang, parse_method=parse_method, start_page_id=0, end_page_id=None, page_index_offset=chunk_start, tqdm_enable=tqdm_enable)
                )
                _trace_stage("pipeline_nocache.chunk_analyze_done", file=pdf_file_name, chunk_start=chunk_start, pages=len(model_list))

                chunk_middle_json = pipeline_result_to_middle_json(
                    model_list,
                    images_list,
                    pdf_doc,
                    image_writer,
                    batch_model,
                    _lang,
                    _ocr_enable,
                    p_formula_enable,
                    pdf_bytes=chunk_pdf_bytes,
                )
                _trace_stage("pipeline_nocache.chunk_middle_done", file=pdf_file_name, chunk_start=chunk_start, pages=len(chunk_middle_json.get("pdf_info", [])))

                if emit_outputs:
                    for page_info in chunk_middle_json.get("pdf_info", []):
                        page_info["page_idx"] = int(page_info.get("page_idx", 0)) + chunk_start

                    if f_dump_md:
                        chunk_md = pipeline_union_make(chunk_middle_json["pdf_info"], f_make_md_mode, image_dir)
                        if chunk_md:
                            chunk_md_text = str(chunk_md)
                            if md_has_content:
                                trailing_nl = _count_trailing_newlines(md_tail)
                                leading_nl = len(chunk_md_text) - len(chunk_md_text.lstrip("\n"))
                                missing_nl = max(0, 2 - (trailing_nl + leading_nl))
                                if missing_nl > 0:
                                    chunk_md_text = ("\n" * missing_nl) + chunk_md_text
                            _append_text_file(md_output_path, chunk_md_text)
                            md_has_content = True
                            md_tail = (md_tail + chunk_md_text)[-32:]
                            md_outputs += str(chunk_md)

                    if f_dump_middle_json:
                        if middle_fp is None:
                            middle_fp = open(middle_json_path, "w", encoding="utf-8")
                        if not middle_array_started:
                            middle_meta = {k: v for k, v in chunk_middle_json.items() if k != "pdf_info"}
                            _stream_json_array_start(middle_fp, middle_meta, "pdf_info")
                            middle_array_started = True
                        for page_info in chunk_middle_json.get("pdf_info", []):
                            if not middle_first_page:
                                middle_fp.write(",")
                            middle_fp.write(json.dumps(page_info, ensure_ascii=False))
                            middle_first_page = False

                if f_dump_model_output:
                    if model_fp is None:
                        model_fp = open(model_json_path, "w", encoding="utf-8")
                    if not model_array_started:
                        model_fp.write("[")
                        model_array_started = True
                    for model_item in model_list:
                        if not model_first_item:
                            model_fp.write(",")
                        model_fp.write(json.dumps(model_item, ensure_ascii=False))
                        model_first_item = False

                chunk_middle_json = None

                _release_image_list(images_list)
                images_list = None
                if pdf_doc is not None:
                    try:
                        pdf_doc.close()
                    except Exception:
                        pass
                    pdf_doc = None
                model_list = None
                gc.collect()
                _trace_stage("pipeline_nocache.chunk_cleanup_done", file=pdf_file_name, chunk_start=chunk_start)

            if emit_outputs:
                if f_dump_middle_json:
                    if middle_fp is None:
                        middle_fp = open(middle_json_path, "w", encoding="utf-8")
                    if not middle_array_started:
                        _stream_json_array_start(middle_fp, {}, "pdf_info")
                        middle_array_started = True
                    middle_fp.write("]}")
                    middle_fp.close()
                    middle_fp = None
                    try:
                        with open(middle_json_path, "r", encoding="utf-8") as f:
                            middle_json_obj = json.load(f)
                        with open(middle_json_path, "w", encoding="utf-8") as f:
                            f.write(json.dumps(middle_json_obj, ensure_ascii=False, indent=4))
                    except Exception:
                        pass

                if f_dump_model_output:
                    if model_fp is None:
                        model_fp = open(model_json_path, "w", encoding="utf-8")
                    if not model_array_started:
                        model_fp.write("[")
                        model_array_started = True
                    model_fp.write("]")
                    model_fp.close()
                    model_fp = None

                middle_json_loaded = None
                if any([f_draw_layout_bbox, f_draw_span_bbox, f_draw_line_sort_bbox, f_dump_content_list]):
                    if f_dump_middle_json and os.path.exists(middle_json_path):
                        with open(middle_json_path, "r", encoding="utf-8") as f:
                            middle_json_loaded = json.load(f)
                    else:
                        middle_json_loaded = {"pdf_info": []}
                    pdf_info = middle_json_loaded.get("pdf_info", [])
                    if f_draw_layout_bbox:
                        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{output_file_name}_layout.pdf")
                    if f_draw_span_bbox:
                        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{output_file_name}_span.pdf")
                    if f_draw_line_sort_bbox:
                        draw_line_sort_bbox(pdf_info, pdf_bytes, local_md_dir, f"{output_file_name}_line_sort.pdf")
                    if f_dump_content_list:
                        content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                        md_writer.write_string(
                            f"{output_file_name}_content_list.json",
                            json.dumps(content_list, ensure_ascii=False, indent=4),
                        )

                if f_dump_model_output and os.path.exists(model_json_path):
                    with open(model_json_path, "r", encoding="utf-8") as f:
                        json_outputs += f.read()
                elif f_dump_middle_json and os.path.exists(middle_json_path):
                    with open(middle_json_path, "r", encoding="utf-8") as f:
                        json_outputs += f.read()

                if f_dump_md and os.path.exists(md_output_path):
                    _split_images_by_md_references(local_md_dir, output_file_name)

                output_metas.append(
                    {
                        "input_name": pdf_file_name,
                        "output_dir": local_md_dir,
                        "output_file_name": output_file_name,
                        "md_path": md_output_path,
                        "images_md_dir": os.path.join(local_md_dir, "images_md"),
                        "middle_json_path": middle_json_path,
                        "model_json_path": model_json_path,
                    }
                )

                _trace_stage("pipeline_nocache.doc_output_done", file=pdf_file_name)
            else:
                _trace_stage("pipeline_nocache.doc_output_skipped", file=pdf_file_name)
        finally:
            if middle_fp is not None:
                try:
                    middle_fp.close()
                except Exception:
                    pass
            if model_fp is not None:
                try:
                    model_fp.close()
                except Exception:
                    pass
            _release_image_list(images_list)
            if pdf_doc is not None:
                try:
                    pdf_doc.close()
                except Exception:
                    pass
            gc.collect()
            _trace_stage("pipeline_nocache.doc_cleanup_done", file=pdf_file_name)
    _trace_stage("pipeline_nocache.done")
    return md_outputs, json_outputs, output_metas

def _process_pipeline(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        p_lang_list,
        parse_method,
        p_formula_enable,
        p_table_enable,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_make_md_mode,
        f_draw_line_sort_bbox,
        BatchAnalyze,
        start_page_id=0,
        end_page_id=None,
        tqdm_enable = False
):
    if BatchAnalyze.enable_cache:
        return _process_pipeline_cache(output_dir, pdf_file_names, pdf_bytes_list, p_lang_list, parse_method,
                                p_formula_enable, p_table_enable, f_draw_layout_bbox, f_draw_span_bbox,
                                f_dump_md, f_dump_middle_json, f_dump_model_output, f_dump_orig_pdf,
                                f_dump_content_list, f_make_md_mode, f_draw_line_sort_bbox, BatchAnalyze,
                                start_page_id, end_page_id, tqdm_enable)
    else :
        return _process_pipeline_nocache(output_dir, pdf_file_names, pdf_bytes_list, p_lang_list, parse_method,
                                p_formula_enable, p_table_enable, f_draw_layout_bbox, f_draw_span_bbox,
                                f_dump_md, f_dump_middle_json, f_dump_model_output, f_dump_orig_pdf,
                                f_dump_content_list, f_make_md_mode, f_draw_line_sort_bbox, BatchAnalyze,
                                start_page_id, end_page_id, tqdm_enable)

# async def _async_process_vlm(
#         output_dir,
#         pdf_file_names,
#         pdf_bytes_list,
#         backend,
#         f_draw_layout_bbox,
#         f_draw_span_bbox,
#         f_dump_md,
#         f_dump_middle_json,
#         f_dump_model_output,
#         f_dump_orig_pdf,
#         f_dump_content_list,
#         f_make_md_mode,
#         server_url=None,
#         **kwargs,
# ):
#     """Asynchronous processing of VLM backend logic"""
#     parse_method = "vlm"
#     f_draw_span_bbox = False
#     if not backend.endswith("client"):
#         server_url = None

#     for idx, pdf_bytes in enumerate(pdf_bytes_list):
#         pdf_file_name = pdf_file_names[idx]
#         local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
#         image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

#         middle_json, infer_result = await aio_vlm_doc_analyze(
#             pdf_bytes, image_writer=image_writer, backend=backend, server_url=server_url, **kwargs,
#         )

#         pdf_info = middle_json["pdf_info"]

#         _process_output(
#             pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
#             md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
#             f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
#             f_make_md_mode, middle_json, infer_result, is_pipeline=False
#         )

# def _process_vlm(
#         output_dir,
#         pdf_file_names,
#         pdf_bytes_list,
#         backend,
#         f_draw_layout_bbox,
#         f_draw_span_bbox,
#         f_dump_md,
#         f_dump_middle_json,
#         f_dump_model_output,
#         f_dump_orig_pdf,
#         f_dump_content_list,
#         f_make_md_mode,
#         server_url=None,
#         **kwargs,
# ):
#     """Synchronous processing of VLM backend logic"""
#     parse_method = "vlm"
#     f_draw_span_bbox = False
#     if not backend.endswith("client"):
#         server_url = None

#     for idx, pdf_bytes in enumerate(pdf_bytes_list):
#         pdf_file_name = pdf_file_names[idx]
#         local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
#         image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

#         middle_json, infer_result = vlm_doc_analyze(
#             pdf_bytes, image_writer=image_writer, backend=backend, server_url=server_url, **kwargs,
#         )

#         pdf_info = middle_json["pdf_info"]

#         _process_output(
#             pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
#             md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
#             f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
#             f_make_md_mode, middle_json, infer_result, is_pipeline=False
#         )

# def _process_hybrid(
#         output_dir,
#         pdf_file_names,
#         pdf_bytes_list,
#         h_lang_list,
#         parse_method,
#         inline_formula_enable,
#         backend,
#         f_draw_layout_bbox,
#         f_draw_span_bbox,
#         f_dump_md,
#         f_dump_middle_json,
#         f_dump_model_output,
#         f_dump_orig_pdf,
#         f_dump_content_list,
#         f_make_md_mode,
#         server_url=None,
#         **kwargs,
# ):
#     from mineru.backend.hybrid.hybrid_analyze import doc_analyze as hybrid_doc_analyze
#     """Synchronous processing of hybrid backend logic"""
#     if not backend.endswith("client"):
#         server_url = None

#     for idx, (pdf_bytes, lang) in enumerate(zip(pdf_bytes_list, h_lang_list)):
#         pdf_file_name = pdf_file_names[idx]
#         local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, f"hybrid_{parse_method}")
#         image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

#         middle_json, infer_result, _vlm_ocr_enable = hybrid_doc_analyze(
#             pdf_bytes,
#             image_writer=image_writer,
#             backend=backend,
#             parse_method=parse_method,
#             language=lang,
#             inline_formula_enable=inline_formula_enable,
#             server_url=server_url,
#             **kwargs,
#         )

#         pdf_info = middle_json["pdf_info"]

#         # f_draw_span_bbox = not _vlm_ocr_enable
#         f_draw_span_bbox = False

#         _process_output(
#             pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
#             md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
#             f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
#             f_make_md_mode, middle_json, infer_result, is_pipeline=False
#         )

# async def _async_process_hybrid(
#         output_dir,
#         pdf_file_names,
#         pdf_bytes_list,
#         h_lang_list,
#         parse_method,
#         inline_formula_enable,
#         backend,
#         f_draw_layout_bbox,
#         f_draw_span_bbox,
#         f_dump_md,
#         f_dump_middle_json,
#         f_dump_model_output,
#         f_dump_orig_pdf,
#         f_dump_content_list,
#         f_make_md_mode,
#         server_url=None,
#         **kwargs,
# ):
#     from mineru.backend.hybrid.hybrid_analyze import aio_doc_analyze as aio_hybrid_doc_analyze
#     """Asynchronous processing of hybrid backend logic"""
#     if not backend.endswith("client"):
#         server_url = None

#     for idx, (pdf_bytes, lang) in enumerate(zip(pdf_bytes_list, h_lang_list)):
#         pdf_file_name = pdf_file_names[idx]
#         local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, f"hybrid_{parse_method}")
#         image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

#         middle_json, infer_result, _vlm_ocr_enable = await aio_hybrid_doc_analyze(
#             pdf_bytes,
#             image_writer=image_writer,
#             backend=backend,
#             parse_method=parse_method,
#             language=lang,
#             inline_formula_enable=inline_formula_enable,
#             server_url=server_url,
#             **kwargs,
#         )

#         pdf_info = middle_json["pdf_info"]

#         # f_draw_span_bbox = not _vlm_ocr_enable
#         f_draw_span_bbox = False

#         _process_output(
#             pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
#             md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
#             f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
#             f_make_md_mode, middle_json, infer_result, is_pipeline=False
#         )


def do_parse(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        p_lang_list: list[str],
        BatchAnalyze=None,
        backend="pipeline",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        server_url=None,
        f_draw_layout_bbox=True,
        f_draw_span_bbox=True,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_pdf=True,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
        f_draw_line_sort_bbox=True,
        start_page_id=0,
        end_page_id=None,
        return_output_meta=False,
        tqdm_enable = False,
        **kwargs,
):
    # Preprocess PDF byte data
    pdf_bytes_list = _prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id)
    md_outputs, json_outputs, output_metas = _process_pipeline(output_dir, pdf_file_names, pdf_bytes_list, p_lang_list, parse_method,
                          formula_enable, table_enable, f_draw_layout_bbox, f_draw_span_bbox, f_dump_md,
                          f_dump_middle_json, f_dump_model_output, f_dump_orig_pdf, f_dump_content_list,
                          f_make_md_mode, f_draw_line_sort_bbox, BatchAnalyze, start_page_id, end_page_id, tqdm_enable)
    if return_output_meta:
        return md_outputs, json_outputs, output_metas
    return md_outputs, json_outputs

    # if backend == "pipeline":
        # return _process_pipeline(output_dir, pdf_file_names, pdf_bytes_list, p_lang_list, parse_method,
        #                   formula_enable, table_enable, f_draw_layout_bbox, f_draw_span_bbox, f_dump_md,
        #                   f_dump_middle_json, f_dump_model_output, f_dump_orig_pdf, f_dump_content_list,
        #                   f_make_md_mode, f_draw_line_sort_bbox, BatchAnalyze, start_page_id, end_page_id)
    # else:
        # if backend.startswith("vlm-"):
        #     backend = backend[4:]

        #     if backend == "vllm-async-engine":
        #         raise Exception("vlm-vllm-async-engine backend is not supported in sync mode, please use vlm-vllm-engine backend")

        #     if backend == "auto-engine":
        #         backend = get_vlm_engine(inference_engine='auto', is_async=False)

        #     os.environ['MINERU_VLM_FORMULA_ENABLE'] = str(formula_enable)
        #     os.environ['MINERU_VLM_TABLE_ENABLE'] = str(table_enable)

        #     _process_vlm(
        #         output_dir, pdf_file_names, pdf_bytes_list, backend,
        #         f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
        #         f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
        #         server_url, **kwargs,
        #     )
        # elif backend.startswith("hybrid-"):
        #     backend = backend[7:]

        #     if backend == "vllm-async-engine":
        #         raise Exception(
        #             "hybrid-vllm-async-engine backend is not supported in sync mode, please use hybrid-vllm-engine backend")

        #     if backend == "auto-engine":
        #         backend = get_vlm_engine(inference_engine='auto', is_async=False)

        #     os.environ['MINERU_VLM_TABLE_ENABLE'] = str(table_enable)
        #     os.environ['MINERU_VLM_FORMULA_ENABLE'] = "true"

        #     _process_hybrid(
        #         output_dir, pdf_file_names, pdf_bytes_list, p_lang_list, parse_method, formula_enable, backend,
        #         f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
        #         f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
        #         server_url, **kwargs,
        #     )


async def aio_do_parse(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        p_lang_list: list[str],
        BatchAnalyze=None,
        backend="pipeline",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        server_url=None,
        f_draw_layout_bbox=True,
        f_draw_span_bbox=True,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_pdf=True,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
        f_draw_line_sort_bbox=True,
        start_page_id=0,
        end_page_id=None,
        return_output_meta=False,
        **kwargs,
):
    # Preprocess PDF byte data
    pdf_bytes_list = _prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id)
    md_outputs, json_outputs, output_metas = _process_pipeline(output_dir, pdf_file_names, pdf_bytes_list, p_lang_list, parse_method,
                          formula_enable, table_enable, f_draw_layout_bbox, f_draw_span_bbox, f_dump_md,
                          f_dump_middle_json, f_dump_model_output, f_dump_orig_pdf, f_dump_content_list,
                          f_make_md_mode, f_draw_line_sort_bbox, BatchAnalyze, start_page_id, end_page_id)
    if return_output_meta:
        return md_outputs, json_outputs, output_metas
    return md_outputs, json_outputs

    # if backend == "pipeline":
    #     # pipelineThe mode does not currently support asynchronous processing. Use synchronous processing.
    #     _process_pipeline(output_dir, pdf_file_names, pdf_bytes_list, p_lang_list, parse_method,
    #                       formula_enable, table_enable, f_draw_layout_bbox, f_draw_span_bbox, f_dump_md,
    #                       f_dump_middle_json, f_dump_model_output, f_dump_orig_pdf, f_dump_content_list,
    #                       f_make_md_mode, f_draw_line_sort_bbox, BatchAnalyze, start_page_id, end_page_id)
    # else:
    #     if backend.startswith("vlm-"):
    #         backend = backend[4:]

    #         if backend == "vllm-engine":
    #             raise Exception("vlm-vllm-engine backend is not supported in async mode, please use vlm-vllm-async-engine backend")

    #         if backend == "auto-engine":
    #             backend = get_vlm_engine(inference_engine='auto', is_async=True)

    #         os.environ['MINERU_VLM_FORMULA_ENABLE'] = str(formula_enable)
    #         os.environ['MINERU_VLM_TABLE_ENABLE'] = str(table_enable)

    #         await _async_process_vlm(
    #             output_dir, pdf_file_names, pdf_bytes_list, backend,
    #             f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
    #             f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
    #             server_url, **kwargs,
    #         )
    #     elif backend.startswith("hybrid-"):
    #         backend = backend[7:]

    #         if backend == "vllm-engine":
    #             raise Exception("hybrid-vllm-engine backend is not supported in async mode, please use hybrid-vllm-async-engine backend")

    #         if backend == "auto-engine":
    #             backend = get_vlm_engine(inference_engine='auto', is_async=True)

    #         os.environ['MINERU_VLM_TABLE_ENABLE'] = str(table_enable)
    #         os.environ['MINERU_VLM_FORMULA_ENABLE'] = "true"

    #         await _async_process_hybrid(
    #             output_dir, pdf_file_names, pdf_bytes_list, p_lang_list, parse_method, formula_enable, backend,
    #             f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
    #             f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
    #             server_url, **kwargs,
    #         )


if __name__ == "__main__":
    # pdf_path = "../../demo/pdfs/demo3.pdf"
    pdf_path = "C:/Users/zhaoxiaomeng/Downloads/4546d0e2-ba60-40a5-a17e-b68555cec741.pdf"

    try:
       do_parse("./output", [Path(pdf_path).stem], [read_fn(Path(pdf_path))],["ch"],
                end_page_id=10,
                backend='vlm-huggingface'
                # backend = 'pipeline'
                )
    except Exception as e:
        logger.exception(e)
