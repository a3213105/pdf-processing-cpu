import os
import time

import numpy as np
import torch
import uuid

from magic_pdf.operators.models import InferenceResult
from magic_pdf.model.sub_modules.model_utils import get_vram
from magic_pdf.config.enums import SupportedPdfParseMethod
import magic_pdf.model as model_config
from magic_pdf.data.dataset import Dataset
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.libs.config_reader import (get_device, get_formula_config,
                                          get_layout_config,
                                          get_local_models_dir,
                                          get_table_recog_config)
from magic_pdf.model.model_list import MODEL

class ModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(
        self,
        enable_ov: bool,
        Layout_infer_type: str,
        MFD_infer_type: str,
        MFR_enc_infer_type: str,
        MFR_dec_infer_type: str,
        OCR_det_infer_type: str,
        OCR_rec_infer_type: str,
        Table_infer_type: str,
        Lang_infer_type: str,
        Page_infer_type: str,
        nstreams: int,
        ocr: bool,
        show_log: bool,
        lang=None,
        layout_model=None,
        formula_enable=None,
        table_enable=None,
    ):
        key = (ocr, show_log, lang, layout_model, formula_enable, table_enable)
        if key not in self._models:
            self._models[key] = custom_model_init(
                enable_ov=enable_ov,
                Layout_infer_type=Layout_infer_type,
                MFD_infer_type=MFD_infer_type,
                MFR_enc_infer_type=MFR_enc_infer_type,
                MFR_dec_infer_type=MFR_dec_infer_type,
                OCR_det_infer_type=OCR_det_infer_type,
                OCR_rec_infer_type=OCR_rec_infer_type,
                Table_infer_type=Table_infer_type,
                Lang_infer_type=Lang_infer_type,
                Page_infer_type=Page_infer_type,
                nstreams=nstreams,
                ocr=ocr,
                show_log=show_log,
                lang=lang,
                layout_model=layout_model,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )
        return self._models[key]


def custom_model_init(
    enable_ov: bool,
    Layout_infer_type: str,
    MFD_infer_type: str,
    MFR_enc_infer_type: str,
    MFR_dec_infer_type: str,
    OCR_det_infer_type: str,
    OCR_rec_infer_type: str,
    Table_infer_type: str,
    Lang_infer_type: str,
    Page_infer_type: str,
    nstreams: int,
    ocr: bool = False,
    show_log: bool = False,
    lang=None,
    layout_model=None,
    formula_enable=None,
    table_enable=None,
):
    model = None
    if model_config.__model_mode__ == 'lite':
        # logger.warning(
        #     'The Lite mode is provided for developers to conduct testing only, and the output quality is '
        #     'not guaranteed to be reliable.'
        # )
        model = MODEL.Paddle
    elif model_config.__model_mode__ == 'full':
        model = MODEL.PEK

    if model_config.__use_inside_model__:
        model_init_start = time.time()
        if model == MODEL.Paddle:
            from magic_pdf.model.pp_structure_v2 import CustomPaddleModel

            custom_model = CustomPaddleModel(ocr=ocr, show_log=show_log, lang=lang)
        elif model == MODEL.PEK:
            from magic_pdf.model.pdf_extract_kit import CustomPEKModel

            # 从配置文件读取model-dir和device
            local_models_dir = get_local_models_dir()
            device = get_device()

            layout_config = get_layout_config()
            if layout_model is not None:
                layout_config['model'] = layout_model

            formula_config = get_formula_config()
            if formula_enable is not None:
                formula_config['enable'] = formula_enable

            table_config = get_table_recog_config()
            if table_enable is not None:
                table_config['enable'] = table_enable

            model_input = {
                'enable_ov': enable_ov,
                'Layout_infer_type': Layout_infer_type,
                'MFD_infer_type': MFD_infer_type,
                'MFR_enc_infer_type': MFR_enc_infer_type,
                'MFR_dec_infer_type': MFR_dec_infer_type,
                'OCR_det_infer_type': OCR_det_infer_type,
                'OCR_rec_infer_type': OCR_rec_infer_type,
                'Table_infer_type': Table_infer_type,
                'Lang_infer_type': Lang_infer_type,
                'Page_infer_type': Page_infer_type,
                'nstreams': nstreams,
                'ocr': ocr,
                'show_log': show_log,
                'models_dir': local_models_dir,
                'device': device,
                'table_config': table_config,
                'layout_config': layout_config,
                'formula_config': formula_config,
                'lang': lang,
            }
            custom_model = CustomPEKModel(**model_input)
        else:
            # logger.error('Not allow model_name!')
            print('Not allow model_name!')
            exit(1)
        model_init_cost = time.time() - model_init_start
        # logger.info(f'model init cost: {model_init_cost}')
    else:
        # logger.error('use_inside_model is False, not allow to use inside model')
        print('use_inside_model is False, not allow to use inside model')
        exit(1)

    return custom_model

def doc_analyze(
    dataset: Dataset,
    enable_ov: bool,
    Layout_infer_type: str,
    MFD_infer_type: str,
    MFR_enc_infer_type: str,
    MFR_dec_infer_type: str,
    OCR_det_infer_type: str,
    OCR_rec_infer_type: str,
    Table_infer_type: str,
    Lang_infer_type: str,
    Page_infer_type: str,
    nstreams=8,
    ocr: bool = False,
    show_log: bool = False,
    start_page_id=0,
    end_page_id=None,
    lang=None,
    layout_model=None,
    formula_enable=None,
    table_enable=None,
):
    end_page_id = (
        end_page_id
        if end_page_id is not None and end_page_id >= 0
        else len(dataset) - 1
    )

    MIN_BATCH_INFERENCE_SIZE = int(os.environ.get('MINERU_MIN_BATCH_INFERENCE_SIZE', 200))
    images = []
    page_wh_list = []
    for index in range(len(dataset)):
        if start_page_id <= index <= end_page_id:
            page_data = dataset.get_page(index)
            img_dict = page_data.get_image()
            images.append(img_dict['img'])
            page_wh_list.append((img_dict['width'], img_dict['height']))

    images_with_extra_info = [(images[index], ocr, dataset._lang) for index in range(len(images))]

    if len(images) >= MIN_BATCH_INFERENCE_SIZE:
        batch_size = MIN_BATCH_INFERENCE_SIZE
        batch_images = [images_with_extra_info[i:i+batch_size] for i in range(0, len(images_with_extra_info), batch_size)]
    else:
        batch_images = [images_with_extra_info]

    results = []
    for batch_image in batch_images:
        result = may_batch_image_analyze(batch_image, enable_ov, Layout_infer_type,
                                         MFD_infer_type, MFR_enc_infer_type, MFR_dec_infer_type,
                                         OCR_det_infer_type, OCR_rec_infer_type, Table_infer_type,
                                         Lang_infer_type, Page_infer_type, nstreams,
                                         ocr, show_log, layout_model,
                                         formula_enable, table_enable)
        results.extend(result)

    model_json = []
    for index in range(len(dataset)):
        if start_page_id <= index <= end_page_id:
            result = results.pop(0)
            page_width, page_height = page_wh_list.pop(0)
        else:
            result = []
            page_height = 0
            page_width = 0

        page_info = {'page_no': index, 'width': page_width, 'height': page_height}
        page_dict = {'layout_dets': result, 'page_info': page_info}
        model_json.append(page_dict)

    from magic_pdf.operators.models import InferenceResult
    return InferenceResult(model_json, dataset)

def batch_doc_analyze(
    datasets: list[Dataset],
    parse_method: str = 'auto',
    show_log: bool = False,
    lang=None,
    layout_model=None,
    formula_enable=None,
    table_enable=None,
):
    MIN_BATCH_INFERENCE_SIZE = int(os.environ.get('MINERU_MIN_BATCH_INFERENCE_SIZE', 200))
    batch_size = MIN_BATCH_INFERENCE_SIZE
    page_wh_list = []

    images_with_extra_info = []
    for dataset in datasets:

        ocr = False
        if parse_method == 'auto':
            if dataset.classify() == SupportedPdfParseMethod.TXT:
                ocr = False
            elif dataset.classify() == SupportedPdfParseMethod.OCR:
                ocr = True
        elif parse_method == 'ocr':
            ocr = True
        elif parse_method == 'txt':
            ocr = False

        _lang = dataset._lang

        for index in range(len(dataset)):
            page_data = dataset.get_page(index)
            img_dict = page_data.get_image()
            page_wh_list.append((img_dict['width'], img_dict['height']))
            images_with_extra_info.append((img_dict['img'], ocr, _lang))

    batch_images = [images_with_extra_info[i:i+batch_size] for i in range(0, len(images_with_extra_info), batch_size)]
    results = []
    processed_images_count = 0
    for index, batch_image in enumerate(batch_images):
        processed_images_count += len(batch_image)
        # logger.info(f'Batch {index + 1}/{len(batch_images)}: {processed_images_count} pages/{len(images_with_extra_info)} pages')
        result = may_batch_image_analyze(batch_image, enable_ov, Layout_infer_type,
                                         MFD_infer_type, MFR_enc_infer_type, MFR_dec_infer_type,
                                         OCR_det_infer_type, OCR_rec_infer_type, Table_infer_type,
                                         Lang_infer_type, Page_infer_type, nstreams,
                                         show_log, layout_model, formula_enable, table_enable)
        results.extend(result)

    infer_results = []
    from magic_pdf.operators.models import InferenceResult
    for index in range(len(datasets)):
        dataset = datasets[index]
        model_json = []
        for i in range(len(dataset)):
            result = results.pop(0)
            page_width, page_height = page_wh_list.pop(0)
            page_info = {'page_no': i, 'width': page_width, 'height': page_height}
            page_dict = {'layout_dets': result, 'page_info': page_info}
            model_json.append(page_dict)
        infer_results.append(InferenceResult(model_json, dataset))
    return infer_results


def may_batch_image_analyze(
        images_with_extra_info: list[(np.ndarray, bool, str)],
        enable_ov: bool,
        Layout_infer_type: str,
        MFD_infer_type: str,
        MFR_enc_infer_type: str,
        MFR_dec_infer_type: str,
        OCR_det_infer_type: str,
        OCR_rec_infer_type: str,
        Table_infer_type: str,
        Lang_infer_type: str,
        Page_infer_type: str,
        nstreams: int,
        ocr: bool,
        show_log: bool = False,
        layout_model=None,
        formula_enable=None,
        table_enable=None):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(idx)

    from magic_pdf.model.batch_analyze import BatchAnalyze

    model_manager = ModelSingleton()

    # images = [image for image, _, _ in images_with_extra_info]
    device = get_device()
    batch_ratio = 1

    if str(device).startswith('npu') or str(device).startswith('cuda'):
        vram = get_vram(device)
        if vram is not None:
            gpu_memory = int(os.getenv('VIRTUAL_VRAM_SIZE', round(vram)))
            if gpu_memory >= 16:
                batch_ratio = 16
            elif gpu_memory >= 12:
                batch_ratio = 8
            elif gpu_memory >= 8:
                batch_ratio = 4
            elif gpu_memory >= 6:
                batch_ratio = 2
            else:
                batch_ratio = 1
            # logger.info(f'gpu_memory: {gpu_memory} GB, batch_ratio: {batch_ratio}')
        else:
            # Default batch_ratio when VRAM can't be determined
            batch_ratio = 1
            # logger.info(f'Could not determine GPU memory, using default batch_ratio: {batch_ratio}')


    # doc_analyze_start = time.time()
       
    batch_model = BatchAnalyze(model_manager, enable_ov, Layout_infer_type,
                               MFD_infer_type, MFR_enc_infer_type, MFR_dec_infer_type,
                               OCR_det_infer_type, OCR_rec_infer_type, Table_infer_type,
                               Lang_infer_type, Page_infer_type, nstreams, batch_ratio, show_log,
                               layout_model, formula_enable, table_enable)
    results = batch_model(images_with_extra_info)
    return results

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.batch_analyze import BatchAnalyze

def init_models(
        enable_ov: bool,
        Layout_infer_type: str,
        MFD_infer_type: str,
        MFR_enc_infer_type: str,
        MFR_dec_infer_type: str,
        OCR_det_infer_type: str,
        OCR_rec_infer_type: str,
        Table_infer_type: str,
        Lang_infer_type: str,
        Page_infer_type: str,
        nstreams: int,
        ocr: bool):
    model_manager = ModelSingleton()
    batch_model = BatchAnalyze(model_manager=model_manager, enable_ov=enable_ov,
                               Layout_infer_type=Layout_infer_type, MFD_infer_type=MFD_infer_type,
                               MFR_enc_infer_type=MFR_enc_infer_type, MFR_dec_infer_type=MFR_dec_infer_type,
                               OCR_det_infer_type=OCR_det_infer_type, OCR_rec_infer_type=OCR_rec_infer_type,
                               Table_infer_type=Table_infer_type, Lang_infer_type=Lang_infer_type,
                               Page_infer_type=Page_infer_type, nstreams=nstreams, batch_ratio=1,
                               show_log=False, layout_model = None,formula_enable = None, table_enable = None)
    return batch_model

def may_batch_image_analyze_direct(
        images_with_extra_info: list[(np.ndarray, bool, str)],
        batch_model: BatchAnalyze):
    # doc_analyze_start = time.time()
    results = batch_model(images_with_extra_info)
    # doc_analyze_time = round(time.time() - doc_analyze_start, 2)
    # pages = len(images_with_extra_info)
    return results

def doc_analyze_pages(pdf_model, images_with_extra_info) :
    MIN_BATCH_INFERENCE_SIZE = int(os.environ.get('MINERU_MIN_BATCH_INFERENCE_SIZE', 100))

    if len(images_with_extra_info) >= MIN_BATCH_INFERENCE_SIZE:
        batch_size = MIN_BATCH_INFERENCE_SIZE
        batch_images = [images_with_extra_info[i:i+batch_size] for i in range(0, len(images_with_extra_info), batch_size)]
    else:
        batch_images = [images_with_extra_info]

    results = []
    for batch_image in batch_images:
        result = may_batch_image_analyze_direct(batch_image, pdf_model)
        results.extend(result)
    return results

def doc_analyze_direct_0(
    pdf_bytes: bytes,
    pdf_model: BatchAnalyze,
    return_md = False,
    return_json = False,
    return_layout = False,
    return_span = False,
    output_dir = "./outputs",
    input_name = None,
):
    ocr = False

    if output_dir is None:
        image_writer = None
        md_writer = None
        output_md_filename = None
        image_output_dir = "./images"
        return_span = False
        return_layout = False
    else:
        if input_name is None:
            random_uuid = uuid.uuid4()
            input_name = f"{random_uuid}"
        output_data_dir = f"{output_dir}/{input_name}"
        image_output_dir = f"{output_data_dir}/images"
        image_writer = FileBasedDataWriter(image_output_dir)
        os.makedirs(image_output_dir, exist_ok=True)
        md_writer = FileBasedDataWriter(output_dir)
        output_md_filename = f"{output_data_dir}/{input_name}.md"

    ## Create Dataset Instance
    dataset = PymuDocDataset(pdf_bytes)

    if dataset.classify() == SupportedPdfParseMethod.OCR:
        ocr=True
    else :
        ocr=False

    start_page_id = 0
    end_page_id =  len(dataset) - 1
    MIN_BATCH_INFERENCE_SIZE = int(os.environ.get('MINERU_MIN_BATCH_INFERENCE_SIZE', 100))
    images = []
    page_wh_list = []
    for index in range(len(dataset)):
        if start_page_id <= index <= end_page_id:
            page_data = dataset.get_page(index)
            img_dict = page_data.get_image()
            images.append(img_dict['img'])
            page_wh_list.append((img_dict['width'], img_dict['height']))

    images_with_extra_info = [(images[index], ocr, dataset._lang) for index in range(len(images))]

    if len(images) >= MIN_BATCH_INFERENCE_SIZE:
        batch_size = MIN_BATCH_INFERENCE_SIZE
        batch_images = [images_with_extra_info[i:i+batch_size] for i in range(0, len(images_with_extra_info), batch_size)]
    else:
        batch_images = [images_with_extra_info]

    results = []
    for batch_image in batch_images:
        result = may_batch_image_analyze_direct(batch_image, pdf_model)
        results.extend(result)

    model_json = []
    all_page_info = []
    for index in range(len(dataset)):
        if start_page_id <= index <= end_page_id:
            result = results.pop(0)
            page_width, page_height = page_wh_list.pop(0)
        else:
            result = []
            page_height = 0
            page_width = 0

        page_info = {'page_no': index, 'width': page_width, 'height': page_height}
        page_dict = {'layout_dets': result, 'page_info': page_info}
        all_page_info.append(page_info)
        model_json.append(page_dict)

    from magic_pdf.operators.models import InferenceResult
    infer_result = InferenceResult(model_json, dataset)

    ### get model inference result
    # model_inference_result = infer_result.get_infer_res()

    if ocr :
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else :
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    if md_writer:
        pipe_result.dump_md(md_writer, output_md_filename, f"images",)

    if return_layout :
        pipe_result.draw_layout(os.path.join(output_dir, f"{output_data_dir}/{input_name}_layout.pdf"))

    if return_span :
        pipe_result.draw_span(os.path.join(output_dir, f"{output_data_dir}/{input_name}_spans.pdf"))

    if return_md:
        md_content = pipe_result.get_markdown(output_dir)
    else :
        md_content = None

    ### get content list content
    if return_json :
        content_list_content = pipe_result.get_content_list(output_dir)
    else :
        content_list_content = None

    return md_content, content_list_content, all_page_info, output_md_filename

def doc_analyze_direct_1(
    pdf_bytes: bytes,
    pdf_model: BatchAnalyze,
    return_md = False,
    return_json = False,
    return_layout = False,
    return_span = False,
    output_dir = "./outputs",
    input_name = None,
):
    start_page_id = 0
    ocr = False
    md_content = ""
    content_list_content = ""

    if output_dir is None:
        image_writer = None
        md_writer = None
        output_md_filename = None
        image_output_dir = "./images"
        return_span = False
        return_layout = False
    else:
        if input_name is None:
            random_uuid = uuid.uuid4()
            input_name = f"{random_uuid}"
        output_data_dir = f"{output_dir}/{input_name}"
        image_output_dir = f"{output_data_dir}/images"
        image_writer = FileBasedDataWriter(image_output_dir)
        os.makedirs(image_output_dir, exist_ok=True)
        md_writer = FileBasedDataWriter(output_dir)
        output_md_filename = f"{output_data_dir}/{input_name}.md"

    ## Create Dataset Instance
    dataset = PymuDocDataset(pdf_bytes)

    if dataset.classify() == SupportedPdfParseMethod.OCR:
        ocr=True
    else :
        ocr=False

    end_page_id =  len(dataset) - 1
    end_page_id = 1
    
    step = 1
    total_pages = len(dataset)
    # total_pages = 2
    for start_page_id in range(0, total_pages, step):
        images = []
        page_wh_list = []
        end_page_id = min(start_page_id + step, total_pages)
        for index in range(start_page_id, end_page_id):
            page_data = dataset.get_page(index)
            img_dict = page_data.get_image()
            images.append(img_dict['img'])
            page_wh_list.append((img_dict['width'], img_dict['height']))

        images_with_extra_info = [(images[index], ocr, dataset._lang) for index in range(len(images))]

        results= doc_analyze_pages(pdf_model, images_with_extra_info)

        model_json = []
        # all_page_info = []
        for index in range(end_page_id):
            if index >= start_page_id:
                result = results.pop(0)
                page_width, page_height = page_wh_list.pop(0)
            else :
                result = []
                page_height = 0
                page_width = 0
            page_info = {'page_no': index, 'width': page_width, 'height': page_height}
            page_dict = {'layout_dets': result, 'page_info': page_info}
            # all_page_info.append(page_info)
            model_json.append(page_dict)

        from magic_pdf.operators.models import InferenceResult
        infer_result = InferenceResult(model_json, dataset)

        if ocr :
            pipe_result = infer_result.pipe_ocr_mode(image_writer, start_page_id=start_page_id, end_page_id=end_page_id-1)
        else :
            pipe_result = infer_result.pipe_txt_mode(image_writer, start_page_id=start_page_id, end_page_id=end_page_id-1)

        if md_writer:
            pipe_result.dump_md(md_writer, f"{output_data_dir}/{start_page_id}.md", f"images",)

        if return_layout :
            pipe_result.draw_layout(os.path.join(output_dir, f"{output_data_dir}/{start_page_id}_layout.pdf"))

        if return_span :
            pipe_result.draw_span(os.path.join(output_dir, f"{output_data_dir}/{start_page_id}_spans.pdf"))

        if return_md:
            md_content += pipe_result.get_markdown(output_dir)

        ### get content list content
        if return_json :
            content_list_content += pipe_result.get_content_list(output_dir)

    return md_content, content_list_content, [], output_md_filename

def merge_md(total_pages, step, input_dir: str,output_file: str,):
    with open(output_file, "w", encoding="utf-8") as out:
        for start_page_id in range(0, total_pages, step):
            md_file = f"{input_dir}/{start_page_id}.md"
            with open(md_file, "r", encoding="utf-8") as md:
                out.write(md.read())
            out.write("\n")

def doc_analyze_direct_2(
    pdf_bytes: bytes,
    pdf_model: BatchAnalyze,
    return_md = False,
    return_json = False,
    return_layout = False,
    return_span = False,
    output_dir = "./outputs",
    input_name = None,
):
    ocr = False
    md_content = ""
    content_list_content = ""
    if output_dir is None:
        image_writer = None
        md_writer = None
        output_md_filename = None
        image_output_dir = "./images"
        return_span = False
        return_layout = False
    else:
        if input_name is None:
            random_uuid = uuid.uuid4()
            input_name = f"{random_uuid}"
        output_data_dir = f"{output_dir}/{input_name}"
        image_output_dir = f"{output_data_dir}/images"
        image_writer = FileBasedDataWriter(image_output_dir)
        os.makedirs(image_output_dir, exist_ok=True)
        md_writer = FileBasedDataWriter(output_dir)
        output_md_filename = f"{output_data_dir}/{input_name}.md"

    ## Create Dataset Instance
    dataset = PymuDocDataset(pdf_bytes)

    if dataset.classify() == SupportedPdfParseMethod.OCR:
        ocr=True
    else :
        ocr=False
    
    step = 1
    total_pages = len(dataset)
    # total_pages = 2
    for start_page_id in range(0, total_pages, step):
        images_with_extra_info = []
        page_info_list =[]
        end_page_id = min(start_page_id + step, total_pages)
        for index in range(start_page_id, end_page_id):
            page_data = dataset.get_page(index)
            img_dict = page_data.get_image()
            images_with_extra_info.append((img_dict['img'], ocr, dataset._lang))
            page_info_list.append({'page_no': index, 'width': img_dict['width'], 'height': img_dict['height']})

        results= doc_analyze_pages(pdf_model, images_with_extra_info)

        model_json = []
        for index in range(total_pages):
            if index >= start_page_id and index < end_page_id:
                result = results.pop(0)
                page_info = page_info_list.pop(0)
            else :
                result = []
                page_info = {'page_no': index, 'width': 0, 'height': 0}
            page_dict = {'layout_dets': result, 'page_info': page_info}
            model_json.append(page_dict)

        infer_result = InferenceResult(model_json, dataset)

        if ocr :
            pipe_result = infer_result.pipe_ocr_mode(image_writer, start_page_id=start_page_id, end_page_id=end_page_id-1)
        else :
            pipe_result = infer_result.pipe_txt_mode(image_writer, start_page_id=start_page_id, end_page_id=end_page_id-1)

        if md_writer:
            pipe_result.dump_md(md_writer, f"{output_data_dir}/{start_page_id}.md", f"images",)

        if return_layout :
            pipe_result.draw_layout(os.path.join(output_dir, f"{output_data_dir}/{input_name}_{start_page_id}_layout.pdf"))

        if return_span :
            pipe_result.draw_span(os.path.join(output_dir, f"{output_data_dir}/{input_name}_{start_page_id}_spans.pdf"))

        if return_md:
            md_content += pipe_result.get_markdown(output_dir)

        ### get content list content
        if return_json :
            content_list_content += pipe_result.get_content_list(output_dir)

    print(f"Total pages: {total_pages}, step: {step}, output_data_dir: {output_data_dir}, output_md_filename: {output_md_filename}")
    if output_md_filename is not None:
        merge_md(total_pages, step, output_data_dir, output_md_filename)
    return md_content, content_list_content, None, output_md_filename

def doc_analyze_direct(
    pdf_bytes: bytes,
    pdf_model: BatchAnalyze,
    return_md = False,
    return_json = False,
    return_layout = False,
    return_span = False,
    output_dir = "./outputs",
    input_name = None,
):
    return doc_analyze_direct_0(
        pdf_bytes=pdf_bytes,
        pdf_model=pdf_model,
        return_md=return_md,
        return_json=return_json,
        return_layout=return_layout,
        return_span=return_span,
        output_dir=output_dir,
        input_name=input_name,
    )