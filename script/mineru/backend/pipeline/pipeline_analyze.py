import os
import time
from typing import List, Tuple
from PIL import Image
from loguru import logger
import pypdfium2 as pdfium

from .model_init import MineruPipelineModel
from mineru.utils.config_reader import get_device
from mineru.utils.enum_class import ImageType
from mineru.utils.pdf_classify import classify
from mineru.utils.pdf_image_tools import load_images_from_pdf, load_image_from_pdf
from mineru.utils.model_utils import get_vram, clean_memory
from mineru.utils.pdf_page_id import get_end_page_id
import gc

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Let mps fallback
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # Disable albumations from checking for updates

class ModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    def clear_cache(self):
        keys_to_delete = []
        for k in list(self._models.keys()):
            keys_to_delete.append(k)

        for k in keys_to_delete:
            model = self._models.pop(k, None)
            if model is not None:
                del model
        gc.collect()


    def get_model(self, enable_cache, lang=None, formula_enable=None, table_enable=None, **kwargs):
        key = (lang, formula_enable, table_enable)
        if key in self._models:
            return self._models[key]
        if not enable_cache:
            self.clear_cache()
        self._models[key] = custom_model_init(lang=lang, formula_enable=formula_enable,
                table_enable=table_enable, enable_cache=enable_cache, **kwargs)
        return self._models[key]


def custom_model_init(
    lang=None,
    formula_enable=True,
    table_enable=True,
    **kwargs
):
    model_init_start = time.time()
    # Read model-dir and device from configuration file
    # device = get_device()
    device = 'cpu'
    formula_config = {"enable": formula_enable}
    table_config = {"enable": table_enable}

    model_input = {
        'device': device,
        'table_config': table_config,
        'formula_config': formula_config,
        'lang': lang,
    }
    model_input.update(kwargs)

    custom_model = MineruPipelineModel(**model_input)

    model_init_cost = time.time() - model_init_start
    logger.info(f'model init cost: {model_init_cost}')

    return custom_model


def doc_analyze(batch_model, pdf_bytes_list, lang_list, parse_method: str = 'auto', tqdm_enable=False):
    """
    Appropriately increasing MIN_BATCH_INFERENCE_SIZE can improve performance. Larger MIN_BATCH_INFERENCE_SIZEWill consume more memory,
    It can be set through the environment variable MINERU_MIN_BATCH_INFERENCE_SIZE. The default value is 384.
    """
    min_batch_inference_size = int(os.environ.get('MINERU_MIN_BATCH_INFERENCE_SIZE', 384))

    all_image_lists = []
    all_pdf_docs = []
    ocr_enabled_list = []
    infer_results = [[] for _ in range(len(pdf_bytes_list))]

    load_images_start = time.time()
    for pdf_idx, pdf_bytes in enumerate(pdf_bytes_list):
        # Determine OCR settings
        _ocr_enable = False
        if parse_method == 'auto':
            if classify(pdf_bytes) == 'ocr':
                _ocr_enable = True
        elif parse_method == 'ocr':
            _ocr_enable = True

        ocr_enabled_list.append(_ocr_enable)
        _lang = lang_list[pdf_idx]

        # Collect pages in each dataset and infer in batches
        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL,)
        all_image_lists.append(images_list)
        all_pdf_docs.append(pdf_doc)

        current_batch = []
        current_page_indices = []
        for page_idx, img_dict in enumerate(images_list):
            current_batch.append((img_dict['img_pil'], _ocr_enable, _lang))
            current_page_indices.append(page_idx)

            if len(current_batch) >= min_batch_inference_size:
                batch_results = batch_image_analyze(batch_model, current_batch, tqdm_enable=tqdm_enable)
                for local_idx, result in zip(current_page_indices, batch_results):
                    pil_img = images_list[local_idx]['img_pil']
                    page_info_dict = {'page_no': local_idx, 'width': pil_img.width, 'height': pil_img.height}
                    infer_results[pdf_idx].append({'layout_dets': result, 'page_info': page_info_dict})
                current_batch.clear()
                current_page_indices.clear()

        if current_batch:
            batch_results = batch_image_analyze(batch_model, current_batch, tqdm_enable=tqdm_enable)
            for local_idx, result in zip(current_page_indices, batch_results):
                pil_img = images_list[local_idx]['img_pil']
                page_info_dict = {'page_no': local_idx, 'width': pil_img.width, 'height': pil_img.height}
                infer_results[pdf_idx].append({'layout_dets': result, 'page_info': page_info_dict})

            current_batch.clear()
            current_page_indices.clear()
    total_pages = sum(len(images_list) for images_list in all_image_lists)
    load_images_time = round(time.time() - load_images_start, 2)
    if load_images_time > 0:
        logger.info(f"load images cost: {load_images_time}, speed: {round(total_pages / load_images_time, 3)} images/s")
    else:
        logger.info(f"load images cost: {load_images_time}, total pages: {total_pages}")

    return infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list


def doc_analyze_1by1(batch_model, pdf_bytes, lang, parse_method: str = 'auto',
                     start_page_id=0, end_page_id=None, page_index_offset=0, tqdm_enable=False):
    # Determine OCR settings
    _ocr_enable = False

    if parse_method == 'auto':
        if classify(pdf_bytes) == 'ocr':
            _ocr_enable = True
    elif parse_method == 'ocr':
        _ocr_enable = True

    _lang = lang

    pdf_doc = pdfium.PdfDocument(pdf_bytes)

    end_page_id = get_end_page_id(end_page_id, len(pdf_doc))
    page_image_info_all = []
    results = []
    for page_idx in range(start_page_id, end_page_id+1):
        page_image_info = load_image_from_pdf(
            pdf_bytes,
            pdf_doc,
            image_type=ImageType.PIL,
            start_page_id=page_idx,
            end_page_id=page_idx,
            log_start_page_id=page_idx + page_index_offset,
            log_end_page_id=page_idx + page_index_offset,
        )[0]
        page_image = page_image_info['img_pil']
        batch_image = [(page_image, _ocr_enable, _lang)]

        # Execute batch processing
        batch_results = batch_image_analyze(batch_model, batch_image, tqdm_enable=tqdm_enable)
        # Build return results
        page_info_dict = {'page_no': page_idx, 'width': page_image.width, 'height': page_image.height}
        page_dict = {'layout_dets': batch_results[0], 'page_info': page_info_dict}
        results.append(page_dict)
        page_image_info_all.append(page_image_info)
    return results, page_image_info_all, pdf_doc, _lang, _ocr_enable


def get_batch_info():
    device = get_device()

    if str(device).startswith('npu'):
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                torch_npu.npu.set_compile_mode(jit_compile=False)
        except Exception as e:
            raise RuntimeError(
                "NPU is selected as device, but torch_npu is not available. "
                "Please ensure that the torch_npu package is installed correctly."
            ) from e

    gpu_memory = get_vram(device)
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
    # logger.info(
    #         f'GPU Memory: {gpu_memory} GB, Batch Ratio: {batch_ratio}. '
    # )

    # Detect the version number of torch
    import torch
    from packaging import version
    device_type = os.getenv("MINERU_LMDEPLOY_DEVICE", "")
    force_disable_ocr_det_batch = os.getenv("MINERU_FORCE_DISABLE_OCR_DET_BATCH", "0") == "1"
    if force_disable_ocr_det_batch:
        enable_ocr_det_batch = False
    elif (
            version.parse(torch.__version__) >= version.parse("2.8.0")
            or str(device).startswith('mps')
            or device_type.lower() in ["corex"]
    ):
        enable_ocr_det_batch = False
    else:
        enable_ocr_det_batch = True
    return batch_ratio, enable_ocr_det_batch


def batch_image_analyze(batch_model, images_with_extra_info: List[Tuple[Image.Image, bool, str]], tqdm_enable):
    results = batch_model(images_with_extra_info, tqdm_enable=tqdm_enable)
    clean_memory(get_device())
    return results