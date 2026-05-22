import html

import cv2
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import os

from .model_list import AtomicModel
from ...utils.config_reader import get_formula_enable, get_table_enable
from ...utils.model_utils import crop_img, get_res_list_from_layout_res, clean_vram
from ...utils.ocr_utils import merge_det_boxes, update_det_boxes, sorted_boxes
from ...utils.ocr_utils import get_adjusted_mfdetrec_res, get_ocr_result_list, OcrConfidence, get_rotate_crop_image
from ...utils.pdf_image_tools import get_crop_np_img

YOLO_LAYOUT_BASE_BATCH_SIZE = 1
MFD_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 16
TABLE_ORI_CLS_BATCH_SIZE = 16
TABLE_Wired_Wireless_CLS_BATCH_SIZE = 16


class BatchAnalyze:
    def __init__(self, model_manager, enable_cache, enable_ov: bool, Layout_infer_type: str,
        MFD_infer_type: str, MFR_enc_infer_type: str, MFR_dec_infer_type: str, OCR_det_infer_type: str,
        OCR_rec_infer_type: str, wired_table_type: str, WirelessTable_type: str, img_orientation_cls_type: str,
        table_cls_type: str, layoutreader_type: str, nstreams: int, batch_ratio: int, formula_enable: bool,
        table_enable: bool, enable_ocr_det_batch: bool = True):
        self.batch_ratio = batch_ratio
        self.formula_enable = get_formula_enable(formula_enable)
        self.table_enable = get_table_enable(table_enable)
        self.model_manager = model_manager
        self.enable_ocr_det_batch = enable_ocr_det_batch
        self.enable_ov = enable_ov
        self.Layout_infer_type = Layout_infer_type
        self.MFD_infer_type = MFD_infer_type
        self.MFR_enc_infer_type = MFR_enc_infer_type
        self.MFR_dec_infer_type = MFR_dec_infer_type
        self.OCR_det_infer_type = OCR_det_infer_type
        self.OCR_rec_infer_type = OCR_rec_infer_type
        self.wired_table_type = wired_table_type
        self.WirelessTable_type = WirelessTable_type
        self.img_orientation_cls_type = img_orientation_cls_type
        self.table_cls_type = table_cls_type
        self.layoutreader_type = layoutreader_type
        self.nstreams = nstreams
        self.enable_cache = enable_cache

        YOLO_LAYOUT_BASE_BATCH_SIZE = int(os.environ.get('YOLO_LAYOUT_BASE_BATCH_SIZE', 1))
        MFD_BASE_BATCH_SIZE = int(os.environ.get('MFD_BASE_BATCH_SIZE', 1))
        MFR_BASE_BATCH_SIZE = int(os.environ.get('MFR_BASE_BATCH_SIZE', 1))
        OCR_DET_BASE_BATCH_SIZE = int(os.environ.get('OCR_DET_BASE_BATCH_SIZE', 1))
        TABLE_ORI_CLS_BATCH_SIZE = int(os.environ.get('TABLE_ORI_CLS_BATCH_SIZE', 1))
        TABLE_Wired_Wireless_CLS_BATCH_SIZE = int(os.environ.get('TABLE_Wired_Wireless_CLS_BATCH_SIZE', 1))

        self.model = self.model_manager.get_model(enable_cache=self.enable_cache, enable_ov=self.enable_ov,
                                                  Layout_infer_type=self.Layout_infer_type, MFD_infer_type=self.MFD_infer_type,
                                                  MFR_enc_infer_type=self.MFR_enc_infer_type, MFR_dec_infer_type=self.MFR_dec_infer_type,
                                                  OCR_det_infer_type=self.OCR_det_infer_type, OCR_rec_infer_type=self.OCR_rec_infer_type,
                                                  wired_table_type=self.wired_table_type, WirelessTable_type=self.WirelessTable_type,
                                                  img_orientation_cls_type=self.img_orientation_cls_type, table_cls_type=self.table_cls_type,
                                                  layoutreader_type=self.layoutreader_type, nstreams=self.nstreams, ocr=True, lang = None, formula_enable = self.formula_enable,
                                                  table_enable = self.table_enable,)

    def remove_unused_weight(self) :
        self.model.remove_unused_weight()

    def __call__(self, images_with_extra_info: list, tqdm_enable: bool = False) -> list:
        # images_with_extra_info is a list of tuples: (PIL_image, ocr_enable, lang)
        if len(images_with_extra_info) == 0:
            return []

        images_layout_res = []

        # atom_model_manager = AtomModelSingleton()

        pil_images = [image for image, _, _ in images_with_extra_info]

        np_images = [np.asarray(image) for image, _, _ in images_with_extra_info]

        # doclayout_yolo

        images_layout_res += self.model.get_layout_model().batch_predict(pil_images, YOLO_LAYOUT_BASE_BATCH_SIZE, tqdm_enable=tqdm_enable)
        del pil_images

        if self.formula_enable:
            # Formula detection
            images_mfd_res = self.model.get_mfd_model().batch_predict(np_images, MFD_BASE_BATCH_SIZE, tqdm_enable=tqdm_enable)

            # Formula recognition
            images_formula_list = self.model.get_mfr_model().batch_predict(images_mfd_res, np_images,
                batch_size=self.batch_ratio * MFR_BASE_BATCH_SIZE, tqdm_enable=tqdm_enable)
            mfr_count = 0
            for image_index in range(len(np_images)):
                images_layout_res[image_index] += images_formula_list[image_index]
                mfr_count += len(images_formula_list[image_index])
            del images_mfd_res
            del images_formula_list

        # Clean up video memory
        clean_vram(self.model.device, vram_threshold=8)

        ocr_res_list_all_page = []
        table_res_list_all_page = []
        for index in range(len(np_images)):
            _, ocr_enable, _lang = images_with_extra_info[index]
            layout_res = images_layout_res[index]
            np_img = np_images[index]

            ocr_res_list, table_res_list, single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res)
            )

            ocr_res_list_all_page.append({'ocr_res_list':ocr_res_list,
                                          'lang':_lang,
                                          'ocr_enable':ocr_enable,
                                          'np_img':np_img,
                                          'single_page_mfdetrec_res':single_page_mfdetrec_res,
                                          'layout_res':layout_res,
                                          })

            for table_res in table_res_list:
                def get_crop_table_img(scale):
                    crop_xmin, crop_ymin = int(table_res['poly'][0]), int(table_res['poly'][1])
                    crop_xmax, crop_ymax = int(table_res['poly'][4]), int(table_res['poly'][5])
                    bbox = (int(crop_xmin / scale), int(crop_ymin / scale), int(crop_xmax / scale), int(crop_ymax / scale))
                    return get_crop_np_img(bbox, np_img, scale=scale)

                wireless_table_img = get_crop_table_img(scale = 1)
                wired_table_img = get_crop_table_img(scale = 10/3)

                table_res_list_all_page.append({'table_res':table_res,
                                                'lang':_lang,
                                                'table_img':wireless_table_img,
                                                'wired_table_img':wired_table_img,
                                              })

        # Table recognition table recognition
        if self.table_enable:
            table_chunk_size = int(os.environ.get('MINERU_TABLE_CONSUME_CHUNK_SIZE', 16))

            def split_chunks(data, chunk_size):
                if chunk_size <= 0:
                    yield data
                    return
                for start in range(0, len(data), chunk_size):
                    yield data[start:start + chunk_size]

            # Image rotation batch processing
            img_orientation_cls_model = self.model.get_img_ori_model()
            try:
                if self.enable_cache:
                    iter_chunks = [table_res_list_all_page]
                else:
                    iter_chunks = split_chunks(table_res_list_all_page, table_chunk_size)

                for table_chunk in iter_chunks:
                    if self.enable_ocr_det_batch:
                        img_orientation_cls_model.batch_predict(table_chunk, det_batch_size=self.batch_ratio * OCR_DET_BASE_BATCH_SIZE,
                                                                batch_size=TABLE_ORI_CLS_BATCH_SIZE, tqdm_enable=tqdm_enable)
                    else:
                        for table_res in table_chunk:
                            rotate_label = img_orientation_cls_model.predict(table_res['table_img'], tqdm_enable=tqdm_enable)
                            img_orientation_cls_model.img_rotate(table_res, rotate_label)
            except Exception as e:
                logger.warning(
                    f"Image orientation classification failed: {e}, using original image"
                )

            # Table classification
            try:
                if self.enable_cache:
                    iter_chunks = [table_res_list_all_page]
                else:
                    iter_chunks = split_chunks(table_res_list_all_page, table_chunk_size)
                table_cls_model = self.model.get_table_cls_model()
                for table_chunk in iter_chunks:
                    table_cls_model.batch_predict(table_chunk, batch_size=TABLE_Wired_Wireless_CLS_BATCH_SIZE, tqdm_enable=tqdm_enable)
            except Exception as e:
                logger.warning(
                    f"Table classification failed: {e}, using default model"
                )

            # OCR det process, sequential execution
            rec_img_lang_group = defaultdict(list)
            det_ocr_engine = self.model.get_ocr_model(det_db_box_thresh=0.5, det_db_unclip_ratio=1.6, enable_merge_det_boxes=False,)
            tqdm_desc = f"Table-det Predict with OV_{self.OCR_det_infer_type}" if self.enable_ov else "Table-det Predict"
            if self.enable_cache:
                for index, table_res_dict in enumerate(tqdm(table_res_list_all_page, desc=tqdm_desc, disable=not tqdm_enable)):
                    bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
                    ocr_result = det_ocr_engine.ocr(bgr_image, rec=False, tqdm_enable=tqdm_enable)[0]
                    table_lang = table_res_dict["lang"]
                    for dt_box in ocr_result:
                        rec_img_lang_group[table_lang].append(
                            {
                                "cropped_img": get_rotate_crop_image(
                                    bgr_image, np.asarray(dt_box, dtype=np.float32)
                                ),
                                "dt_box": np.asarray(dt_box, dtype=np.float32),
                                "table_id": index,
                            }
                        )

                for _lang, rec_img_list in rec_img_lang_group.items():
                    ocr_engine = self.model.get_ocr_model(det_db_box_thresh=0.5, det_db_unclip_ratio=1.6, enable_merge_det_boxes=False, lang=_lang)
                    cropped_img_list = [item["cropped_img"] for item in rec_img_list]
                    ocr_res_list = ocr_engine.ocr(cropped_img_list, det=False, tqdm_enable=tqdm_enable, tqdm_desc=f"Table-rec-{_lang}")[0]
                    for img_dict, ocr_res in zip(rec_img_list, ocr_res_list):
                        if table_res_list_all_page[img_dict["table_id"]].get("ocr_result"):
                            table_res_list_all_page[img_dict["table_id"]]["ocr_result"].append(
                                [img_dict["dt_box"], html.escape(ocr_res[0]), ocr_res[1]]
                            )
                        else:
                            table_res_list_all_page[img_dict["table_id"]]["ocr_result"] = [
                                [img_dict["dt_box"], html.escape(ocr_res[0]), ocr_res[1]]
                            ]
            else:
                for table_chunk in split_chunks(table_res_list_all_page, table_chunk_size):
                    for table_res_dict in tqdm(table_chunk, desc=tqdm_desc, disable=not tqdm_enable):
                        bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
                        table_lang = table_res_dict["lang"]
                        ocr_result = det_ocr_engine.ocr(bgr_image, rec=False, tqdm_enable=tqdm_enable, tqdm_desc=f"Table-rec-table-{table_lang}")[0]
                        if not ocr_result:
                            continue
                        ocr_engine = self.model.get_ocr_model(det_db_box_thresh=0.5, det_db_unclip_ratio=1.6, enable_merge_det_boxes=False, lang=table_lang)
                        rec_inputs = []
                        dt_boxes_list = []
                        for dt_box in ocr_result:
                            dt_box_np = np.asarray(dt_box, dtype=np.float32)
                            dt_boxes_list.append(dt_box_np)
                            rec_inputs.append(get_rotate_crop_image(bgr_image, dt_box_np))

                        ocr_res_list = ocr_engine.ocr(
                            rec_inputs,
                            det=False,
                            tqdm_enable=tqdm_enable,
                            tqdm_desc=f"Table-rec-{table_lang}",
                        )[0]
                        table_res_dict["ocr_result"] = [
                            [dt_box, html.escape(ocr_res[0]), ocr_res[1]]
                            for dt_box, ocr_res in zip(dt_boxes_list, ocr_res_list)
                        ]

            clean_vram(self.model.device, vram_threshold=8)

            # First use the wireless table model for all tables, then use the wired table model for tables classified as wired
            wireless_table_model = self.model.get_wireless_model()
            if self.enable_cache:
                iter_chunks = [table_res_list_all_page]
            else:
                iter_chunks = split_chunks(table_res_list_all_page, table_chunk_size)

            tdpm_desc = f"Table-wired Predict with OV_{self.wired_table_type}" if self.enable_ov else "Table-wired Predict"
            for table_chunk in iter_chunks:
                wireless_table_model.batch_predict(table_chunk, tqdm_enable=tqdm_enable)

                wired_table_res_list = []
                for table_res_dict in table_chunk:
                    if (
                        (table_res_dict["table_res"]["cls_label"] == AtomicModel.WirelessTable
                         and table_res_dict["table_res"]["cls_score"] < 0.9)
                       or table_res_dict["table_res"]["cls_label"] == AtomicModel.WiredTable
                    ):
                        wired_table_res_list.append(table_res_dict)
                    del table_res_dict["table_res"]["cls_label"]
                    del table_res_dict["table_res"]["cls_score"]

                if wired_table_res_list:
                    for table_res_dict in tqdm(wired_table_res_list, desc=tdpm_desc, disable=not tqdm_enable):
                        if not table_res_dict.get("ocr_result", None):
                            continue

                        wired_table_model = self.model.get_wired_model(lang=table_res_dict["lang"],)
                        table_res_dict["table_res"]["html"] = wired_table_model.predict(
                            table_res_dict["wired_table_img"],
                            table_res_dict["ocr_result"],
                            table_res_dict["table_res"].get("html", None),
                            tqdm_enable=tqdm_enable
                        )

                for table_res_dict in table_chunk:
                    html_code = table_res_dict["table_res"].get("html", "") or ""
                    if "<table>" in html_code and "</table>" in html_code:
                        start_index = html_code.find("<table>")
                        end_index = html_code.rfind("</table>") + len("</table>")
                        table_res_dict["table_res"]["html"] = html_code[start_index:end_index]

                    if not self.enable_cache:
                        table_res_dict.pop("table_img", None)
                        table_res_dict.pop("wired_table_img", None)
                        table_res_dict.pop("ocr_result", None)

        # OCR det
        if self.enable_ocr_det_batch:
            RESOLUTION_GROUP_STRIDE = 64
            tqdm_desc = f"OCR-det Predict batch with OV_{self.OCR_det_infer_type}" if self.enable_ov else "OCR-det Predict"
            for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc=tqdm_desc, disable=not tqdm_enable):
                _lang = ocr_res_list_dict['lang']
                ocr_model = self.model.get_ocr_model(det_db_box_thresh=0.3,lang=_lang,)

                resolution_groups = defaultdict(list)
                for res in ocr_res_list_dict['ocr_res_list']:
                    new_image, useful_list = crop_img(
                        res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                    )
                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                    h, w = bgr_image.shape[:2]
                    target_h = ((h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                    target_w = ((w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                    resolution_groups[(target_h, target_w)].append((
                        bgr_image, useful_list, adjusted_mfdetrec_res
                    ))

                for (target_h, target_w), group_crops in resolution_groups.items():
                    batch_images = []
                    for bgr_image, _, _ in group_crops:
                        h, w = bgr_image.shape[:2]
                        padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
                        padded_img[:h, :w] = bgr_image
                        batch_images.append(padded_img)

                    det_batch_size = min(len(batch_images), self.batch_ratio * OCR_DET_BASE_BATCH_SIZE)
                    batch_results = ocr_model.text_detector.batch_predict(batch_images, det_batch_size, tqdm_enable=tqdm_enable)

                    for (bgr_image, useful_list, adjusted_mfdetrec_res), (dt_boxes, _) in zip(group_crops, batch_results):
                        if dt_boxes is None or len(dt_boxes) == 0:
                            continue
                        dt_boxes_sorted = sorted_boxes(dt_boxes)
                        dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []
                        dt_boxes_final = (
                            update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                            if dt_boxes_merged and adjusted_mfdetrec_res
                            else dt_boxes_merged
                        )
                        if not dt_boxes_final:
                            continue
                        ocr_res = [box.tolist() if hasattr(box, 'tolist') else box for box in dt_boxes_final]
                        ocr_result_list = get_ocr_result_list(ocr_res, useful_list, ocr_res_list_dict['ocr_enable'], bgr_image, _lang,)
                        ocr_res_list_dict['layout_res'].extend(ocr_result_list)
                del resolution_groups
        else:
            # Raw single frame processing mode
            tqdm_desc = f"OCR-det-rec Predict with OV_{self.OCR_det_infer_type}" if self.enable_ov else "OCR-det-rec Predict"
            for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc=tqdm_desc, disable=not tqdm_enable):
                # Process each area that requires OCR processing
                _lang = ocr_res_list_dict['lang']
                # Get OCR results for this language's images
                ocr_model = self.model.get_ocr_model(det_db_box_thresh=0.3, lang=_lang,)
                for res in ocr_res_list_dict['ocr_res_list']:
                    new_image, useful_list = crop_img(
                        res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(ocr_res_list_dict['single_page_mfdetrec_res'], useful_list)
                    # OCR-det
                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                    ocr_res = ocr_model.ocr(bgr_image, mfd_res=adjusted_mfdetrec_res, rec=False, tqdm_enable=tqdm_enable)[0]

                    # Integration results
                    if ocr_res:
                        ocr_result_list = get_ocr_result_list(
                            ocr_res, useful_list, ocr_res_list_dict['ocr_enable'],bgr_image, _lang
                        )
                        ocr_res_list_dict['layout_res'].extend(ocr_result_list)

        # OCR rec
        ocr_items_by_lang = defaultdict(list)

        for layout_res in images_layout_res:
            for layout_res_item in layout_res:
                if layout_res_item['category_id'] in [15]:
                    if 'np_img' in layout_res_item and 'lang' in layout_res_item:
                        lang = layout_res_item['lang']
                        ocr_items_by_lang[lang].append((layout_res_item, layout_res_item['np_img']))
                        # Remove the fields after adding to lists
                        layout_res_item.pop('np_img')
                        layout_res_item.pop('lang')

        if len(ocr_items_by_lang) > 0:
            # Process each language separately
            tqdm_desc = f"OCR-rec Predict with OV_{self.OCR_rec_infer_type}" if self.enable_ov else "OCR-rec Predict"
            for lang, item_img_pairs in tqdm(ocr_items_by_lang.items(), desc=tqdm_desc, disable=not tqdm_enable):
            # for lang, item_img_pairs in ocr_items_by_lang.items():
                if len(item_img_pairs) > 0:
                    # Get OCR results for this language's images
                    ocr_model = self.model.get_ocr_model(det_db_box_thresh=0.3,lang=lang,)
                    img_crop_list = [pair[1] for pair in item_img_pairs]
                    ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=tqdm_enable)[0]
                    # Verify we have matching counts
                    assert len(ocr_res_list) == len(
                        item_img_pairs), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(item_img_pairs)} for lang: {lang}'

                    # Process OCR results for this language
                    for index, (layout_res_item, _) in enumerate(item_img_pairs):
                        ocr_text, ocr_score = ocr_res_list[index]
                        layout_res_item['text'] = ocr_text
                        layout_res_item['score'] = float(f"{ocr_score:.3f}")
                        if ocr_score < OcrConfidence.min_confidence:
                            layout_res_item['category_id'] = 16
                        else:
                            layout_res_bbox = [layout_res_item['poly'][0], layout_res_item['poly'][1],
                                               layout_res_item['poly'][4], layout_res_item['poly'][5]]
                            layout_res_width = layout_res_bbox[2] - layout_res_bbox[0]
                            layout_res_height = layout_res_bbox[3] - layout_res_bbox[1]
                            if (
                                    ocr_text in [
                                        '（204Number', '（20', '（2', '（2Number', '（20Number', 'Number', '（204',
                                        '(cid:)', '(ci:)', '(cd:1)', 'cd:)', 'c)', '(cd:)', 'c', 'id:)',
                                        ':)', '√:)', '√i:)', '−i:)', '−:', 'i:)',
                                    ]
                                    and ocr_score < 0.8
                                    and layout_res_width < layout_res_height
                            ):
                                layout_res_item['category_id'] = 16

        return images_layout_res
