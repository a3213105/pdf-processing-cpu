import time
import cv2
 
from tqdm import tqdm
import torch
from magic_pdf.config.constants import MODEL_NAME
from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
from magic_pdf.model.sub_modules.model_utils import (
    clean_vram,crop_img,get_res_list_from_layout_res)
from magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.ocr_utils import (
    get_adjusted_mfdetrec_res,get_ocr_result_list)

YOLO_LAYOUT_BASE_BATCH_SIZE = 1
MFD_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16


class BatchAnalyze:
    def __init__(self,model_manager,enable_ov: bool,Layout_infer_type: str,
        MFD_infer_type: str,MFR_enc_infer_type: str,MFR_dec_infer_type: str,
        OCR_det_infer_type: str,OCR_rec_infer_type: str,Table_infer_type: str,
        Lang_infer_type: str,Page_infer_type: str,nstreams: int,batch_ratio: int,
        show_log,layout_model,formula_enable,table_enable):
        self.model_manager = model_manager
        self.batch_ratio = batch_ratio
        self.show_log = show_log
        self.layout_model = layout_model
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.enable_ov = enable_ov
        self.Layout_infer_type = Layout_infer_type
        self.MFD_infer_type = MFD_infer_type
        self.MFR_enc_infer_type = MFR_enc_infer_type
        self.MFR_dec_infer_type = MFR_dec_infer_type
        self.OCR_det_infer_type = OCR_det_infer_type
        self.OCR_rec_infer_type = OCR_rec_infer_type
        self.Table_infer_type = Table_infer_type
        self.Lang_infer_type = Lang_infer_type
        self.Page_infer_type = Page_infer_type
        self.nstreams = nstreams
        self.model = self.model_manager.get_model(
                enable_ov=self.enable_ov,
                Layout_infer_type=self.Layout_infer_type,
                MFD_infer_type=self.MFD_infer_type,
                MFR_enc_infer_type=self.MFR_enc_infer_type,
                MFR_dec_infer_type=self.MFR_dec_infer_type,
                OCR_det_infer_type=self.OCR_det_infer_type,
                OCR_rec_infer_type=self.OCR_rec_infer_type,
                Table_infer_type=self.Table_infer_type,
                Lang_infer_type=self.Lang_infer_type,
                Page_infer_type=self.Page_infer_type,
                nstreams=self.nstreams,
                ocr=True,
                show_log=self.show_log,
                lang = None,
                layout_model = self.layout_model,
                formula_enable = self.formula_enable,
                table_enable = self.table_enable,
            )

    def __call__(self,images_with_extra_info: list, tqdm_enable = False) -> list:
        if len(images_with_extra_info) == 0:
            return []
        images_layout_res = []
        # layout_start_time = time.time()
        atom_model_manager = AtomModelSingleton()

        images = [image for image,_,_ in images_with_extra_info]

        if self.model.layout_model_name == MODEL_NAME.LAYOUTLMv3:
            # layoutlmv3
            for image in images:
                layout_res = self.model.layout_model(image,ignore_catids=[])
                images_layout_res.append(layout_res)
        elif self.model.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            # doclayout_yolo
            layout_images = []
            for image_index,image in enumerate(images):
                layout_images.append(image)
            images_layout_res += self.model.layout_model.batch_predict(
                layout_images,YOLO_LAYOUT_BASE_BATCH_SIZE
            )

        if self.model.apply_formula:
            # 公式检测
            # mfd_start_time = time.time()
            images_mfd_res = self.model.mfd_model.batch_predict(
                images,MFD_BASE_BATCH_SIZE
            )
            # mfd_stop_time = time.time()
            # logger.info(
            #     f'mfd time: {round(time.time() - mfd_start_time,2)},image num: {len(images)}'
            # )

            # 公式识别
            # mfr_start_time = time.time()
            images_formula_list = self.model.mfr_model.batch_predict(
                images_mfd_res,
                images,
                batch_size=self.batch_ratio * MFR_BASE_BATCH_SIZE,
            )
            # mfr_stop_time = time.time()

            # mfr_count = 0
            for image_index in range(len(images)):
                images_layout_res[image_index] += images_formula_list[image_index]
                # mfr_count += len(images_formula_list[image_index])
            # logger.info(
            #     f'mfr time: {round(time.time() - mfr_start_time,2)},image num: {mfr_count}'
            # )

        # 清理显存
        ocr_res_list_all_page = []
        table_res_list_all_page = []
        for index in range(len(images)):
            _,ocr_enable,_lang = images_with_extra_info[index]
            layout_res = images_layout_res[index]
            np_array_img = images[index]

            ocr_res_list,table_res_list,single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res)
            )

            ocr_res_list_all_page.append({'ocr_res_list':ocr_res_list,
                                          'lang':_lang,
                                          'ocr_enable':ocr_enable,
                                          'np_array_img':np_array_img,
                                          'single_page_mfdetrec_res':single_page_mfdetrec_res,
                                          'layout_res':layout_res,
                                          })

            for table_res in table_res_list:
                table_img,_ = crop_img(table_res,np_array_img)
                table_res_list_all_page.append({'table_res':table_res,
                                                'lang':_lang,
                                                'table_img':table_img,
                                              })

        # 文本框检测
        # det_start = time.time()
        if self.enable_ov:
            desc_str = f"OCR-Det_OV_{self.OCR_det_infer_type} Predict"
        else:
            desc_str = f"OCR-Det_{self.OCR_det_infer_type} Predict"
        for ocr_res_list_dict in tqdm(ocr_res_list_all_page,desc=desc_str, disable=not tqdm_enable):
            # Process each area that requires OCR processing
            _lang = ocr_res_list_dict['lang']
            # Get OCR results for this language's images
            ocr_model = atom_model_manager.get_atom_model(
                enable_ov = self.enable_ov,
                OCR_det_infer_type = self.OCR_det_infer_type,
                OCR_rec_infer_type = self.OCR_rec_infer_type,
                nstreams = self.nstreams,
                atom_model_name = 'ocr',
                ocr_show_log = False,
                det_db_box_thresh = 0.3,
                lang = _lang
            )
            for res in ocr_res_list_dict['ocr_res_list']:
                new_image,useful_list = crop_img(
                    res,ocr_res_list_dict['np_array_img'],crop_paste_x=50,crop_paste_y=50
                )
                adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                    ocr_res_list_dict['single_page_mfdetrec_res'],useful_list
                )

                # OCR-det
                new_image = cv2.cvtColor(new_image,cv2.COLOR_RGB2BGR)
                ocr_res = ocr_model.ocr(
                    new_image,mfd_res=adjusted_mfdetrec_res,rec=False
                )[0]

                # Integration results
                if ocr_res:
                    ocr_result_list = get_ocr_result_list(ocr_res,useful_list,ocr_res_list_dict['ocr_enable'],new_image,_lang)
                    ocr_res_list_dict['layout_res'].extend(ocr_result_list)

            # det_count += len(ocr_res_list_dict['ocr_res_list'])
        # det_stop = time.time()
        # logger.info(f'ocr-det time: {round(det_stop-det_start,2)},image num: {det_count}')

        # 表格识别 table recognition
        if self.model.apply_table:
            # table_start = time.time()
            if self.enable_ov:
                desc_str = f"Table_OV_{self.Table_infer_type} Predict"
            else:
                desc_str = f"Table_{self.Table_infer_type} Predict"
            for table_res_dict in tqdm(table_res_list_all_page,desc=desc_str, disable=not tqdm_enable):
                _lang = table_res_dict['lang']
                atom_model_manager = AtomModelSingleton()
                ocr_engine = atom_model_manager.get_atom_model(
                    enable_ov = self.enable_ov,
                    OCR_det_infer_type = self.OCR_det_infer_type,
                    OCR_rec_infer_type = self.OCR_rec_infer_type,
                    nstreams = self.nstreams,
                    atom_model_name='ocr',
                    ocr_show_log=False,
                    det_db_box_thresh=0.5,
                    det_db_unclip_ratio=1.6,
                    lang=_lang
                )
                table_model = atom_model_manager.get_atom_model(
                    enable_ov = self.enable_ov,
                    OCR_det_infer_type = self.OCR_det_infer_type,
                    OCR_rec_infer_type = self.OCR_rec_infer_type,
                    Table_infer_type = self.Table_infer_type,
                    nstreams = self.nstreams,
                    atom_model_name='table',
                    table_model_name='rapid_table',
                    table_model_path='',
                    table_max_time=400,
                    device='cpu',
                    ocr_engine=ocr_engine,
                    table_sub_model_name='slanet_plus'
                )
                html_code,table_cell_bboxes,logic_points,elapse = table_model.predict(table_res_dict['table_img'])
                # 判断是否返回正常
                if html_code:
                    expected_ending = html_code.strip().endswith(
                        '</html>'
                    ) or html_code.strip().endswith('</table>')
                    if expected_ending:
                        table_res_dict['table_res']['html'] = html_code
                #     else:
                #         logger.warning(
                #             'table recognition processing fails,not found expected HTML table end'
                #         )
                # else:
                #     logger.warning(
                #         'table recognition processing fails,not get html return'
                #     )
            # table_stop = time.time()
            # logger.info(f'table time: {round(table_stop - table_start,2)},image num: {len(table_res_list_all_page)}')
                  
        # Create dictionaries to store items by language
        need_ocr_lists_by_lang = {}  # Dict of lists for each language
        img_crop_lists_by_lang = {}  # Dict of lists for each language

        for layout_res in images_layout_res:
            for layout_res_item in layout_res:
                if layout_res_item['category_id'] in [15]:
                    if 'np_img' in layout_res_item and 'lang' in layout_res_item:
                        lang = layout_res_item['lang']

                        # Initialize lists for this language if not exist
                        if lang not in need_ocr_lists_by_lang:
                            need_ocr_lists_by_lang[lang] = []
                            img_crop_lists_by_lang[lang] = []

                        # Add to the appropriate language-specific lists
                        need_ocr_lists_by_lang[lang].append(layout_res_item)
                        img_crop_lists_by_lang[lang].append(layout_res_item['np_img'])

                        # Remove the fields after adding to lists
                        layout_res_item.pop('np_img')
                        layout_res_item.pop('lang')

        if len(img_crop_lists_by_lang) > 0:
            # Process OCR by language
            # rec_start = time.time()
            # Process each language separately
            for lang,img_crop_list in img_crop_lists_by_lang.items():
                if len(img_crop_list) > 0:
                    # Get OCR results for this language's images
                    atom_model_manager = AtomModelSingleton()
                    ocr_model = atom_model_manager.get_atom_model(
                        enable_ov = self.enable_ov,
                        OCR_det_infer_type = self.OCR_det_infer_type,
                        OCR_rec_infer_type = self.OCR_rec_infer_type,
                        nstreams = self.nstreams,
                        atom_model_name='ocr',
                        ocr_show_log=False,
                        det_db_box_thresh=0.3,
                        lang=lang
                    )
                    ocr_res_list = ocr_model.ocr(img_crop_list,det=False, tqdm_enable=False)[0]
                    # Verify we have matching counts
                    assert len(ocr_res_list) == len(
                        need_ocr_lists_by_lang[lang]),f'ocr_res_list: {len(ocr_res_list)},need_ocr_list: {len(need_ocr_lists_by_lang[lang])} for lang: {lang}'

                    # Process OCR results for this language
                    for index,layout_res_item in enumerate(need_ocr_lists_by_lang[lang]):
                        ocr_text,ocr_score = ocr_res_list[index]
                        layout_res_item['text'] = ocr_text
                        layout_res_item['score'] = float(f"{ocr_score:.3f}")

            # rec_stop = time.time()
            # logger.info(f'ocr-rec time: {round(rec_stop - rec_start,2)},total images processed: {total_processed}')
        return images_layout_res
