import os

import torch
from loguru import logger

from .model_list import AtomicModel
from ...model.layout.doclayoutyolo import DocLayoutYOLOModel
from ...model.mfd.yolo_v8 import YOLOv8MFDModel
from ...model.mfr.unimernet.Unimernet import UnimernetModel
from ...model.mfr.pp_formulanet_plus_m.predict_formula import FormulaRecognizer
from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR, convert_lang
from ...model.ori_cls.paddle_ori_cls import PaddleOrientationClsModel
from ...model.table.cls.paddle_table_cls import PaddleTableClsModel
# from ...model.table.rec.RapidTable import RapidTableModel
from ...model.table.rec.slanet_plus.main import RapidTableModel
from ...model.table.rec.unet_table.main import UnetTableModel
# from ...utils.config_reader import get_device
from ...utils.enum_class import ModelPath
from ...utils.models_download_utils import auto_download_and_get_model_root_path
import gc

MFR_MODEL = os.getenv('MINERU_FORMULA_CH_SUPPORT', 'False')
if MFR_MODEL.lower() in ['true', '1', 'yes']:
    MFR_MODEL = "pp_formulanet_plus_m"
elif MFR_MODEL.lower() in ['false', '0', 'no']:
    MFR_MODEL = "unimernet_small"
else:
    logger.warning(f"Invalid MINERU_FORMULA_CH_SUPPORT value: {MFR_MODEL}, set to default 'False'")
    MFR_MODEL = "unimernet_small"


def img_orientation_cls_model_init(enable_cache, enable_ov, img_orientation_cls_type,
                                   OCR_det_infer_type, OCR_rec_infer_type, nstreams):
    atom_model_manager = AtomModelSingleton()
    ocr_engine = atom_model_manager.get_atom_model(
        enable_cache=enable_cache,
        atom_model_name=AtomicModel.OCR,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        lang="ch_lite",
        enable_merge_det_boxes=False,
        enable_ov=enable_ov,
        OCR_det_infer_type=OCR_det_infer_type,
        OCR_rec_infer_type=OCR_rec_infer_type,
        nstreams=nstreams,
    )
    cls_model = PaddleOrientationClsModel(enable_ov, img_orientation_cls_type, ocr_engine)
    return cls_model


def table_cls_model_init(enable_ov, table_cls_type):
    return PaddleTableClsModel(enable_ov, table_cls_type)


def wired_table_model_init(enable_cache, enable_ov, wired_table_type, OCR_det_infer_type, OCR_rec_infer_type, nstreams, lang=None):
    atom_model_manager = AtomModelSingleton()
    ocr_engine = atom_model_manager.get_atom_model(
        enable_cache=enable_cache,
        atom_model_name=AtomicModel.OCR,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        lang=lang,
        enable_merge_det_boxes=False,
        enable_ov=enable_ov,
        OCR_det_infer_type=OCR_det_infer_type,
        OCR_rec_infer_type=OCR_rec_infer_type,
        nstreams=nstreams,
    )
    table_model = UnetTableModel(enable_ov, wired_table_type, ocr_engine)
    return table_model


def wireless_table_model_init(enable_cache, enable_ov, wireless_table_type, OCR_det_infer_type,
                              OCR_rec_infer_type, nstreams, lang=None):
    atom_model_manager = AtomModelSingleton()
    ocr_engine = atom_model_manager.get_atom_model(
        enable_cache=enable_cache,
        atom_model_name=AtomicModel.OCR,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        lang=lang,
        enable_merge_det_boxes=False,
        enable_ov=enable_ov,
        OCR_det_infer_type=OCR_det_infer_type,
        OCR_rec_infer_type=OCR_rec_infer_type,
        nstreams=nstreams,
    )
    table_model = RapidTableModel(enable_ov, wireless_table_type, ocr_engine)
    return table_model


def mfd_model_init(weight, enable_ov, infer_type, device='cpu'):
    if str(device).startswith('npu'):
        device = torch.device(device)
    mfd_model = YOLOv8MFDModel(weight, enable_ov, infer_type, device)
    return mfd_model


def mfr_model_init(weight_dir, enable_ov, enc_infer_type, dec_infer_type, device='cpu'):
    if MFR_MODEL == "unimernet_small":
        mfr_model = UnimernetModel(weight_dir, enable_ov, enc_infer_type, dec_infer_type, device)
    elif MFR_MODEL == "pp_formulanet_plus_m":
        mfr_model = FormulaRecognizer(weight_dir, enable_ov, enc_infer_type, device)
    else:
        logger.error('MFR model name not allow')
        exit(1)
    return mfr_model


def doclayout_yolo_model_init(weight, enable_ov, infer_type, device='cpu'):
    if str(device).startswith('npu'):
        device = torch.device(device)
    model = DocLayoutYOLOModel(weight, enable_ov, infer_type, device)
    return model


def ocr_model_init(enable_ov, OCR_det_infer_type, OCR_rec_infer_type, nstreams,
                   det_db_box_thresh=0.3, lang='ch', det_db_unclip_ratio=1.8,
                   enable_merge_det_boxes=True):
    if lang is not None and lang != '':
        model = PytorchPaddleOCR(enable_ov=enable_ov, OCR_det_infer_type=OCR_det_infer_type, OCR_rec_infer_type=OCR_rec_infer_type, nstreams=nstreams,
                                 det_db_box_thresh=det_db_box_thresh, lang=lang, use_dilation=True, det_db_unclip_ratio=det_db_unclip_ratio, enable_merge_det_boxes=enable_merge_det_boxes,)
    else:
        model = PytorchPaddleOCR(enable_ov=enable_ov, OCR_det_infer_type=OCR_det_infer_type, OCR_rec_infer_type=OCR_rec_infer_type, nstreams=nstreams,
                                 det_db_box_thresh=det_db_box_thresh, use_dilation=True, det_db_unclip_ratio=det_db_unclip_ratio, enable_merge_det_boxes=enable_merge_det_boxes,)
    return model


def layoutlm_cls_model_init(model_name, enable_ov, layoutlm_infer_type):
    from mineru.utils.block_sort import LayoutReaderModel
    model = LayoutReaderModel(model_name, enable_ov, layoutlm_infer_type)
    return model

class AtomModelSingleton:
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
        
    def get_atom_model(self, enable_cache, atom_model_name: str, **kwargs):       
        lang = kwargs.get('lang', 'ch')
        device = kwargs.get('device', 'cpu')
        lang = convert_lang(device, lang)

        if atom_model_name in [AtomicModel.WiredTable, AtomicModel.WirelessTable]:
            key = (
                atom_model_name,
                lang
            )
        elif atom_model_name in [AtomicModel.OCR]:
            key = (
                atom_model_name,
                kwargs.get('det_db_box_thresh', 0.3),
                lang,
                kwargs.get('det_db_unclip_ratio', 1.8),
                kwargs.get('enable_merge_det_boxes', True),
            )
        else:
            key = atom_model_name
        
        if key in self._models:
            return self._models[key]

        if not enable_cache:
            self.clear_cache()
        self._models[key] = atom_model_init(model_name=atom_model_name, enable_cache=enable_cache, **kwargs)
        return self._models[key]

def atom_model_init(model_name: str, **kwargs):
    atom_model = None
    if model_name == AtomicModel.Layout:
        atom_model = doclayout_yolo_model_init(
            kwargs.get('doclayout_yolo_weights'),
            kwargs.get('enable_ov'),
            kwargs.get('Layout_infer_type'),
            kwargs.get('device')
        )
    elif model_name == AtomicModel.MFD:
        atom_model = mfd_model_init(
            kwargs.get('mfd_weights'),
            kwargs.get('enable_ov'),
            kwargs.get('MFD_infer_type'),
            kwargs.get('device')
        )
    elif model_name == AtomicModel.MFR:
        atom_model = mfr_model_init(
            kwargs.get('mfr_weight_dir'),
            kwargs.get('enable_ov'),
            kwargs.get('MFR_enc_infer_type'),
            kwargs.get('MFR_dec_infer_type'),
            kwargs.get('device')
        )
    elif model_name == AtomicModel.OCR:
        atom_model = ocr_model_init(
            kwargs.get('enable_ov'),
            kwargs.get('OCR_det_infer_type'),
            kwargs.get('OCR_rec_infer_type'),
            kwargs.get('nstreams'),
            kwargs.get('det_db_box_thresh', 0.3),
            kwargs.get('lang'),
            kwargs.get('det_db_unclip_ratio', 1.8),
            kwargs.get('enable_merge_det_boxes', True)
        )
    elif model_name == AtomicModel.WirelessTable:
        atom_model = wireless_table_model_init(
            kwargs.get('enable_cache'),
            kwargs.get('enable_ov'),
            kwargs.get('WirelessTable_type'),
            kwargs.get('OCR_det_infer_type'),
            kwargs.get('OCR_rec_infer_type'),
            kwargs.get('nstreams'),
            kwargs.get('lang'),
        )
    elif model_name == AtomicModel.WiredTable:
        atom_model = wired_table_model_init(
            kwargs.get('enable_cache'),
            kwargs.get('enable_ov'),
            kwargs.get('wired_table_type'),
            kwargs.get('OCR_det_infer_type'),
            kwargs.get('OCR_rec_infer_type'),
            kwargs.get('nstreams'),
            kwargs.get('lang'),
        )
    elif model_name == AtomicModel.TableCls:
        atom_model = table_cls_model_init(
            kwargs.get('enable_ov'),
            kwargs.get('table_cls_type'))
    elif model_name == AtomicModel.ImgOrientationCls:
        atom_model = img_orientation_cls_model_init(
            kwargs.get('enable_cache'),
            kwargs.get('enable_ov'),
            kwargs.get('img_orientation_cls_type'),
            kwargs.get('OCR_det_infer_type'),
            kwargs.get('OCR_rec_infer_type'),
            kwargs.get('nstreams'),
            )
    elif model_name == AtomicModel.LayoutReader:
        atom_model = layoutlm_cls_model_init(
            model_name=model_name,
            enable_ov=kwargs.get('enable_ov'),
            layoutlm_infer_type=kwargs.get('layoutlm_infer_type'),
        )
    else:
        logger.error('model name not allow')
        exit(1)

    if atom_model is None:
        logger.error('model init failed')
        exit(1)
    else:
        return atom_model

class MineruPipelineModel:
    def __init__(self, **kwargs):
        self.formula_config = kwargs.get('formula_config')
        self.apply_formula = self.formula_config.get('enable', True)
        self.table_config = kwargs.get('table_config')
        self.apply_table = self.table_config.get('enable', True)
        # self.lang = kwargs.get('lang', None)
        self.device = kwargs.get('device', 'cpu')
        self.enable_ov = kwargs.get('enable_ov', True)
        self.Layout_infer_type = kwargs.get('Layout_infer_type', 'bf16')
        self.MFD_infer_type = kwargs.get('MFD_infer_type', 'bf16')
        self.MFR_enc_infer_type = kwargs.get('MFR_enc_infer_type', 'bf16')
        self.MFR_dec_infer_type = kwargs.get('MFR_dec_infer_type', 'bf16')
        self.OCR_det_infer_type = kwargs.get('OCR_det_infer_type', 'bf16')
        self.OCR_rec_infer_type = kwargs.get('OCR_rec_infer_type', 'bf16')
        self.wired_table_type = kwargs.get('wired_table_type', 'bf16')
        self.WirelessTable_type = kwargs.get('WirelessTable_type', 'bf16')
        self.img_orientation_cls_type = kwargs.get('img_orientation_cls_type', 'bf16')
        self.table_cls_type = kwargs.get('table_cls_type', 'bf16')
        self.layoutreader_type = kwargs.get('layoutreader_type', 'bf16')
        self.nstreams = kwargs.get('nstreams', 1)
        self.enable_cache = kwargs.get('enable_cache', True)
        self.atom_model_manager = AtomModelSingleton()

        self.mfd_model_path = os.path.join(auto_download_and_get_model_root_path(ModelPath.yolo_v8_mfd), ModelPath.yolo_v8_mfd)
        if MFR_MODEL == "unimernet_small":
            self.mfr_model_path = ModelPath.unimernet_small
        elif MFR_MODEL == "pp_formulanet_plus_m":
            self.mfr_model_path = ModelPath.pp_formulanet_plus_m
        else:
            logger.error('MFR model name not allow')
            exit(1)
        self.mfr_model_path = os.path.join(auto_download_and_get_model_root_path(self.mfr_model_path), self.mfr_model_path)
        self.doclayout_model_path = os.path.join(auto_download_and_get_model_root_path(ModelPath.doclayout_yolo), ModelPath.doclayout_yolo)
        
        logger.info(f'DocAnalysis init, this may take some times (enable_cache={self.enable_cache})......')
        if self.enable_cache:
            self.init_models()
        logger.info('DocAnalysis init done!')

    def get_mfd_model(self):
        return self.atom_model_manager.get_atom_model(
                enable_cache = self.enable_cache,
                atom_model_name=AtomicModel.MFD,
                mfd_weights=str(self.mfd_model_path),
                enable_ov = self.enable_ov,
                MFD_infer_type = self.MFD_infer_type,
                device=self.device,
            )

    def get_mfr_model(self):
        return self.atom_model_manager.get_atom_model(
            enable_cache = self.enable_cache,
            atom_model_name=AtomicModel.MFR,
            mfr_weight_dir=self.mfr_model_path,
            enable_ov = self.enable_ov,
            MFR_enc_infer_type = self.MFR_enc_infer_type,
            MFR_dec_infer_type = self.MFR_dec_infer_type,
            device=self.device,
        )

    def get_layout_model(self):
        return self.atom_model_manager.get_atom_model(
            enable_cache = self.enable_cache,
            atom_model_name=AtomicModel.Layout,
            doclayout_yolo_weights=str(self.doclayout_model_path),
            enable_ov = self.enable_ov,
            Layout_infer_type = self.Layout_infer_type,
            device=self.device,
        )

    def get_ocr_model(self, det_db_box_thresh=0.3, det_db_unclip_ratio=1.8, enable_merge_det_boxes=True, lang=None):
        return self.atom_model_manager.get_atom_model(
            enable_cache = self.enable_cache,
            atom_model_name=AtomicModel.OCR,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio,
            enable_merge_det_boxes=enable_merge_det_boxes,
            lang=lang,
            enable_ov = self.enable_ov,
            OCR_det_infer_type = self.OCR_det_infer_type,
            OCR_rec_infer_type = self.OCR_rec_infer_type,
            nstreams = self.nstreams,
        )
        
    def get_wired_model(self, lang=None):
        return self.atom_model_manager.get_atom_model(
            enable_cache = self.enable_cache,
            atom_model_name=AtomicModel.WiredTable,
            lang=lang,
            enable_ov = self.enable_ov,
            wired_table_type = self.wired_table_type,
            OCR_det_infer_type = self.OCR_det_infer_type,
            OCR_rec_infer_type = self.OCR_rec_infer_type,
            nstreams = self.nstreams,
        )

    def get_wireless_model(self, lang=None):
        return self.atom_model_manager.get_atom_model(
            enable_cache = self.enable_cache,
            atom_model_name=AtomicModel.WirelessTable,
            lang=lang,
            enable_ov = self.enable_ov,
            WirelessTable_type = self.WirelessTable_type,
            OCR_det_infer_type = self.OCR_det_infer_type,
            OCR_rec_infer_type = self.OCR_rec_infer_type,
            nstreams = self.nstreams,
        )

    def get_table_cls_model(self):
        return self.atom_model_manager.get_atom_model(
            enable_cache=self.enable_cache,
            atom_model_name=AtomicModel.TableCls,
            enable_ov = self.enable_ov,
            table_cls_type = self.table_cls_type,
            OCR_det_infer_type = self.OCR_det_infer_type,
            OCR_rec_infer_type = self.OCR_rec_infer_type,
            nstreams = self.nstreams,
        )
        
    def get_img_ori_model(self, lang=None):
        return self.atom_model_manager.get_atom_model(
            enable_cache=self.enable_cache,
            atom_model_name=AtomicModel.ImgOrientationCls,
            enable_ov = self.enable_ov,
            img_orientation_cls_type = self.img_orientation_cls_type,
            OCR_det_infer_type = self.OCR_det_infer_type,
            OCR_rec_infer_type = self.OCR_rec_infer_type,
            nstreams = self.nstreams,
            lang=lang,
        )
        
    def get_layout_reader_model(self, model_name):
        return self.atom_model_manager.get_atom_model(
            enable_cache=self.enable_cache,
            atom_model_name=model_name,
            enable_ov = self.enable_ov,
            layoutlm_infer_type = self.layoutreader_type,
        )

    def init_models(self):
        # Initialize formula detection model
        if self.apply_formula:
            self.get_mfd_model()
            self.get_mfr_model()
        # Initialize layout model
        self.get_layout_model()
        # initializeocr
        self.get_ocr_model(lang='ch')
        self.get_ocr_model(lang='korean')
        self.get_ocr_model(lang='latin')
        self.get_ocr_model(lang='east_slavic')
        self.get_ocr_model(lang='en')
        self.get_ocr_model(lang='el')
        self.get_ocr_model(lang='ta')
        self.get_ocr_model(lang='te')
        self.get_ocr_model(lang='th')
        # init table model
        if self.apply_table:
            self.get_wired_model()
            self.get_wireless_model()
            self.get_table_cls_model()
            self.get_img_ori_model()
        self.get_layout_reader_model("layoutreader")

    def remove_unused_weight(self):
        if self.apply_formula:
            self.get_mfd_model().remove_unused_weight()
            self.get_mfr_model().remove_unused_weight()
        # Initialize layout model
        self.get_layout_model().remove_unused_weight()
        # initializeocr
        self.get_ocr_model(lang='ch').remove_unused_weight()
        self.get_ocr_model(lang='korean').remove_unused_weight()
        self.get_ocr_model(lang='latin').remove_unused_weight()
        self.get_ocr_model(lang='east_slavic').remove_unused_weight()
        self.get_ocr_model(lang='en').remove_unused_weight()
        self.get_ocr_model(lang='el').remove_unused_weight()
        self.get_ocr_model(lang='ta').remove_unused_weight()
        self.get_ocr_model(lang='te').remove_unused_weight()
        self.get_ocr_model(lang='th').remove_unused_weight()
        # init table model
        if self.apply_table:
            self.get_wired_model().remove_unused_weight()
            self.get_wireless_model().remove_unused_weight()
            self.get_table_cls_model().remove_unused_weight()
            self.get_img_ori_model().remove_unused_weight()
        self.get_layout_reader_model("layoutreader").remove_unused_weight()
