import torch
import os
 

from magic_pdf.config.constants import MODEL_NAME
from magic_pdf.model.model_list import AtomicModel
from magic_pdf.model.sub_modules.language_detection.yolov11.YOLOv11 import YOLOv11LangDetModel
from magic_pdf.model.sub_modules.layout.doclayout_yolo.DocLayoutYOLO import DocLayoutYOLOModel
from magic_pdf.model.sub_modules.mfd.yolov8.YOLOv8 import YOLOv8MFDModel
from magic_pdf.model.sub_modules.mfr.unimernet.Unimernet import UnimernetModel
from magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.pytorch_paddle import PytorchPaddleOCR
from magic_pdf.model.sub_modules.table.rapidtable.rapid_table import RapidTableModel
from magic_pdf.libs.config_reader import get_local_layoutreader_model_dir
from magic_pdf.model.sub_modules.ov_operator_async import LayoutReaderProcessor
from magic_pdf.libs.config_reader import get_device, get_local_models_dir

def table_model_init(table_model_type, model_path, max_time, enable_ov,
                     OCR_det_infer_type, OCR_rec_infer_type, Table_infer_type, nstreams,
                     _device_='cpu', lang=None, table_sub_model_name=None):
    # print(f"###✅ table_model_init table_model_type={table_model_type}, model_path={model_path}, "
    #       f"enable_ov={enable_ov}, OCR_det_infer_type={OCR_det_infer_type}, "
    #       f"OCR_rec_infer_type={OCR_rec_infer_type}, Table_infer_type={Table_infer_type}, "
    #       f"nstreams={nstreams}, table_sub_model_name={table_sub_model_name}")
    if table_model_type == MODEL_NAME.STRUCT_EQTABLE:
        from magic_pdf.model.sub_modules.table.structeqtable.struct_eqtable import StructTableModel
        table_model = StructTableModel(model_path, max_new_tokens=2048, max_time=max_time)
    elif table_model_type == MODEL_NAME.TABLE_MASTER:
        from magic_pdf.model.sub_modules.table.tablemaster.tablemaster_paddle import TableMasterPaddleModel
        config = {
            'model_dir': model_path,
            'device': _device_,
            'enable_ov': enable_ov,
            'infer_type_det': OCR_det_infer_type,
            'infer_type_rec': OCR_rec_infer_type,
            'nstreams': nstreams,
        }
        table_model = TableMasterPaddleModel(config)
    elif table_model_type == MODEL_NAME.RAPID_TABLE:
        atom_model_manager = AtomModelSingleton()
        ocr_engine = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            enable_ov=enable_ov,
            infer_type_det = OCR_det_infer_type,
            infer_type_rec = OCR_rec_infer_type,
            nstreams = nstreams,
            ocr_show_log=False,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            lang=lang
        )
        table_model = RapidTableModel(model_path, ocr_engine, table_sub_model_name, enable_ov, Table_infer_type)
    else:
        # logger.error('table model type not allow')
        exit(1)

    return table_model

def mfd_model_init(weight_dir, enable_ov, infer_type, device='cpu'):
    # print(f"###✅ mfd_model_init weight={weight_dir}, "
    #       f"enable_ov={enable_ov}, infer_type={infer_type}")
    mfd_model = YOLOv8MFDModel(weight_dir, enable_ov, infer_type, device)
    return mfd_model

def mfr_model_init(weight_dir, cfg_path, enable_ov, infer_type_enc, infer_type_dec, device='cpu'):
    # print(f"###✅ mfr_model_init weight_dir={weight_dir}, cfg_path={cfg_path}, "
    #       f"enable_ov={enable_ov}, infer_type_enc={infer_type_enc}, infer_type_dec={infer_type_dec}")
    mfr_model = UnimernetModel(weight_dir, cfg_path, enable_ov, infer_type_enc, infer_type_dec, device)
    return mfr_model

def layout_model_init(weight_dir, config_file, enable_ov, infer_type, device):
    # print(f"###✅ layout_model_init weight_dir={weight_dir}, config_file={config_file}, "
    #       f"enable_ov={enable_ov}, infer_type={infer_type}")
    from magic_pdf.model.sub_modules.layout.layoutlmv3.model_init import Layoutlmv3_Predictor
    model = Layoutlmv3_Predictor(weight, config_file, enable_ov, infer_type, device)
    return model

def doclayout_yolo_model_init(weight_dir, enable_ov, infer_type, device='cpu'):
    # print(f"###✅ doclayout_yolo_model_init weight_dir={weight_dir}, "
    #       f"enable_ov={enable_ov}, infer_type={infer_type}")
    model = DocLayoutYOLOModel(weight_dir, enable_ov, infer_type, device)
    return model

def langdetect_model_init(weight_dir, enable_ov, infer_type, device='cpu'):
    # print(f"###✅ langdetect_model_init weight_dir={weight_dir}, "
    #       f"enable_ov={enable_ov}, infer_type={infer_type}")
    model = YOLOv11LangDetModel(weight_dir, enable_ov, infer_type, device)
    return model

def layoutreader_model_init(model_name: str, enable_ov, infer_type):
    layoutreader_model_dir = get_local_layoutreader_model_dir()
    # print(f"###✅ layoutreader_model_init model_path={layoutreader_model_dir}, "
    #             f"enable_ov={enable_ov}, infer_type={infer_type}")
    device = torch.device("cpu")
    if model_name == 'layoutreader':
        # 检测modelscope的缓存目录是否存在
        if enable_ov :
            layoutreader_model_dir_ov = layoutreader_model_dir + "/layoutreader.xml"
            if os.path.exists(layoutreader_model_dir_ov):
                ov_model = LayoutReaderProcessor(layoutreader_model_dir_ov)
                ov_model.setup_model(stream_num = 1, infer_type=infer_type)
                return ov_model
        from transformers import LayoutLMv3ForTokenClassification
        class LayoutLMv3ForTokenClassificationWrapper:
            def __init__(self, model, model_path, infer_type, device) :
                self.device = device
                self.infer_type = infer_type
                self.model = model
                if self.infer_type=='bf16':
                    self.model = model.to(self.device).eval().bfloat16()
                elif self.infer_type=='f16':
                    self.model = model.to(self.device).eval().float16()
                else:
                    self.model = model.to(self.device).eval()
                self.dtype = self.model.dtype
                self.using_ov = False
                self.model_path = model_path
                # print(f"### LayoutLMv3ForTokenClassificationWrapper infer_type={infer_type}, dtype={self.dtype}")
            
            def eval(self) :
                return self.model.eval()
                
            def __call__(self, **kargs):
                return self.model(**kargs)

        if os.path.exists(layoutreader_model_dir):
            model = LayoutLMv3ForTokenClassification.from_pretrained(layoutreader_model_dir)
        else:
            # logger.warning('local layoutreader model not exists, use online model from huggingface')
            model = LayoutLMv3ForTokenClassification.from_pretrained('hantian/layoutreader')
        model_wrapper = LayoutLMv3ForTokenClassificationWrapper(model, layoutreader_model_dir, infer_type, device)
    else:
        # logger.error('model name not allow')
        exit(1)
    return model_wrapper

def ocr_model_init(enable_ov: bool, infer_type_det: str, infer_type_rec: str, nstreams: int = 1, show_log: bool = False,
                   det_db_box_thresh=0.3, lang=None, use_dilation=True, det_db_unclip_ratio=1.8,):
    ocr_models_dir = os.path.join(get_local_models_dir(), 'OCR', 'paddleocr_torch')
    # print(f"###✅ ocr_model_init lang={lang}, ocr_models_dir={ocr_models_dir}, "
    #       f"enable_ov={enable_ov}, infer_type_det={infer_type_det}, "
    #       f"infer_type_rec={infer_type_rec}, nstreams={nstreams}")
    if lang is not None and lang != '':
        model = PytorchPaddleOCR(
            enable_ov=enable_ov,
            infer_type_det=infer_type_det,
            infer_type_rec=infer_type_rec,
            nstreams=nstreams,
            show_log=show_log,
            det_db_box_thresh=det_db_box_thresh,
            lang=lang,
            use_dilation=use_dilation,
            det_db_unclip_ratio=det_db_unclip_ratio,
        )
    else:
        model = PytorchPaddleOCR(
            enable_ov=enable_ov,
            infer_type_det=infer_type_det,
            infer_type_rec=infer_type_rec,
            nstreams=nstreams,
            show_log=show_log,
            det_db_box_thresh=det_db_box_thresh,
            use_dilation=use_dilation,
            det_db_unclip_ratio=det_db_unclip_ratio,
        )
    return model

class AtomModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_atom_model(self, atom_model_name: str, **kwargs):
        lang = kwargs.get('lang', None)
        layout_model_name = kwargs.get('layout_model_name', None)
        table_model_name = kwargs.get('table_model_name', None)

        if atom_model_name in [AtomicModel.OCR]:
            key = (atom_model_name, lang)
            if key not in self._models:
                self._models[key] = atom_model_init(model_name=atom_model_name, **kwargs)
            return self._models[key]
        else :
            return atom_model_init(model_name=atom_model_name, **kwargs)
    
    def get_atom_model_map(self, atom_model_name: str, **kwargs):
        lang = kwargs.get('lang', None)
        layout_model_name = kwargs.get('layout_model_name', None)
        table_model_name = kwargs.get('table_model_name', None)

        if atom_model_name in [AtomicModel.OCR]:
            key = (atom_model_name, lang)
        elif atom_model_name in [AtomicModel.Layout]:
            key = (atom_model_name, layout_model_name)
        elif atom_model_name in [AtomicModel.Table]:
            key = (atom_model_name, table_model_name, lang)
        else:
            key = atom_model_name

        if key not in self._models:
            self._models[key] = atom_model_init(model_name=atom_model_name, **kwargs)
        return self._models[key]

def atom_model_init(model_name: str, **kwargs):
    atom_model = None
    if model_name == AtomicModel.Layout:
        if kwargs.get('layout_model_name') == MODEL_NAME.LAYOUTLMv3:
            atom_model = layout_model_init(
                kwargs.get('layout_weights'),
                kwargs.get('layout_config_file'),
                kwargs.get('enable_ov'),
                kwargs.get('Layout_infer_type'),
                kwargs.get('device')
            )
        elif kwargs.get('layout_model_name') == MODEL_NAME.DocLayout_YOLO:
            atom_model = doclayout_yolo_model_init(
                kwargs.get('doclayout_yolo_weights'),
                kwargs.get('enable_ov'),
                kwargs.get('Layout_infer_type'),
                kwargs.get('device')
            )
        else:
            # logger.error('layout model name not allow')
            exit(1)
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
            kwargs.get('mfr_cfg_path'),
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
            kwargs.get('ocr_show_log'),
            kwargs.get('det_db_box_thresh'),
            kwargs.get('lang'),
        )
    elif model_name == AtomicModel.Table:
        atom_model = table_model_init(
            kwargs.get('table_model_name'),
            kwargs.get('table_model_path'),
            kwargs.get('table_max_time'),
            kwargs.get('enable_ov'),
            kwargs.get('OCR_det_infer_type'),
            kwargs.get('OCR_rec_infer_type'),
            kwargs.get('Table_infer_type'),
            kwargs.get('nstreams'),
            kwargs.get('device'),
            kwargs.get('lang'),
            kwargs.get('table_sub_model_name')
        )
    elif model_name == AtomicModel.LangDetect:
        if kwargs.get('langdetect_model_name') == MODEL_NAME.YOLO_V11_LangDetect:
            atom_model = langdetect_model_init(
                kwargs.get('langdetect_model_weight'),
                kwargs.get('enable_ov'),
                kwargs.get('Lang_infer_type'),
                kwargs.get('device')
            )
        else:
            # logger.error('langdetect model name not allow')
            exit(1)
    elif model_name == AtomicModel.LayoutReader:
        atom_model = layoutreader_model_init(
            MODEL_NAME.LayoutReader,
            kwargs.get('enable_ov'),
            kwargs.get('Page_infer_type')
        )
    else:
        # logger.error('model name not allow')
        exit(1)

    if atom_model is None:
        # logger.error('model init failed')
        exit(1)
    else:
        return atom_model
