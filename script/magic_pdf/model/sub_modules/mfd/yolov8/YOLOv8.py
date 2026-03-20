from tqdm import tqdm
from ultralytics import YOLO
import os
import time
import torch
from ...ov_operator_async import YoloProcessor

class YOLOv8MFDModel(object):
    def __init__(self, weight, enable_ov, infer_type, device="cpu"):
        self.enable_ov = enable_ov
        file_name = os.path.basename(weight)
        file_name_without_extension = os.path.splitext(file_name)[0]
        self.ov_file_name = f"{weight}/{file_name_without_extension}.xml".replace(".pt", "_openvino_model")
        try :
            if self.enable_ov:
                self.mfd_model = YOLO(f"{weight}".replace(".pt", "_openvino_model"), task="detect")
            else :
                self.mfd_model = YOLO(weight, task="detect")
        except Exception as e:
            print(f"### Error loading YOLO model from {weight}: {str(e)}")
            self.mfd_model = YOLO(weight, task="detect")

        self.device = device
        self.infer_type = infer_type
        # self.ov_file_name = f"{weight}".replace(".pt", ".onnx")
        if self.enable_ov:
            if not os.path.isfile(self.ov_file_name) :
                    path = self.mfd_model.export(format="openvino", dynamic=True, simplify=False, device="CPU")  
                    print(f"### export YOLO from {weight} to {path}, ov_file={self.ov_file_name}")
            self.ov_yolo = YoloProcessor(self.ov_file_name)
            self.ov_yolo.setup_model(stream_num = 1, infer_type=self.infer_type)
            args={'task': 'detect', 'imgsz': 1888, 'conf': 0.25, 'iou': 0.45, 'batch': 1, 'mode': 'predict',
                  'verbose': False, 'single_cls': False, 'save': False, 'rect': True, 'device': 'cpu'}
            self.mfd_model.predictor = (self.mfd_model._smart_load("predictor"))(overrides=args)
            self.mfd_model.predictor.setup_model(model=self.mfd_model.model, verbose=False)
            def infer(*args):
                result = self.ov_yolo(args)
                return torch.from_numpy(result[0])
            self.mfd_model.predictor.inference = infer
            # self.mfd_model.predictor.model.pt = False
        else :
            self.ov_yolo = None

    def predict(self, image):
        mfd_res = self.mfd_model.predict(
                image, imgsz=1888, conf=0.25, iou=0.45, verbose=False, device=self.device
            )[0]
        return mfd_res

    def batch_predict(self, images: list, batch_size: int, tqdm_enable = False) -> list:
        images_mfd_res = []
        if self.ov_yolo is not None :
            desc_str = f"MFD_OV_{self.infer_type} Predict"
            for index in tqdm(range(0, len(images), batch_size), desc=desc_str, disable=not tqdm_enable):
                mfd_res = [
                    image_res.cpu()
                    for image_res in self.mfd_model.predict(
                        images[index : index + batch_size],
                        imgsz=1888,
                        conf=0.25,
                        iou=0.45,
                        verbose=False,
                        device=self.device,
                    )
                ]
                for image_res in mfd_res:
                    images_mfd_res.append(image_res)
        else :
            desc_str = f"MFD_{self.infer_type} Predict"
            for index in tqdm(range(0, len(images), batch_size), desc=desc_str, disable=not tqdm_enable):
                mfd_res = [
                    image_res.cpu()
                    for image_res in self.mfd_model.predict(
                        images[index : index + batch_size],
                        imgsz=1888,
                        conf=0.25,
                        iou=0.45,
                        verbose=False,
                        device=self.device,
                    )
                ]
                for image_res in mfd_res:
                    images_mfd_res.append(image_res)
        return images_mfd_res
