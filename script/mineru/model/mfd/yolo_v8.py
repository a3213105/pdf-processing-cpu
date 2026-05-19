import os
from typing import List, Union

import torch
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw
from mineru.model.ov_operator_async import OnnxSessProcessor
import torch

from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
class YOLOv8MFDModel:
    def __init__(
        self,
        weight: str,
        enable_ov: bool,
        infer_type: str,
        device: str = "cpu",
        imgsz: int = 1888,
        conf: float = 0.25,
        iou: float = 0.45,
    ):
        self.device = torch.device(device)
        self.torch_path = weight
        self.enable_ov = enable_ov
        file_name = os.path.basename(weight)
        file_name_without_extension = os.path.splitext(file_name)[0]
        self.ov_file_name = f"{weight}/{file_name_without_extension}.xml".replace(".pt", "_openvino_model")
        try :
            if self.enable_ov:
                self.model = YOLO(f"{weight}".replace(".pt", "_openvino_model"), task="detect")
            else :
                self.model = YOLO(weight, task="detect")
        except Exception as e:
            print(f"### Error loading YOLO model from {weight}: {str(e)}")
            self.model = YOLO(weight, task="detect")

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

        self.infer_type = infer_type
        # self.ov_file_name = f"{weight}".replace(".pt", ".onnx")
        if self.enable_ov:
            if not os.path.isfile(self.ov_file_name) :
                    path = self.model.export(format="openvino", dynamic=True, simplify=False, device="CPU")  
                    print(f"### export YOLO from {weight} to {path}, ov_file={self.ov_file_name}")
            self.ov_yolo = OnnxSessProcessor(self.ov_file_name, "YOLOv8")
            self.ov_yolo.setup_model(stream_num = 1, infer_type=self.infer_type)
            args={'task': 'detect', 'imgsz': self.imgsz, 'conf': self.conf, 'iou': self.iou, 'batch': 1, 'mode': 'predict',
                  'verbose': False, 'single_cls': False, 'save': False, 'rect': True, 'device': 'cpu'}
            self.model.predictor = (self.model._smart_load("predictor"))(overrides=args)
            self.model.predictor.setup_model(model=self.model.model, verbose=False)
            def infer(*args):
                result = self.ov_yolo(args)
                return torch.from_numpy(result[0])
            self.model.predictor.inference = infer
            # self.model.predictor.model.pt = False
        else :
            self.ov_yolo = None

    def remove_unused_weight(self) :
        if self.ov_yolo is not None:
            if os.path.isfile(self.torch_path):
                os.remove(self.torch_path)

    def _run_predict(
        self,
        inputs: Union[np.ndarray, Image.Image, List],
        is_batch: bool = False,
        conf: float = None,
    ) -> List:
        preds = self.model.predict(
            inputs,
            imgsz=self.imgsz,
            conf=conf if conf is not None else self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device
        )
        return [pred.cpu() for pred in preds] if is_batch else preds[0].cpu()

    def predict(
            self,
            image: Union[np.ndarray, Image.Image],
            conf: float = None,
    ):
        return self._run_predict(image, is_batch=False, conf=conf)

    def batch_predict(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 4,
        conf: float = None,
        tqdm_enable: bool = False
    ) -> List:
        results = []
        tqdm_desc = f"MFD Predict with OV_{self.infer_type}" if self.enable_ov else "MFD Predict"
        with tqdm(total=len(images), desc=tqdm_desc, disable=not tqdm_enable) as pbar:
            for idx in range(0, len(images), batch_size):
                batch = images[idx: idx + batch_size]
                batch_preds = self._run_predict(batch, is_batch=True, conf=conf)
                results.extend(batch_preds)
                pbar.update(len(batch))
        return results

    def visualize(
        self,
        image: Union[np.ndarray, Image.Image],
        results: List
    ) -> Image.Image:

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        formula_list = []
        for xyxy, conf, cla in zip(
                results.boxes.xyxy.cpu(), results.boxes.conf.cpu(), results.boxes.cls.cpu()
        ):
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            new_item = {
                "category_id": 13 + int(cla.item()),
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf.item()), 2),
            }
            formula_list.append(new_item)

        draw = ImageDraw.Draw(image)
        for res in formula_list:
            poly = res['poly']
            xmin, ymin, xmax, ymax = poly[0], poly[1], poly[4], poly[5]
            print(
                f"Detected box: {xmin}, {ymin}, {xmax}, {ymax}, Category ID: {res['category_id']}, Score: {res['score']}")
            # Draw a frame on an image using PIL
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
            # Draw confidence next to the box
            draw.text((xmax + 10, ymin + 10), f"{res['score']:.2f}", fill="red", font_size=22)
        return image

if __name__ == '__main__':
    image_path = r"C:\Users\zhaoxiaomeng\Downloads\screenshot-20250821-192948.png"
    yolo_v8_mfd_weights = os.path.join(auto_download_and_get_model_root_path(ModelPath.yolo_v8_mfd),
                                          ModelPath.yolo_v8_mfd)
    device = 'cuda'
    model = YOLOv8MFDModel(
        weight=yolo_v8_mfd_weights,
        device=device,
    )
    image = Image.open(image_path)
    results = model.predict(image)

    image = model.visualize(image, results)

    image.show()  # show image