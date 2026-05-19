import os
from typing import List, Dict, Union

from doclayout_yolo import YOLOv10
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw
from mineru.model.ov_operator_async import OnnxSessProcessor
import torch

from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path

class DocLayoutYOLOModel:
    def __init__(
        self,
        weight: str,
        enable_ov: bool,
        infer_type: str,
        device: str = "cuda",
        imgsz: int = 1280,
        conf: float = 0.1,
        iou: float = 0.45,
    ):
        self.model = None
        self.enable_ov = enable_ov
        self.torch_path = weight
        file_name = os.path.basename(weight)
        file_name_without_extension = os.path.splitext(file_name)[0]
        self.ov_file_name = f"{weight}/{file_name_without_extension}.xml".replace(".pt", "_openvino_model")
        try :
            if self.enable_ov:
                self.model = YOLOv10(f"{weight}".replace(".pt", "_openvino_model"), task="detect", verbose=False)
        except Exception as e:
            print(f"### Error loading YOLO model from {weight}: {str(e)}")

        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        
        if self.model is None :
            self.model = YOLOv10(weight, task="detect")

        self.infer_type = infer_type
        if self.enable_ov:
            if not os.path.isfile(self.ov_file_name) :
                    path = self.model.export(format="openvino", dynamic=True)  
                    print(f"### export YOLOv10 from {weight} to {path}, ov_file={self.ov_file_name}")
            self.ov_yolo = OnnxSessProcessor(self.ov_file_name, "YOLOv10")
            self.ov_yolo.setup_model(stream_num = 1, infer_type=self.infer_type)
            args={'task': 'detect',
                                  'imgsz': self.imgsz,
                                  'conf': self.conf,
                                  'iou': self.iou,
                                  'batch': 1,
                                  'mode': 'predict',
                                  'verbose': False,
                                  'single_cls': False,
                                  'save': False,
                                  'rect': True,
                                  'device': 'cpu'}
            self.model.predictor = (self.model._smart_load("predictor"))(overrides=args)
            self.model.predictor.setup_model(model=self.model.model, verbose=False)
            def infer(*args):
                result = self.ov_yolo(args)
                return torch.from_numpy(result[0])
            self.model.predictor.inference = infer
        else :
            self.ov_yolo = None
            
    def remove_unused_weight(self) :
        if self.ov_yolo is not None:
            if os.path.isfile(self.torch_path):
                os.remove(self.torch_path)

    def _parse_prediction(self, prediction) -> List[Dict]:
        layout_res = []

        # Fault tolerance
        if not hasattr(prediction, "boxes") or prediction.boxes is None:
            return layout_res

        for xyxy, conf, cls in zip(
            prediction.boxes.xyxy.cpu(),
            prediction.boxes.conf.cpu(),
            prediction.boxes.cls.cpu(),
        ):
            coords = list(map(int, xyxy.tolist()))
            xmin, ymin, xmax, ymax = coords
            layout_res.append({
                "category_id": int(cls.item()),
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf.item()), 3),
            })
        return layout_res

    def predict(self, image: Union[np.ndarray, Image.Image]) -> List[Dict]:
        prediction = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )[0]
        return self._parse_prediction(prediction)

    def batch_predict(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 4,
        tqdm_enable: bool = False
    ) -> List[List[Dict]]:
        results = []
        tqdm_desc = f"Layout Predict with OV_{self.infer_type}" if self.enable_ov else "Layout Predict"
        with tqdm(total=len(images), desc=tqdm_desc, disable=not tqdm_enable) as pbar:
            for idx in range(0, len(images), batch_size):
                batch = images[idx: idx + batch_size]
                if batch_size == 1:
                    conf = 0.9 * self.conf
                else:
                    conf = self.conf
                predictions = self.model.predict(
                    batch,
                    imgsz=self.imgsz,
                    conf=conf,
                    iou=self.iou,
                    verbose=False,
                )
                for pred in predictions:
                    results.append(self._parse_prediction(pred))
                pbar.update(len(batch))
        return results

    def visualize(
            self,
            image: Union[np.ndarray, Image.Image],
            results: List
    ) -> Image.Image:

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        draw = ImageDraw.Draw(image)
        for res in results:
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
    image_path = r"C:\Users\zhaoxiaomeng\Downloads\Download 1.jpg"
    doclayout_yolo_weights = os.path.join(auto_download_and_get_model_root_path(ModelPath.doclayout_yolo), ModelPath.doclayout_yolo)
    device = 'cuda'
    model = DocLayoutYOLOModel(
        weight=doclayout_yolo_weights,
        device=device,
    )
    image = Image.open(image_path)
    results = model.predict(image)

    image = model.visualize(image, results)

    image.show()  # show image