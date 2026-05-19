# Copyright (c) Opendatalab. All rights reserved.
import os

from PIL import Image
from collections import defaultdict
from typing import List, Dict
from tqdm import tqdm
import cv2
import numpy as np
import onnxruntime

from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from mineru.model.ov_operator_async import OnnxSessProcessor

class PaddleOrientationClsModel:
    def __init__(self, enable_ov, infer_type, ocr_engine):
        self.sess = None
        self.model_path = os.path.join(auto_download_and_get_model_root_path(ModelPath.paddle_orientation_classification), ModelPath.paddle_orientation_classification)
        self.enable_ov = enable_ov
        self.infer_type = infer_type
        if self.enable_ov:
            try :
                self.sess = OnnxSessProcessor(self.model_path, "PaddleOrientationClsModel")
                self.sess.setup_model(1, self.infer_type)
            except Exception as e:
                print(f"Failed to initialize OpenVINO model, falling back to ONNX. Error: {e}")
                self.sess = None
        if self.sess is None:
            self.enable_ov = False
            self.sess = onnxruntime.InferenceSession(self.model_path)
        self.ocr_engine = ocr_engine
        self.less_length = 256
        self.cw, self.ch = 224, 224
        self.std = [0.229, 0.224, 0.225]
        self.scale = 0.00392156862745098
        self.mean = [0.485, 0.456, 0.406]
        self.labels = ["0", "90", "180", "270"]
    
    def remove_unused_weight(self) :
        pass

    def preprocess(self, input_img):
        # Enlarge the image so that its shortest side is 256
        h, w = input_img.shape[:2]
        scale = 256 / min(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        img = cv2.resize(input_img, (w_resize, h_resize), interpolation=1)
        # adjusted to 224*224square
        h, w = img.shape[:2]
        cw, ch = 224, 224
        x1 = max(0, (w - cw) // 2)
        y1 = max(0, (h - ch) // 2)
        x2 = min(w, x1 + cw)
        y2 = min(h, y1 + ch)
        if w < cw or h < ch:
            raise ValueError(
                f"Input image ({w}, {h}) smaller than the target size ({cw}, {ch})."
            )
        img = img[y1:y2, x1:x2, ...]
        # regularization
        split_im = list(cv2.split(img))
        std = [0.229, 0.224, 0.225]
        scale = 0.00392156862745098
        mean = [0.485, 0.456, 0.406]
        alpha = [scale / std[i] for i in range(len(std))]
        beta = [-mean[i] / std[i] for i in range(len(std))]
        for c in range(img.shape[2]):
            split_im[c] = split_im[c].astype(np.float32)
            split_im[c] *= alpha[c]
            split_im[c] += beta[c]
        img = cv2.merge(split_im)
        # 5. Convert to CHW Format
        img = img.transpose((2, 0, 1))
        imgs = [img]
        x = np.stack(imgs, axis=0).astype(dtype=np.float32, copy=False)
        return x

    def predict(self, input_img):
        rotate_label = "0"  # Default to 0 if no rotation detected or not portrait
        if isinstance(input_img, Image.Image):
            np_img = np.asarray(input_img)
        elif isinstance(input_img, np.ndarray):
            np_img = input_img
        else:
            raise ValueError("Input must be a pillow object or a numpy array.")
        bgr_image = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        # First check the overall image aspect ratio (height/width)
        img_height, img_width = bgr_image.shape[:2]
        img_aspect_ratio = img_height / img_width if img_width > 0 else 1.0
        img_is_portrait = img_aspect_ratio > 1.2

        if img_is_portrait:

            det_res = self.ocr_engine.ocr(bgr_image, rec=False)[0]
            # Check if table is rotated by analyzing text box aspect ratios
            if det_res:
                vertical_count = 0
                is_rotated = False

                for box_ocr_res in det_res:
                    p1, p2, p3, p4 = box_ocr_res

                    # Calculate width and height
                    width = p3[0] - p1[0]
                    height = p3[1] - p1[1]

                    aspect_ratio = width / height if height > 0 else 1.0

                    # Count vertical vs horizontal text boxes
                    if aspect_ratio < 0.8:  # Taller than wide - vertical text
                        vertical_count += 1
                    # elif aspect_ratio > 1.2:  # Wider than tall - horizontal text
                    #     horizontal_count += 1

                if vertical_count >= len(det_res) * 0.28 and vertical_count >= 3:
                    is_rotated = True
                # logger.debug(f"Text orientation analysis: vertical={vertical_count}, det_res={len(det_res)}, rotated={is_rotated}")

                # If we have more vertical text boxes than horizontal ones,
                # and vertical ones are significant, table might be rotated
                if is_rotated:
                    x = self.preprocess(np_img)
                    (result,) = self.sess.run(None, {"x": x})
                    rotate_label = self.labels[np.argmax(result)]
                    # logger.debug(f"Orientation classification result: {label}")

        return rotate_label

    def list_2_batch(self, img_list, batch_size=16):
        """
        Convert a list of any length into a specified batch sizeDivide into multiple batches

        Args:
            img_list: input list
            batch_size: The size of each batch, default is 16

        Returns:
            A list containing multiple batches, each batch is a sublist of the original list
        """
        batches = []
        for i in range(0, len(img_list), batch_size):
            batch = img_list[i : min(i + batch_size, len(img_list))]
            batches.append(batch)
        return batches

    def batch_preprocess(self, imgs):
        res_imgs = []
        for img_info in imgs:
            img = np.asarray(img_info["table_img"])
            # Enlarge the image so that its shortest side is 256
            h, w = img.shape[:2]
            scale = 256 / min(h, w)
            h_resize = round(h * scale)
            w_resize = round(w * scale)
            img = cv2.resize(img, (w_resize, h_resize), interpolation=1)
            # adjusted to 224*224square
            h, w = img.shape[:2]
            cw, ch = 224, 224
            x1 = max(0, (w - cw) // 2)
            y1 = max(0, (h - ch) // 2)
            x2 = min(w, x1 + cw)
            y2 = min(h, y1 + ch)
            if w < cw or h < ch:
                raise ValueError(
                    f"Input image ({w}, {h}) smaller than the target size ({cw}, {ch})."
                )
            img = img[y1:y2, x1:x2, ...]
            # regularization
            split_im = list(cv2.split(img))
            std = [0.229, 0.224, 0.225]
            scale = 0.00392156862745098
            mean = [0.485, 0.456, 0.406]
            alpha = [scale / std[i] for i in range(len(std))]
            beta = [-mean[i] / std[i] for i in range(len(std))]
            for c in range(img.shape[2]):
                split_im[c] = split_im[c].astype(np.float32)
                split_im[c] *= alpha[c]
                split_im[c] += beta[c]
            img = cv2.merge(split_im)
            # 5. Convert to CHW Format
            img = img.transpose((2, 0, 1))
            res_imgs.append(img)
        x = np.stack(res_imgs, axis=0).astype(dtype=np.float32, copy=False)
        return x

    def batch_predict(
        self, imgs: List[Dict], det_batch_size: int, batch_size: int = 16, tqdm_enable: bool = False
    ) -> None:

        import torch
        from packaging import version
        if version.parse(torch.__version__) >= version.parse("2.8.0"):
            return None

        """
        Batch predict the incoming rotation information containing the picture information list, and correctly rotate the rotated pictures back
        """
        RESOLUTION_GROUP_STRIDE = 128
        # Skip aspect ratio less than 1.2pictures
        resolution_groups = defaultdict(list)
        for img in imgs:
            # RGBImage Conversion BGR
            bgr_img: np.ndarray = cv2.cvtColor(np.asarray(img["table_img"]), cv2.COLOR_RGB2BGR)
            img["table_img_bgr"] = bgr_img
            img_height, img_width = bgr_img.shape[:2]
            img_aspect_ratio = img_height / img_width if img_width > 0 else 1.0
            if img_aspect_ratio > 1.2:
                # Normalize dimensions to multiples of RESOLUTION_GROUP_STRIDE
                normalized_h = ((img_height + RESOLUTION_GROUP_STRIDE) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE  # Round up to a multiple of RESOLUTION_GROUP_STRIDE
                normalized_w = ((img_width + RESOLUTION_GROUP_STRIDE) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                group_key = (normalized_h, normalized_w)
                resolution_groups[group_key].append(img)

        # Batch processing per resolution group
        rotated_imgs = []
        tpdm_desc = f"Table-ori stage1 Predict with OV_{self.infer_type}" if self.enable_ov else "Table-ori stage1 Predict"
        for group_key, group_imgs in tqdm(resolution_groups.items(), desc=tpdm_desc, disable=not tqdm_enable):
            # Calculate the target size (the largest size in the group, rounded up to a multiple of RESOLUTION_GROUP_STRIDE)
            max_h = max(img["table_img_bgr"].shape[0] for img in group_imgs)
            max_w = max(img["table_img_bgr"].shape[1] for img in group_imgs)
            target_h = ((max_h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
            target_w = ((max_w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE

            # Pad all images to the same size
            batch_images = []
            for img in group_imgs:
                bgr_img = img["table_img_bgr"]
                h, w = bgr_img.shape[:2]
                # Create a white background of target size
                padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
                # Paste the original image into the upper left corner
                padded_img[:h, :w] = bgr_img
                batch_images.append(padded_img)

            # Batch detection
            batch_results = self.ocr_engine.text_detector.batch_predict(
                batch_images, min(len(batch_images), det_batch_size)
            )

            # Detect whether the image is rotated based on batch processing results,Put the rotated image into the list and continue to predict the rotation angle

            for index, (img_info, (dt_boxes, elapse)) in enumerate(
                zip(group_imgs, batch_results)
            ):
                vertical_count = 0
                for box_ocr_res in dt_boxes:
                    p1, p2, p3, p4 = box_ocr_res

                    # Calculate width and height
                    width = p3[0] - p1[0]
                    height = p3[1] - p1[1]

                    aspect_ratio = width / height if height > 0 else 1.0

                    # Count vertical text boxes
                    if aspect_ratio < 0.8:  # Taller than wide - vertical text
                        vertical_count += 1

                if vertical_count >= len(dt_boxes) * 0.28 and vertical_count >= 3:
                    rotated_imgs.append(img_info)

        # Rotation angle prediction for rotated pictures
        if len(rotated_imgs) > 0:
            imgs = self.list_2_batch(rotated_imgs, batch_size=batch_size)
            tqdm_desc = f"Table-ori stage2 Predict with OV_{self.infer_type}" if self.enable_ov else "Table-ori cls stage2 Predict"
            with tqdm(total=len(rotated_imgs), desc=tqdm_desc, disable=not tqdm_enable) as pbar:
                for img_batch in imgs:
                    x = self.batch_preprocess(img_batch)
                    results = self.sess.run(None, {"x": x})
                    for img_info, res in zip(rotated_imgs, results[0]):
                        label = self.labels[np.argmax(res)]
                        self.img_rotate(img_info, label)
                        pbar.update(1)

    def img_rotate(self, img_info, label):
        if label == "270":
            img_info["table_img"] = cv2.rotate(
                np.asarray(img_info["table_img"]),
                cv2.ROTATE_90_CLOCKWISE,
            )
            img_info["wired_table_img"] = cv2.rotate(
                np.asarray(img_info["wired_table_img"]),
                cv2.ROTATE_90_CLOCKWISE,
            )
        elif label == "90":
            img_info["table_img"] = cv2.rotate(
                np.asarray(img_info["table_img"]),
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )
            img_info["wired_table_img"] = cv2.rotate(
                np.asarray(img_info["wired_table_img"]),
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )
        else:
            # 180Degrees and 0 degrees are not processed.
            pass
