import os
import torch
import yaml
from pathlib import Path

from loguru import logger
from tqdm import tqdm
from mineru.model.utils.tools.infer import pytorchocr_utility
from mineru.model.utils.pytorchocr.base_ocr_v20 import BaseOCRV20
from .processors import (
    UniMERNetImgDecode,
    UniMERNetTestTransform,
    LatexImageFormat,
    ToBatch,
    UniMERNetDecode,
)
import openvino as ov
from mineru.model.ov_operator_async import OnnxSessProcessor

class FormulaRecognizer(BaseOCRV20):
    def __init__(
        self,
        weight_dir,
        enable_ov,
        infer_type,
        device="cpu",
    ):
        self.weights_path = os.path.join(
            weight_dir,
            "PP-FormulaNet_plus-M.pth",
        )
        self.yaml_path = os.path.join(
            Path(__file__).parent.parent.parent,
            "utils",
            "pytorchocr",
            "utils",
            "resources",
            "pp_formulanet_arch_config.yaml"
        )
        self.infer_yaml_path = os.path.join(
            weight_dir,
            "PP-FormulaNet_plus-M_inference.yml",
        )

        self.enable_ov = enable_ov
        self.infer_type = infer_type
        self.ov_file_name = f"{weight_dir}/PP-FormulaNet_plus-M.xml"

        if self.enable_ov:
            try:
                self.ov_net = OnnxSessProcessor(self.ov_file_name)
                self.ov_net.setup_model(stream_num = 1, infer_type=self.infer_type)
            except Exception as e:
                print(f"### PP-FormulaNet_plus-M init failed: {e}")
        else :
            network_config = pytorchocr_utility.AnalysisConfig(
                self.weights_path, self.yaml_path
            )
            weights = self.read_pytorch_weights(self.weights_path)

            super(FormulaRecognizer, self).__init__(network_config)

            self.load_state_dict(weights)
            self.device = torch.device(device) if isinstance(device, str) else device
            self.net.to(self.device)
            self.net.eval()
            
            if not os.path.isfile(self.ov_file_name):
                try:
                    ov_model = ov.convert_model(self.net)
                    ov.save_model(ov_model, self.ov_file_name, compress_to_fp16=False)
                    print(f"export ov model to {self.ov_file_name} ")
                except Exception as e:
                    print(f"### convert_model failed: {e}, try simple convert_model")


        with open(self.infer_yaml_path, "r", encoding="utf-8") as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.FullLoader)

        self.pre_tfs = {
            "UniMERNetImgDecode": UniMERNetImgDecode(input_size=(384, 384)),
            "UniMERNetTestTransform": UniMERNetTestTransform(),
            "LatexImageFormat": LatexImageFormat(),
            "ToBatch": ToBatch(),
        }

        self.post_op = UniMERNetDecode(
            character_list=data["PostProcess"]["character_dict"]
        )

    def remove_unused_weight(self) :
        if self.ov_net is not None and os.path.isfile(self.weights_path):
            try:
                os.remove(self.weights_path)
            except Exception as e:
                print(f"FormulaRecognizer Failed to remove {self.weights_path}: {e}")


    def predict(self, img_list, batch_size: int = 64, tqdm_enable: bool = False):
        # Reduce batch size by 50% to avoid potential memory issues during inference.
        batch_size = int(0.5 * batch_size)
        batch_imgs = self.pre_tfs["UniMERNetImgDecode"](imgs=img_list)
        batch_imgs = self.pre_tfs["UniMERNetTestTransform"](imgs=batch_imgs)
        batch_imgs = self.pre_tfs["LatexImageFormat"](imgs=batch_imgs)
        inp = self.pre_tfs["ToBatch"](imgs=batch_imgs)
        inp = torch.from_numpy(inp[0])
        inp = inp.to(self.device)
        rec_formula = []
        with torch.no_grad():
            tqdm_desc = f"MFR Predict with OV_{self.infer_type}" if self.enable_ov else "MFR Predict"
            with tqdm(total=len(inp), desc=tqdm_desc, disable=not tqdm_enable) as pbar:
                for index in range(0, len(inp), batch_size):
                    batch_data = inp[index: index + batch_size]
                    # with torch.amp.autocast(device_type=self.device.type):
                    #     batch_preds = [self.net(batch_data)]
                    batch_preds = [self.net(batch_data)]
                    batch_preds = [p.reshape([-1]) for p in batch_preds[0]]
                    batch_preds = [bp.cpu().numpy() for bp in batch_preds]
                    rec_formula += self.post_op(batch_preds)
                    pbar.update(len(batch_preds))
        return rec_formula

    def batch_predict(
        self,
        images_mfd_res: list,
        images: list,
        batch_size: int = 64,
        interline_enable: bool = True,
    ) -> list:
        images_formula_list = []
        mf_image_list = []
        backfill_list = []
        image_info = []  # Store (area, original_index, image) tuples

        # Collect images with their original indices
        for image_index in range(len(images_mfd_res)):
            mfd_res = images_mfd_res[image_index]
            image = images[image_index]
            formula_list = []

            for idx, (xyxy, conf, cla) in enumerate(
                zip(mfd_res.boxes.xyxy, mfd_res.boxes.conf, mfd_res.boxes.cls)
            ):
                if not interline_enable and cla.item() == 1:
                    continue  # Skip interline regions if not enabled
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                new_item = {
                    "category_id": 13 + int(cla.item()),
                    "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    "score": round(float(conf.item()), 2),
                    "latex": "",
                }
                formula_list.append(new_item)
                bbox_img = image[ymin:ymax, xmin:xmax]
                area = (xmax - xmin) * (ymax - ymin)

                curr_idx = len(mf_image_list)
                image_info.append((area, curr_idx, bbox_img))
                mf_image_list.append(bbox_img)

            images_formula_list.append(formula_list)
            backfill_list += formula_list

        # Stable sort by area
        image_info.sort(key=lambda x: x[0])  # sort by area
        sorted_indices = [x[1] for x in image_info]
        sorted_images = [x[2] for x in image_info]

        # Create mapping for results
        index_mapping = {
            new_idx: old_idx for new_idx, old_idx in enumerate(sorted_indices)
        }

        if len(sorted_images) > 0:
            # make predictions
            batch_size = min(batch_size, max(1, 2 ** (len(sorted_images).bit_length() - 1))) if sorted_images else 1
            rec_formula = self.predict(sorted_images, batch_size)
        else:
            rec_formula = []

        # Restore original order
        unsorted_results = [""] * len(rec_formula)
        for new_idx, latex in enumerate(rec_formula):
            original_idx = index_mapping[new_idx]
            unsorted_results[original_idx] = latex

        for res, latex in zip(backfill_list, unsorted_results):
            res["latex"] = latex

        return images_formula_list
