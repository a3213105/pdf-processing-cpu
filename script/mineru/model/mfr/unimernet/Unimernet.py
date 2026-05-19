import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mineru.utils.boxbase import calculate_iou
import openvino as ov
# from .unimernet_hf import UnimernetModel_ov
from mineru.model.ov_operator_async import UnimernetEncDecModelWrapper
from .unimernet_hf.unimer_swin import UnimerSwinImageProcessor
from .unimernet_hf.modeling_unimernet import TokenizerWrapper
from transformers import AutoTokenizer
import os
import glob
class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        raw_image = self.image_paths[idx]
        if self.transform:
            image = self.transform(raw_image)
            return image

class UnimernetModel(object):
    def __init__(self, weight_dir, enable_ov, enc_type, dec_type, _device_="cpu"):
        self.torch_weight = weight_dir
        self.enc_infer_type = enc_type
        self.dec_infer_type = dec_type
        self.transform = UnimerSwinImageProcessor()
        self.tokenizer = TokenizerWrapper(AutoTokenizer.from_pretrained(self.torch_weight))
        # self.ov_model: UnimernetModel_ov = UnimernetModel_ov(None, weight_dir, enc_type, dec_type)
        self.ov_model = UnimernetEncDecModelWrapper(None, self.torch_weight, self.enc_infer_type, self.dec_infer_type)
        if enable_ov and self.ov_model.using_ov :
            self.enable_ov = True
            return 
        else :
            self.enable_ov = False

        if not self.enable_ov :
            from .unimernet_hf import UnimernetModel
            if _device_.startswith("mps") or _device_.startswith("npu") or _device_.startswith("musa"):
                model = UnimernetModel.from_pretrained(weight_dir, attn_implementation="eager")
            else:
                model = UnimernetModel.from_pretrained(weight_dir)
            self.device = torch.device(_device_)
            model.to(self.device)
            if not _device_.startswith("cpu"):
                model = model.to(dtype=torch.float16)
            model.eval()
            self.torch_model = model
            if self.ov_model.converted_to_ov:
                inputs = torch.randn(torch.Size([16, 1, 192, 672]))
                from mineru.model.ov_model_helper import UnimernetConverterWrapper
                converter = UnimernetConverterWrapper(self.torch_model, self.torch_weight)
                converter.convert_ov_model(inputs)

    def remove_unused_weight(self) :
        return
        if self.enable_ov and self.ov_model is not None:
            for file_path in glob.glob(os.path.join(self.torch_weight, "*")):
                filename = os.path.basename(file_path)
                if filename.endswith(".safetensors"):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Failed to remove {file_path}: {e}")

    @staticmethod
    def _filter_boxes_by_iou(xyxy, conf, cla, iou_threshold=0.8):
        """Filter overlapping boxes whose IOU exceeds the threshold and retain boxes with higher confidence.

        Args:
            xyxy: Box coordinate tensor, shape is (N, 4)
            conf: Confidence tensor, shape is (N,)
            cla: Category tensor, shape is (N,)
            iou_threshold: IOUThreshold, default 0.9

        Returns:
            Filtered xyxy, conf, claTensor
        """
        if len(xyxy) == 0:
            return xyxy, conf, cla

        # Converted to CPU for processing
        xyxy_cpu = xyxy.cpu()
        conf_cpu = conf.cpu()

        n = len(xyxy_cpu)
        keep = [True] * n

        for i in range(n):
            if not keep[i]:
                continue
            bbox1 = xyxy_cpu[i].tolist()
            for j in range(i + 1, n):
                if not keep[j]:
                    continue
                bbox2 = xyxy_cpu[j].tolist()
                iou = calculate_iou(bbox1, bbox2)
                if iou > iou_threshold:
                    # Keep boxes with higher confidence
                    if conf_cpu[i] >= conf_cpu[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break  # iis deleted and jumps out of the inner loop

        keep_indices = [i for i in range(n) if keep[i]]
        if len(keep_indices) == n:
            return xyxy, conf, cla

        keep_indices = torch.tensor(keep_indices, dtype=torch.long)
        return xyxy[keep_indices], conf[keep_indices], cla[keep_indices]

    def predict(self, mfd_res, image):
        formula_list = []
        mf_image_list = []

        # Perform IOU deduplication on the detection frame and retain the frame with higher confidence.
        xyxy_filtered, conf_filtered, cla_filtered = self._filter_boxes_by_iou(
            mfd_res.boxes.xyxy, mfd_res.boxes.conf, mfd_res.boxes.cls
        )

        for xyxy, conf, cla in zip(
            xyxy_filtered.cpu(), conf_filtered.cpu(), cla_filtered.cpu()
        ):
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            new_item = {
                "category_id": 13 + int(cla.item()),
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf.item()), 2),
                "latex": "",
            }
            formula_list.append(new_item)
            bbox_img = image[ymin:ymax, xmin:xmax]
            mf_image_list.append(bbox_img)

        dataset = MathDataset(mf_image_list, transform=self.torch_model.transform)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=0)
        mfr_res = []
        for mf_img in dataloader:
            mf_img = mf_img.to(dtype=self.torch_model.dtype)
            mf_img = mf_img.to(self.device)
            with torch.no_grad():
                output = self.torch_model.generate({"image": mf_img})
            mfr_res.extend(output["fixed_str"])
        for res, latex in zip(formula_list, mfr_res):
            res["latex"] = latex
        return formula_list

    def batch_predict(
            self,
            images_mfd_res: list,
            images: list,
            batch_size: int = 64,
            interline_enable: bool = True,
            tqdm_enable: bool = False
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

            # Perform IOU deduplication on the detection frame and retain the frame with higher confidence.
            xyxy_filtered, conf_filtered, cla_filtered = self._filter_boxes_by_iou(
                mfd_res.boxes.xyxy, mfd_res.boxes.conf, mfd_res.boxes.cls
            )

            for idx, (xyxy, conf, cla) in enumerate(zip(
                    xyxy_filtered, conf_filtered, cla_filtered
            )):
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
        index_mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(sorted_indices)}


        # if batch_size > len(sorted_images)，Then set it to a power of 2 not exceeding len(sorted_images)
        batch_size = min(batch_size, max(1, 2 ** (len(sorted_images).bit_length() - 1))) if sorted_images else 1
        # Process batches and store results
        mfr_res = []
        # for mf_img in dataloader:

        if self.enable_ov :
            # mfr_res = self.ov_model.inference(sorted_images, batch_size)
            mfr_res = []
            desc_str = f"MFR Predict with OV_{self.enc_infer_type}_{self.dec_infer_type}"
            for mf_img in tqdm(sorted_images, desc=desc_str, disable=not tqdm_enable):
                mf_img = self.transform(mf_img).unsqueeze(0)
                outputs = self.ov_model.generate(mf_img)
                mfr_res.extend(outputs)
            pred_str = self.tokenizer.token2str(mfr_res)
            from ..utils import latex_rm_whitespace
            mfr_res = [latex_rm_whitespace(s) for s in pred_str]
        else :
            # Create dataset with sorted images
            dataset = MathDataset(sorted_images, transform=self.torch_model.transform)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
            tqdm_desc = "MFR Predict"
            with tqdm(total=len(sorted_images), desc=tqdm_desc, disable=not tqdm_enable) as pbar:
                for index, mf_img in enumerate(dataloader):
                    mf_img = mf_img.to(dtype=self.torch_model.dtype)
                    mf_img = mf_img.to(self.device)
                    with torch.no_grad():
                        output = self.torch_model.generate({"image": mf_img}, batch_size=batch_size)
                        # if self.ov_model.converted_to_ov:
                        #     self.ov_model.convert_ov_model(self.torch_model, mf_img)

                    mfr_res.extend(output["fixed_str"])

                    # Update the progress bar and increase batch_size each time, but be aware that the last batch may be less than batch_size
                    current_batch_size = min(batch_size, len(sorted_images) - index * batch_size)
                    pbar.update(current_batch_size)

        # Restore original order
        unsorted_results = [""] * len(mfr_res)
        for new_idx, latex in enumerate(mfr_res):
            original_idx = index_mapping[new_idx]
            unsorted_results[original_idx] = latex

        # Fill results back
        for res, latex in zip(backfill_list, unsorted_results):
            res["latex"] = latex

        return images_formula_list
