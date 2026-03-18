import os
from pathlib import Path
import cv2
import numpy as np
import torch
 
from rapid_table import RapidTable, RapidTableInput
from rapid_table.main import ModelType

from magic_pdf.libs.config_reader import get_device
try :
    from ...ov_operator_async import RapidTableProcesser
except ImportError as e:
    print(f"### import ov_operator_async failed, {e}")


class RapidTableModel(object):
    def __init__(self, model_path, ocr_engine, table_sub_model_name='slanet_plus',
                 enable_ov = True, infer_type = True):
        sub_model_list = [model.value for model in ModelType]
        if table_sub_model_name is None:
            input_args = RapidTableInput()
        elif table_sub_model_name in  sub_model_list:
            if torch.cuda.is_available() and table_sub_model_name == "unitable":
                input_args = RapidTableInput(model_type=table_sub_model_name, use_cuda=True, device=get_device())
            else:
                root_dir = Path(__file__).absolute().parent.parent.parent.parent.parent
                slanet_plus_model_path = os.path.join(model_path, 'slanet_plus', 'slanet-plus.onnx')
                input_args = RapidTableInput(model_type=table_sub_model_name, model_path=slanet_plus_model_path)
        else:
            raise ValueError(f"Invalid table_sub_model_name: {table_sub_model_name}. It must be one of {sub_model_list}")
        self.table_model = RapidTable(input_args)
        self.enable_ov = enable_ov
        self.infer_type = infer_type
        if self.enable_ov:
            self.table_model_ov = RapidTableProcesser(slanet_plus_model_path)
            self.table_model_ov.setup_model(stream_num = 1, infer_type=self.infer_type)
            def ov_infer(*args):
                # print(f"RapidTableModel args={args[0][0].shape}")
                result = self.table_model_ov(args[0])
                return result[0], result[1]
            self.table_model.table_structure.session = ov_infer
            # print(f"### load table_model_ov model {slanet_plus_model_path}, infer_type={self.infer_type}")
        self.ocr_engine = ocr_engine

    def predict(self, image):
        bgr_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        # First check the overall image aspect ratio (height/width)
        img_height, img_width = bgr_image.shape[:2]
        img_aspect_ratio = img_height / img_width if img_width > 0 else 1.0
        img_is_portrait = img_aspect_ratio > 1.2
        
        if img_is_portrait:
            det_res = self.ocr_engine.ocr(bgr_image, rec=False)
            det_res = det_res[0]
            # Check if table is rotated by analyzing text box aspect ratios
            is_rotated = False
            if det_res:
                vertical_count = 0
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

                # If we have more vertical text boxes than horizontal ones,
                # and vertical ones are significant, table might be rotated
                if vertical_count >= len(det_res) * 0.3:
                    is_rotated = True

                # logger.debug(f"Text orientation analysis: vertical={vertical_count}, det_res={len(det_res)}, rotated={is_rotated}")

            # Rotate image if necessary
            if is_rotated:
                # logger.debug("Table appears to be in portrait orientation, rotating 90 degrees clockwise")
                image = cv2.rotate(np.asarray(image), cv2.ROTATE_90_CLOCKWISE)
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Continue with OCR on potentially rotated image
        ocr_result = self.ocr_engine.ocr(bgr_image)[0]
        if ocr_result:
            ocr_result = [[item[0], item[1][0], item[1][1]] for item in ocr_result if
                      len(item) == 2 and isinstance(item[1], tuple)]
        else:
            ocr_result = None

        if ocr_result:
            # print(f"RapidTableModel image={image.shape}, ocr_result={len(ocr_result)}")
            table_results = self.table_model(np.asarray(image), ocr_result)
            html_code = table_results.pred_html
            table_cell_bboxes = table_results.cell_bboxes
            logic_points = table_results.logic_points
            elapse = table_results.elapse
            return html_code, table_cell_bboxes, logic_points, elapse
        else:
            return None, None, None, None
