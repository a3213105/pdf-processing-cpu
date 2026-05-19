# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import json
from mineru.utils.os_env_config import get_op_num_threads
from .table_structure_utils import (
    OrtInferSession,
    TableLabelDecode,
    TablePreprocess,
    BatchTablePreprocess,
)
from mineru.model.ov_operator_async import OnnxSessProcessor

class TableStructurer:
    def __init__(self, enable_ov, wireless_table_type, config: Dict[str, Any]):
        self.preprocess_op = TablePreprocess()
        self.batch_preprocess_op = BatchTablePreprocess()
        self.enable_ov = enable_ov
        self.infer_type = wireless_table_type
        self.session = None
        json_path = config["model_path"] + ".json"
        if self.enable_ov:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self.character = json.load(f)
                self.table_model_ov = OnnxSessProcessor(config["model_path"], "TableStructurer")
                self.table_model_ov.setup_model(stream_num = 1, infer_type=self.infer_type)
                def ov_infer(*args):
                    result = self.table_model_ov(args[0])
                    return result[0], result[1]
                self.session = ov_infer
                # self.character = ['<thead>', '</thead>', '<tbody>', '</tbody>', '<tr>', '</tr>', '<td', '>', '</td>', ' colspan="2"', ' colspan="3"', ' colspan="4"', ' colspan="5"', ' colspan="6"', ' colspan="7"', ' colspan="8"', ' colspan="9"', ' colspan="10"', ' colspan="11"', ' colspan="12"', ' colspan="13"', ' colspan="14"', ' colspan="15"', ' colspan="16"', ' colspan="17"', ' colspan="18"', ' colspan="19"', ' colspan="20"', ' rowspan="2"', ' rowspan="3"', ' rowspan="4"', ' rowspan="5"', ' rowspan="6"', ' rowspan="7"', ' rowspan="8"', ' rowspan="9"', ' rowspan="10"', ' rowspan="11"', ' rowspan="12"', ' rowspan="13"', ' rowspan="14"', ' rowspan="15"', ' rowspan="16"', ' rowspan="17"', ' rowspan="18"', ' rowspan="19"', ' rowspan="20"', '<td></td>']
            except Exception as e:
                print(f"### TableStructurer ov model failed, {e}")
                self.session = None
        if self.session is None:
            config["intra_op_num_threads"] = get_op_num_threads("MINERU_INTRA_OP_NUM_THREADS")
            config["inter_op_num_threads"] = get_op_num_threads("MINERU_INTER_OP_NUM_THREADS")
            self.session = OrtInferSession(config)
            self.character = self.session.get_metadata()
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.character, f, ensure_ascii=False, indent=2)
        self.postprocess_op = TableLabelDecode(self.character)

    def remove_unused_weight(self) :
        pass

    def process(self, img):
        starttime = time.time()
        data = {"image": img}
        data = self.preprocess_op(data)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()

        outputs = self.session([img])

        preds = {"loc_preds": outputs[0], "structure_probs": outputs[1]}

        shape_list = np.expand_dims(data[-1], axis=0)
        post_result = self.postprocess_op(preds, [shape_list])

        bbox_list = post_result["bbox_batch_list"][0]

        structure_str_list = post_result["structure_batch_list"][0]
        structure_str_list = structure_str_list[0]
        structure_str_list = (
            ["<html>", "<body>", "<table>"]
            + structure_str_list
            + ["</table>", "</body>", "</html>"]
        )
        elapse = time.time() - starttime
        return structure_str_list, bbox_list, elapse

    def batch_process(
        self, img_list: List[np.ndarray]
    ) -> List[Tuple[List[str], np.ndarray, float]]:
        """Batch processing of image lists
        Args:
            img_list: image list
        Returns:
            The resulting list, each element containing (table_struct_str, cell_bboxes, elapse)
        """
        starttime = time.perf_counter()

        batch_data = self.batch_preprocess_op(img_list)
        preprocessed_images = batch_data[0]
        shape_lists = batch_data[1]

        preprocessed_images = np.array(preprocessed_images)
        bbox_preds, struct_probs = self.session([preprocessed_images])

        batch_size = preprocessed_images.shape[0]
        results = []
        for bbox_pred, struct_prob, shape_list in zip(
            bbox_preds, struct_probs, shape_lists
        ):
            preds = {
                "loc_preds": np.expand_dims(bbox_pred, axis=0),
                "structure_probs": np.expand_dims(struct_prob, axis=0),
            }
            shape_list = np.expand_dims(shape_list, axis=0)
            post_result = self.postprocess_op(preds, [shape_list])
            bbox_list = post_result["bbox_batch_list"][0]
            structure_str_list = post_result["structure_batch_list"][0]
            structure_str_list = structure_str_list[0]
            structure_str_list = (
                ["<html>", "<body>", "<table>"]
                + structure_str_list
                + ["</table>", "</body>", "</html>"]
            )
            results.append((structure_str_list, bbox_list, 0))

        total_elapse = time.perf_counter() - starttime
        for i in range(len(results)):
            results[i] = (results[i][0], results[i][1], total_elapse / batch_size)

        return results
