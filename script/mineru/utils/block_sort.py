# Copyright (c) Opendatalab. All rights reserved.
import os
import statistics
import warnings
from typing import List, OrderedDict
import torch
from loguru import logger

from mineru.utils.config_reader import get_device
from mineru.utils.enum_class import BlockType, ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from mineru.model.ov_operator_async import LayoutLMv3ClsProcessor


def sort_blocks_by_bbox(blocks, page_w, page_h, footnote_blocks, model_manager):

    """Get all lines and calculate the height of the text line"""
    line_height = get_line_height(blocks)

    """Get all lines and sort the lines"""
    sorted_bboxes = sort_lines_by_model(blocks, page_w, page_h, line_height, footnote_blocks, model_manager)

    """Calculate the sequence relationship of blocks based on the median of the line"""
    blocks = cal_block_index(blocks, sorted_bboxes)

    """Restore image and table blocks back to group form to participate in subsequent processes"""
    blocks = revert_group_blocks(blocks)

    """Rearrange blocks"""
    sorted_blocks = sorted(blocks, key=lambda b: b['index'])

    """blockInternal rearrangement (sorting of multiple captions or footnotes within img and table blocks)"""
    for block in sorted_blocks:
        if block['type'] in [BlockType.IMAGE, BlockType.TABLE]:
            block['blocks'] = sorted(block['blocks'], key=lambda b: b['index'])

    return sorted_blocks


def get_line_height(blocks):
    page_line_height_list = []
    for block in blocks:
        if block['type'] in [
            BlockType.TEXT, BlockType.TITLE,
            BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE,
            BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE
        ]:
            for line in block['lines']:
                bbox = line['bbox']
                page_line_height_list.append(int(bbox[3] - bbox[1]))
    if len(page_line_height_list) > 0:
        return statistics.median(page_line_height_list)
    else:
        return 10


def sort_lines_by_model(fix_blocks, page_w, page_h, line_height, footnote_blocks, model_manager):
    page_line_list = []

    def add_lines_to_block(b):
        line_bboxes = insert_lines_into_block(b['bbox'], line_height, page_w, page_h)
        b['lines'] = []
        for line_bbox in line_bboxes:
            b['lines'].append({'bbox': line_bbox, 'spans': []})
        page_line_list.extend(line_bboxes)

    for block in fix_blocks:
        if block['type'] in [
            BlockType.TEXT, BlockType.TITLE,
            BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE,
            BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE
        ]:
            if len(block['lines']) == 0:
                add_lines_to_block(block)
            elif block['type'] in [BlockType.TITLE] and len(block['lines']) == 1 and (block['bbox'][3] - block['bbox'][1]) > line_height * 2:
                block['real_lines'] = list(block['lines'])
                add_lines_to_block(block)
            else:
                for line in block['lines']:
                    bbox = line['bbox']
                    page_line_list.append(bbox)
        elif block['type'] in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.INTERLINE_EQUATION]:
            block['real_lines'] = list(block['lines'])
            add_lines_to_block(block)

    for block in footnote_blocks:
        footnote_block = {'bbox': block[:4]}
        add_lines_to_block(footnote_block)

    if len(page_line_list) > 500:  # layoutreaderSupports up to 512line
        return None

    # Sort using layoutreader
    x_scale = 1000.0 / page_w
    y_scale = 1000.0 / page_h
    boxes = []
    # logger.info(f"Scale: {x_scale}, {y_scale}, Boxes len: {len(page_line_list)}")
    for left, top, right, bottom in page_line_list:
        if left < 0:
            logger.warning(
                f'left < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            left = 0
        if right > page_w:
            logger.warning(
                f'right > page_w, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            right = page_w
        if top < 0:
            logger.warning(
                f'top < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            top = 0
        if bottom > page_h:
            logger.warning(
                f'bottom > page_h, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            bottom = page_h

        left = round(left * x_scale)
        top = round(top * y_scale)
        right = round(right * x_scale)
        bottom = round(bottom * y_scale)
        assert (
            1000 >= right >= left >= 0 and 1000 >= bottom >= top >= 0
        ), f'Invalid box. right: {right}, left: {left}, bottom: {bottom}, top: {top}'  # noqa: E126, E121
        boxes.append([left, top, right, bottom])
    model = model_manager.model.get_layout_reader_model('layoutreader')
    with torch.no_grad():
        orders = do_predict(boxes, model)
    sorted_bboxes = [page_line_list[i] for i in orders]
    return sorted_bboxes


def insert_lines_into_block(block_bbox, line_height, page_w, page_h):
    # block_bboxis a tuple (x0, y0, x1, y1)，Among them (x0, y0)is the coordinate of the lower left corner, (x1, y1)is the coordinate of the upper right corner
    x0, y0, x1, y1 = block_bbox

    block_height = y1 - y0
    block_weight = x1 - x0

    # If the block height is less than n lines of text, the bbox of the block is returned directly.
    if line_height * 2 < block_height:
        if (
            block_height > page_h * 0.25 and page_w * 0.5 > block_weight > page_w * 0.25
        ):  # It may be a double-column structure and can be cut into smaller pieces.
            lines = int(block_height / line_height)
        else:
            # If the width of the block exceeds 0.4Page width, divide the block into 3 rows (it is a complex layout, the picture cannot be cut too thin)
            if block_weight > page_w * 0.4:
                lines = 3
            elif block_weight > page_w * 0.25:  # （It may be a three-column structure, but also cut it into details)
                lines = int(block_height / line_height)
            else:  # Determine aspect ratio
                if block_height / block_weight > 1.2:  # Regardless of whether they are slender or slender
                    return [[x0, y0, x1, y1]]
                else:  # If it’s not slender, it should still be divided into two rows.
                    lines = 2

        line_height = (y1 - y0) / lines

        # Determine at which y position to start drawing the line
        current_y = y0

        # Used to store position information of lines[(x0, y), ...]
        lines_positions = []

        for i in range(lines):
            lines_positions.append([x0, current_y, x1, current_y + line_height])
            current_y += line_height
        return lines_positions

    else:
        return [[x0, y0, x1, y1]]


class LayoutReaderModel:
    def __init__(self, model_name: str, enable_ov, infer_type):
        self.device = get_device()
        bf_16_support = False
        if self.device.startswith("cuda"):
            if torch.cuda.get_device_properties(self.device).major >= 8:
                bf_16_support = True
        elif self.device.startswith("mps"):
            bf_16_support = True
        elif self.device.startswith("gcu"):
            if hasattr(torch, 'gcu') and torch.gcu.is_available():
                if torch.gcu.is_bf16_supported():
                    bf_16_support = True
        elif self.device.startswith("musa"):
            if hasattr(torch, 'musa') and torch.musa.is_available():
                if torch.musa.is_bf16_supported():
                    bf_16_support = True
        elif self.device.startswith("npu"):
            if hasattr(torch, 'npu') and torch.npu.is_available():
                if torch.npu.is_bf16_supported():
                    bf_16_support = True
        elif self.device.startswith("mlu"):
            if hasattr(torch, 'mlu') and torch.mlu.is_available():
                if torch.mlu.is_bf16_supported():
                    bf_16_support = True
        elif self.device.startswith("sdaa"):
            if hasattr(torch, 'sdaa') and torch.sdaa.is_available():
                if torch.sdaa.is_bf16_supported():
                    bf_16_support = True  

        if model_name == 'layoutreader':
            # Check whether the cache directory of modelscope exists
            self.layoutreader_model_dir = os.path.join(auto_download_and_get_model_root_path(ModelPath.layout_reader), ModelPath.layout_reader)
            self.ov_file_name = f"{self.layoutreader_model_dir}/layoutreader.xml"
            self.ov_net = None
            self.enable_ov = enable_ov
            self.infer_type = infer_type
            if enable_ov:
                try:
                    self.ov_net = LayoutLMv3ClsProcessor(self.ov_file_name)
                    self.ov_net.setup_model(stream_num=1, infer_type=self.infer_type)
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenVINO model: {e}")
                    self.ov_net = None
            
            if self.ov_net is None:
                from transformers import LayoutLMv3ForTokenClassification
                if os.path.exists(self.layoutreader_model_dir):
                    self.model = LayoutLMv3ForTokenClassification.from_pretrained(self.layoutreader_model_dir)
                else:
                    logger.warning('local layoutreader model not exists, use online model from huggingface')
                    self.model = LayoutLMv3ForTokenClassification.from_pretrained('hantian/layoutreader')
                if self.enable_ov:
                    bbox = torch.tensor([[[  0,   0,   0,   0], [15, 15, 22, 22], [55, 88, 92, 92],
                                          [105, 185, 222, 202], [305, 385, 422, 402], [505, 585, 622, 602],
                                          [705, 785, 822, 802], [  0,   0,   0,   0]]])
                    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1,]])
                    input_ids = torch.tensor([[0, 3, 3, 3, 3, 3, 3, 2]])
                    from mineru.model.ov_model_helper import LayoutreaderConverter
                    converter = LayoutreaderConverter(self.model)
                    example_inputs = {'input_ids': input_ids, 'bbox': bbox, 'attention_mask': attention_mask,}
                    converter.convert_model(xml_path=self.ov_file_name, example_inputs=example_inputs)
                if bf_16_support:
                    self.model.to(self.device).eval().bfloat16()
                else:
                    self.model.to(self.device).eval()
        else:
            logger.error('model name not allow')
            exit(1)

    def remove_unused_weight(self):
        model_file = f"{self.layoutreader_model_dir}/pytorch_model.bin"
        if self.ov_net is not None and os.path.exists(model_file):
            try:
                os.remove(model_file)
            except Exception as e:
                logger.warning(f"LayoutReader Failed to remove layoutreader model dir: {e}")

    def __call__(self, length, **xargs):
        if self.ov_net is not None:
            logits = self.ov_net(xargs)
            logits = logits[1 : length + 1, :length]
            import numpy as np
            orders = np.argsort(logits, axis=-1)
        else :
            logits = self.model(**xargs).logits.cpu().squeeze(0)
            logits = logits[1 : length + 1, :length]
            orders = logits.argsort(descending=False)
        from mineru.model.reading_order.layout_reader import parse_logits_list as parse_logits
        return parse_logits(logits, orders)

def do_predict(boxes: List[List[int]], model) -> List[int]:
    from mineru.model.reading_order.layout_reader import boxes2inputs, prepare_inputs
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
        inputs = boxes2inputs(boxes)
        inputs = prepare_inputs(inputs, model)
        return model(len(boxes), **inputs)

def cal_block_index(fix_blocks, sorted_bboxes):
    if sorted_bboxes is not None:
        # Sort using layoutreader
        for block in fix_blocks:
            line_index_list = []
            if len(block['lines']) == 0:
                block['index'] = sorted_bboxes.index(block['bbox'])
            else:
                for line in block['lines']:
                    line['index'] = sorted_bboxes.index(line['bbox'])
                    line_index_list.append(line['index'])
                median_value = statistics.median(line_index_list)
                block['index'] = median_value

            # Delete chart body blockVirtual line information in, And backfill with real_lines information
            if block['type'] in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.TITLE, BlockType.INTERLINE_EQUATION]:
                if 'real_lines' in block:
                    block['virtual_lines'] = list(block['lines'])
                    block['lines'] = list(block['real_lines'])
                    del block['real_lines']
    else:
        # Sort using xycut
        block_bboxes = []
        for block in fix_blocks:
            # if block['bbox']Any value less than 0 is set to 0
            block['bbox'] = [max(0, x) for x in block['bbox']]
            block_bboxes.append(block['bbox'])

            # Delete chart body blockVirtual line information in, And backfill with real_lines information
            if block['type'] in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.TITLE, BlockType.INTERLINE_EQUATION]:
                if 'real_lines' in block:
                    block['virtual_lines'] = list(block['lines'])
                    block['lines'] = list(block['real_lines'])
                    del block['real_lines']

        import numpy as np
        from mineru.model.reading_order.xycut import recursive_xy_cut

        random_boxes = np.array(block_bboxes)
        np.random.shuffle(random_boxes)
        res = []
        recursive_xy_cut(np.asarray(random_boxes).astype(int), np.arange(len(block_bboxes)), res)
        assert len(res) == len(block_bboxes)
        sorted_boxes = random_boxes[np.array(res)].tolist()

        for i, block in enumerate(fix_blocks):
            block['index'] = sorted_boxes.index(block['bbox'])

        # Generate line index
        sorted_blocks = sorted(fix_blocks, key=lambda b: b['index'])
        line_inedx = 1
        for block in sorted_blocks:
            for line in block['lines']:
                line['index'] = line_inedx
                line_inedx += 1

    return fix_blocks


def revert_group_blocks(blocks):
    image_groups = {}
    table_groups = {}
    new_blocks = []
    for block in blocks:
        if block['type'] in [BlockType.IMAGE_BODY, BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE]:
            group_id = block['group_id']
            if group_id not in image_groups:
                image_groups[group_id] = []
            image_groups[group_id].append(block)
        elif block['type'] in [BlockType.TABLE_BODY, BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE]:
            group_id = block['group_id']
            if group_id not in table_groups:
                table_groups[group_id] = []
            table_groups[group_id].append(block)
        else:
            new_blocks.append(block)

    for group_id, blocks in image_groups.items():
        new_blocks.append(process_block_list(blocks, BlockType.IMAGE_BODY, BlockType.IMAGE))

    for group_id, blocks in table_groups.items():
        new_blocks.append(process_block_list(blocks, BlockType.TABLE_BODY, BlockType.TABLE))

    return new_blocks


def process_block_list(blocks, body_type, block_type):
    indices = [block['index'] for block in blocks]
    median_index = statistics.median(indices)

    body_bbox = next((block['bbox'] for block in blocks if block.get('type') == body_type), [])

    return {
        'type': block_type,
        'bbox': body_bbox,
        'blocks': blocks,
        'index': median_index,
    }