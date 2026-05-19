# Copyright (c) Opendatalab. All rights reserved.
import collections
import math
import re
import statistics

import cv2
import numpy as np
from loguru import logger

from mineru.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio, calculate_iou, \
    get_minbox_if_overlap_by_ratio
from mineru.utils.enum_class import BlockType, ContentType
from mineru.utils.pdf_image_tools import get_crop_img
from mineru.utils.pdf_text_tool import get_page


def remove_outside_spans(spans, all_bboxes, all_discarded_blocks):
    def get_block_bboxes(blocks, block_type_list):
        return [block[0:4] for block in blocks if block[7] in block_type_list]

    image_bboxes = get_block_bboxes(all_bboxes, [BlockType.IMAGE_BODY])
    table_bboxes = get_block_bboxes(all_bboxes, [BlockType.TABLE_BODY])
    other_block_type = []
    for block_type in BlockType.__dict__.values():
        if not isinstance(block_type, str):
            continue
        if block_type not in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY]:
            other_block_type.append(block_type)
    other_block_bboxes = get_block_bboxes(all_bboxes, other_block_type)
    discarded_block_bboxes = get_block_bboxes(all_discarded_blocks, [BlockType.DISCARDED])

    new_spans = []

    for span in spans:
        span_bbox = span['bbox']
        span_type = span['type']

        if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.4 for block_bbox in
               discarded_block_bboxes):
            new_spans.append(span)
            continue

        if span_type == ContentType.IMAGE:
            if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.5 for block_bbox in
                   image_bboxes):
                new_spans.append(span)
        elif span_type == ContentType.TABLE:
            if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.5 for block_bbox in
                   table_bboxes):
                new_spans.append(span)
        else:
            if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.5 for block_bbox in
                   other_block_bboxes):
                new_spans.append(span)

    return new_spans


def remove_overlaps_low_confidence_spans(spans):
    dropped_spans = []
    #  Remove overlapping spans with low confidence
    for span1 in spans:
        for span2 in spans:
            if span1 != span2:
                # span1 or span2 None of them should be there dropped_spans middle
                if span1 in dropped_spans or span2 in dropped_spans:
                    continue
                else:
                    if calculate_iou(span1['bbox'], span2['bbox']) > 0.9:
                        if span1['score'] < span2['score']:
                            span_need_remove = span1
                        else:
                            span_need_remove = span2
                        if (
                            span_need_remove is not None
                            and span_need_remove not in dropped_spans
                        ):
                            dropped_spans.append(span_need_remove)

    if len(dropped_spans) > 0:
        for span_need_remove in dropped_spans:
            spans.remove(span_need_remove)

    return spans, dropped_spans


def remove_overlaps_min_spans(spans):
    dropped_spans = []
    #  Remove the smaller ones of overlapping spans
    for span1 in spans:
        for span2 in spans:
            if span1 != span2:
                # span1 or span2 None of them should be there dropped_spans middle
                if span1 in dropped_spans or span2 in dropped_spans:
                    continue
                else:
                    overlap_box = get_minbox_if_overlap_by_ratio(span1['bbox'], span2['bbox'], 0.65)
                    if overlap_box is not None:
                        span_need_remove = next((span for span in spans if span['bbox'] == overlap_box), None)
                        if span_need_remove is not None and span_need_remove not in dropped_spans:
                            dropped_spans.append(span_need_remove)
    if len(dropped_spans) > 0:
        for span_need_remove in dropped_spans:
            spans.remove(span_need_remove)

    return spans, dropped_spans


def __replace_ligatures(text: str):
    ligatures = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬅ': 'ft', 'ﬆ': 'st'
    }
    return re.sub('|'.join(map(re.escape, ligatures.keys())), lambda m: ligatures[m.group()], text)

def __replace_unicode(text: str):
    ligatures = {
        '\r\n': '', '\u0002': '-',
    }
    return re.sub('|'.join(map(re.escape, ligatures.keys())), lambda m: ligatures[m.group()], text)


"""pdf_text dictplan charlevel"""
def txt_spans_extract(pdf_page, spans, pil_img, scale, all_bboxes, all_discarded_blocks):

    page_dict = get_page(pdf_page)

    page_all_chars = []
    page_all_lines = []
    for block in page_dict['blocks']:
        for line in block['lines']:
            rotation_degrees = math.degrees(line['rotation'])
            # The rotation angle is not 0, 90, 180, 270line, skip directly (the value of rotation_degrees may not be an integer)
            if not any(abs(rotation_degrees - angle) < 0.1 for angle in [0, 90, 180, 270]):
                continue
            page_all_lines.append(line)
            for span in line['spans']:
                for char in span['chars']:
                    page_all_chars.append(char)

    # Calculate the median height of all sapn
    span_height_list = []
    for span in spans:
        if span['type'] in [ContentType.TEXT]:
            span_height = span['bbox'][3] - span['bbox'][1]
            span['height'] = span_height
            span['width'] = span['bbox'][2] - span['bbox'][0]
            span_height_list.append(span_height)
    if len(span_height_list) == 0:
        return spans
    else:
        median_span_height = statistics.median(span_height_list)

    useful_spans = []
    unuseful_spans = []
    # Two characteristics of vertical span: 1. Height exceeds multiple lines 2. Aspect ratio exceeds a certain value
    vertical_spans = []
    for span in spans:
        if span['type'] in [ContentType.TEXT]:
            for block in all_bboxes + all_discarded_blocks:
                if block[7] in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.INTERLINE_EQUATION]:
                    continue
                if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block[0:4]) > 0.5:
                    if span['height'] > median_span_height * 2.3 and span['height'] > span['width'] * 2.3:
                        vertical_spans.append(span)
                    elif block in all_bboxes:
                        useful_spans.append(span)
                    else:
                        unuseful_spans.append(span)
                    break

    """The vertical span box is filled directly with line"""
    if len(vertical_spans) > 0:
        for pdfium_line in page_all_lines:
            for span in vertical_spans:
                if calculate_overlap_area_in_bbox1_area_ratio(pdfium_line['bbox'].bbox, span['bbox']) > 0.5:
                    for pdfium_span in pdfium_line['spans']:
                        span['content'] += pdfium_span['text']
                    break

        for span in vertical_spans:
            if len(span['content']) == 0:
                spans.remove(span)

    """The horizontal span box is first filled with char, and then the empty span box is filled with ocr."""
    new_spans = []

    for span in useful_spans + unuseful_spans:
        if span['type'] in [ContentType.TEXT]:
            span['chars'] = []
            new_spans.append(span)

    need_ocr_spans = fill_char_in_spans(new_spans, page_all_chars, median_span_height)

    """do ocr on unpopulated span"""
    if len(need_ocr_spans) > 0:

        for span in need_ocr_spans:
            # Take a screenshot of span's bbox and then ocr
            span_pil_img = get_crop_img(span['bbox'], pil_img, scale)
            span_img = cv2.cvtColor(np.array(span_pil_img), cv2.COLOR_RGB2BGR)
            # Calculate the contrast of span, below 0.20span does not perform ocr
            if calculate_contrast(span_img, img_mode='bgr') <= 0.17:
                spans.remove(span)
                continue

            span['content'] = ''
            span['score'] = 1.0
            span['np_img'] = span_img

    return spans


def fill_char_in_spans(spans, all_chars, median_span_height):
    # Simply sort from top to bottom
    spans = sorted(spans, key=lambda x: x['bbox'][1])

    grid_size = median_span_height
    grid = collections.defaultdict(list)
    for i, span in enumerate(spans):
        start_cell = int(span['bbox'][1] / grid_size)
        end_cell = int(span['bbox'][3] / grid_size)
        for cell_idx in range(start_cell, end_cell + 1):
            grid[cell_idx].append(i)

    for char in all_chars:
        char_center_y = (char['bbox'][1] + char['bbox'][3]) / 2
        cell_idx = int(char_center_y / grid_size)

        candidate_span_indices = grid.get(cell_idx, [])

        for span_idx in candidate_span_indices:
            span = spans[span_idx]
            if calculate_char_in_span(char['bbox'], span['bbox'], char['char']):
                span['chars'].append(char)
                break

    need_ocr_spans = []
    for span in spans:
        chars_to_content(span)
        # Although there are no words in some spans, there are one or two empty placeholders. Filter them by width, height and content length.
        if len(span['content']) * span['height'] < span['width'] * 0.5:
            # logger.info(f"maybe empty span: {len(span['content'])}, {span['height']}, {span['width']}")
            need_ocr_spans.append(span)
        del span['height'], span['width']
    return need_ocr_spans


LINE_STOP_FLAG = ('.', '!', '?', '。', '！', '？', ')', '）', '"', '”', ':', '：', ';', '；', ']', '】', '}', '}', '>', '》', '、', ',', '，', '-', '—', '–',)
LINE_START_FLAG = ('(', '（', '"', '“', '【', '{', '《', '<', '「', '『', '【', '[',)

Span_Height_Radio = 0.33  # The height difference between the central axis of the character and the central axis of the span cannot exceed 1/3 of the span height.
def calculate_char_in_span(char_bbox, span_bbox, char, span_height_radio=Span_Height_Radio):
    char_center_x = (char_bbox[0] + char_bbox[2]) / 2
    char_center_y = (char_bbox[1] + char_bbox[3]) / 2
    span_center_y = (span_bbox[1] + span_bbox[3]) / 2
    span_height = span_bbox[3] - span_bbox[1]

    if (
        span_bbox[0] < char_center_x < span_bbox[2]
        and span_bbox[1] < char_center_y < span_bbox[3]
        and abs(char_center_y - span_center_y) < span_height * span_height_radio  # The height difference between the character's central axis and the span's central axis cannot exceed Span_Height_Radio
    ):
        return True
    else:
        # If char is LINE_STOP_FLAG, there is no need to determine the center point, and another solution is used (the left boundary is within the span area, and the height determination is consistent with the previous logic)
        # The main purpose is to give the ending symbol a chance to enter the span. This char should also be closer to the right boundary of the span.
        if char in LINE_STOP_FLAG:
            if (
                (span_bbox[2] - span_height) < char_bbox[0] < span_bbox[2]
                and char_center_x > span_bbox[0]
                and span_bbox[1] < char_center_y < span_bbox[3]
                and abs(char_center_y - span_center_y) < span_height * span_height_radio
            ):
                return True
        elif char in LINE_START_FLAG:
            if (
                span_bbox[0] < char_bbox[2] < (span_bbox[0] + span_height)
                and char_center_x < span_bbox[2]
                and span_bbox[1] < char_center_y < span_bbox[3]
                and abs(char_center_y - span_center_y) < span_height * span_height_radio
            ):
                return True
        else:
            return False


def chars_to_content(span):
    # Check if the char in span is empty
    if len(span['chars']) == 0:
        pass
    else:
        # Sort chars by char_idx
        span['chars'] = sorted(span['chars'], key=lambda x: x['char_idx'])

        # Calculate the width of each character
        char_widths = [char['bbox'][2] - char['bbox'][0] for char in span['chars']]
        # Calculate the median width
        median_width = statistics.median(char_widths)

        content = ''
        for char in span['chars']:

            # If the distance between x0 of the next char and x1 of the previous char exceeds 0.25characters wide, you need to insert a space in the middle
            char1 = char
            char2 = span['chars'][span['chars'].index(char) + 1] if span['chars'].index(char) + 1 < len(span['chars']) else None
            if char2 and char2['bbox'][0] - char1['bbox'][2] > median_width * 0.25 and char['char'] != ' ' and char2['char'] != ' ':
                content += f"{char['char']} "
            else:
                content += char['char']

        content = __replace_unicode(content)
        content = __replace_ligatures(content)
        content = __replace_ligatures(content)
        span['content'] = content.strip()

    del span['chars']


def calculate_contrast(img, img_mode) -> float:
    """
    Calculate the contrast of a given image.
    :param img: Image, type numpy.ndarray
    :Param img_mode = the color channels of the image,'rgb' or 'bgr'
    :return: image contrast value
    """
    if img_mode == 'rgb':
        # Convert RGB image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img_mode == 'bgr':
        # Convert BGR image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Invalid image mode. Please provide 'rgb' or 'bgr'.")

    # Calculate mean and standard deviation
    mean_value = np.mean(gray_img)
    std_dev = np.std(gray_img)
    # Contrast is defined as the standard deviation divided by the mean (plus a small constant to avoid divide-by-zero errors)
    contrast = std_dev / (mean_value + 1e-6)
    # logger.debug(f"contrast: {contrast}")
    return round(contrast, 2)