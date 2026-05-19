# Copyright (c) Opendatalab. All rights reserved.
from mineru.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from mineru.utils.enum_class import BlockType, ContentType
from mineru.utils.ocr_utils import _is_overlaps_y_exceeds_threshold, _is_overlaps_x_exceeds_threshold

VERTICAL_SPAN_HEIGHT_TO_WIDTH_RATIO_THRESHOLD = 2
VERTICAL_SPAN_IN_BLOCK_THRESHOLD = 0.8

def fill_spans_in_blocks(blocks, spans, radio):
    """Put the spans in allspans into blocks according to their positional relationship."""
    block_with_spans = []
    for block in blocks:
        block_type = block[7]
        block_bbox = block[0:4]
        block_dict = {
            'type': block_type,
            'bbox': block_bbox,
        }
        if block_type in [
            BlockType.IMAGE_BODY, BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE,
            BlockType.TABLE_BODY, BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE
        ]:
            block_dict['group_id'] = block[-1]
        block_spans = []
        for span in spans:
            temp_radio = radio
            span_bbox = span['bbox']
            if span['type'] in [ContentType.IMAGE, ContentType.TABLE]:
                temp_radio = 0.9
            if calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > temp_radio and span_block_type_compatible(span['type'], block_type):
                block_spans.append(span)

        block_dict['spans'] = block_spans
        block_with_spans.append(block_dict)

        # Delete spans that have been placed in block_spans from spans
        if len(block_spans) > 0:
            for span in block_spans:
                spans.remove(span)

    return block_with_spans, spans


def span_block_type_compatible(span_type, block_type):
    if span_type in [ContentType.TEXT, ContentType.INLINE_EQUATION]:
        return block_type in [
            BlockType.TEXT,
            BlockType.TITLE,
            BlockType.IMAGE_CAPTION,
            BlockType.IMAGE_FOOTNOTE,
            BlockType.TABLE_CAPTION,
            BlockType.TABLE_FOOTNOTE,
            BlockType.DISCARDED
        ]
    elif span_type == ContentType.INTERLINE_EQUATION:
        return block_type in [BlockType.INTERLINE_EQUATION, BlockType.TEXT]
    elif span_type == ContentType.IMAGE:
        return block_type in [BlockType.IMAGE_BODY]
    elif span_type == ContentType.TABLE:
        return block_type in [BlockType.TABLE_BODY]
    else:
        return False


def fix_discarded_block(discarded_block_with_spans):
    fix_discarded_blocks = []
    for block in discarded_block_with_spans:
        block = fix_text_block(block)
        fix_discarded_blocks.append(block)
    return fix_discarded_blocks


def fix_text_block(block):
    # Formula spans in text blocks should be converted to inline types
    for span in block['spans']:
        if span['type'] == ContentType.INTERLINE_EQUATION:
            span['type'] = ContentType.INLINE_EQUATION

    # Assuming that more than 80% of the spans in the block have a height that is more than twice the width, it is considered a vertical text block.
    vertical_span_count = sum(
        1 for span in block['spans']
        if (span['bbox'][3] - span['bbox'][1]) / (span['bbox'][2] - span['bbox'][0]) > VERTICAL_SPAN_HEIGHT_TO_WIDTH_RATIO_THRESHOLD
    )
    total_span_count = len(block['spans'])
    if total_span_count == 0:
        vertical_ratio = 0
    else:
        vertical_ratio = vertical_span_count / total_span_count

    if vertical_ratio > VERTICAL_SPAN_IN_BLOCK_THRESHOLD:
        # If it is a vertical text block, it will be processed as vertical lines.
        block_lines = merge_spans_to_vertical_line(block['spans'])
        sort_block_lines = vertical_line_sort_spans_from_top_to_bottom(block_lines)
    else:
        block_lines = merge_spans_to_line(block['spans'])
        sort_block_lines = line_sort_spans_by_left_to_right(block_lines)

    block['lines'] = sort_block_lines
    del block['spans']
    return block


def merge_spans_to_line(spans, threshold=0.6):
    if len(spans) == 0:
        return []
    else:
        # Sort by y0 coordinate
        spans.sort(key=lambda span: span['bbox'][1])

        lines = []
        current_line = [spans[0]]
        for span in spans[1:]:
            # If the current span type is"interline_equation" or Already in the current row"interline_equation"
            # imageand table type, same as above
            if span['type'] in [
                    ContentType.INTERLINE_EQUATION, ContentType.IMAGE,
                    ContentType.TABLE
            ] or any(s['type'] in [
                    ContentType.INTERLINE_EQUATION, ContentType.IMAGE,
                    ContentType.TABLE
            ] for s in current_line):
                # then start a new line
                lines.append(current_line)
                current_line = [span]
                continue

            # If the current span overlaps the last span of the current row on the y-axis, add to the current row
            if _is_overlaps_y_exceeds_threshold(span['bbox'], current_line[-1]['bbox'], threshold):
                current_line.append(span)
            else:
                # Otherwise, start a new line
                lines.append(current_line)
                current_line = [span]

        # add last line
        if current_line:
            lines.append(current_line)

        return lines


def merge_spans_to_vertical_line(spans, threshold=0.6):
    """Combine vertical text spans into vertical lines (read from right to left)"""
    if len(spans) == 0:
        return []
    else:
        # Sort by x2 coordinate from large to small (right to left)
        spans.sort(key=lambda span: span['bbox'][2], reverse=True)

        vertical_lines = []
        current_line = [spans[0]]

        for span in spans[1:]:
            # Special types of elements are listed separately
            if span['type'] in [
                ContentType.INTERLINE_EQUATION, ContentType.IMAGE,
                ContentType.TABLE
            ] or any(s['type'] in [
                ContentType.INTERLINE_EQUATION, ContentType.IMAGE,
                ContentType.TABLE
            ] for s in current_line):
                vertical_lines.append(current_line)
                current_line = [span]
                continue

            # If the current span overlaps the last span of the current row on the y-axis, add to the current row
            if _is_overlaps_x_exceeds_threshold(span['bbox'], current_line[-1]['bbox'], threshold):
                current_line.append(span)
            else:
                vertical_lines.append(current_line)
                current_line = [span]

        # add last column
        if current_line:
            vertical_lines.append(current_line)

        return vertical_lines


# Sort the spans in each line from left to right
def line_sort_spans_by_left_to_right(lines):
    line_objects = []
    for line in lines:
        #  Sort by x0 coordinate
        line.sort(key=lambda span: span['bbox'][0])
        line_bbox = [
            min(span['bbox'][0] for span in line),  # x0
            min(span['bbox'][1] for span in line),  # y0
            max(span['bbox'][2] for span in line),  # x1
            max(span['bbox'][3] for span in line),  # y1
        ]
        line_objects.append({
            'bbox': line_bbox,
            'spans': line,
        })
    return line_objects


def vertical_line_sort_spans_from_top_to_bottom(vertical_lines):
    line_objects = []
    for line in vertical_lines:
        # Sort by y0 coordinate (from top to bottom)
        line.sort(key=lambda span: span['bbox'][1])

        # Calculate the bounding box of the entire column
        line_bbox = [
            min(span['bbox'][0] for span in line),  # x0
            min(span['bbox'][1] for span in line),  # y0
            max(span['bbox'][2] for span in line),  # x1
            max(span['bbox'][3] for span in line),  # y1
        ]

        # Assembly result
        line_objects.append({
            'bbox': line_bbox,
            'spans': line,
        })
    return line_objects


def fix_block_spans(block_with_spans):
    fix_blocks = []
    for block in block_with_spans:
        block_type = block['type']

        if block_type in [BlockType.TEXT, BlockType.TITLE,
                          BlockType.IMAGE_CAPTION, BlockType.IMAGE_CAPTION,
                          BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE
                          ]:
            block = fix_text_block(block)
        elif block_type in [BlockType.INTERLINE_EQUATION, BlockType.IMAGE_BODY, BlockType.TABLE_BODY]:
            block = fix_interline_block(block)
        else:
            continue
        fix_blocks.append(block)
    return fix_blocks


def fix_interline_block(block):
    block_lines = merge_spans_to_line(block['spans'])
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    block['lines'] = sort_block_lines
    del block['spans']
    return block