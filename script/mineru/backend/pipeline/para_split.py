import copy
from loguru import logger
from mineru.utils.enum_class import ContentType, BlockType, SplitFlag
from mineru.utils.language import detect_lang


LINE_STOP_FLAG = ('.', '!', '?', '。', '！', '？', ')', '）', '"', '”', ':', '：', ';', '；')
LIST_END_FLAG = ('.', '。', ';', '；')


class ListLineTag:
    IS_LIST_START_LINE = 'is_list_start_line'
    IS_LIST_END_LINE = 'is_list_end_line'


def __process_blocks(blocks):
    # Preprocess all blocks
    # 1.Group blocks by title and interline_equation
    # 2.bboxThe boundary is reset based on line information

    result = []
    current_group = []

    for i in range(len(blocks)):
        current_block = blocks[i]

        # If the current block is text type
        if current_block['type'] == 'text':
            current_block['bbox_fs'] = copy.deepcopy(current_block['bbox'])
            if 'lines' in current_block and len(current_block['lines']) > 0:
                current_block['bbox_fs'] = [
                    min([line['bbox'][0] for line in current_block['lines']]),
                    min([line['bbox'][1] for line in current_block['lines']]),
                    max([line['bbox'][2] for line in current_block['lines']]),
                    max([line['bbox'][3] for line in current_block['lines']]),
                ]
            current_group.append(current_block)

        # Check if next block exists
        if i + 1 < len(blocks):
            next_block = blocks[i + 1]
            # If the next block is not text type and is title or interline_equation type
            if next_block['type'] in ['title', 'interline_equation']:
                result.append(current_group)
                current_group = []

    # process the last group
    if current_group:
        result.append(current_group)

    return result


def __is_list_or_index_block(block):
    # If a block is a list block The following characteristics should be met at the same time
    # 1.blockThere are multiple lines in it 2.block If there are multiple lines, write them on the left side of the box. 3.blockThere are multiple lines in it No top grid on the right side (dogtooth shape)
    # 1.blockThere are multiple lines in it 2.block If there are multiple lines, write them on the left side of the box. 3.Multiple lines end with endflag
    # 1.blockThere are multiple lines in it 2.block If there are multiple lines, write them on the left side of the box. 3.blockThere are multiple lines in it No top grid on the left side

    # index block is a special list block
    # If a block is an index block The following characteristics should be met at the same time
    # 1.blockThere are multiple lines in it 2.block There are multiple lines in it, and they are written on both sides. 3.lineThe beginning or end of is a number
    if len(block['lines']) >= 2:
        first_line = block['lines'][0]
        line_height = first_line['bbox'][3] - first_line['bbox'][1]
        block_weight = block['bbox_fs'][2] - block['bbox_fs'][0]
        block_height = block['bbox_fs'][3] - block['bbox_fs'][1]
        page_weight, page_height = block['page_size']

        left_close_num = 0
        left_not_close_num = 0
        right_not_close_num = 0
        right_close_num = 0
        lines_text_list = []
        center_close_num = 0
        external_sides_not_close_num = 0
        multiple_para_flag = False
        last_line = block['lines'][-1]

        if page_weight == 0:
            block_weight_radio = 0
        else:
            block_weight_radio = block_weight / page_weight
        # logger.info(f"block_weight_radio: {block_weight_radio}")

        # If the left side of the first row does not have a top grid but the right side has a top grid,The left side of the last row is the top grid but the right side is not. （The first row may not have a top grid on the right)
        if (
            first_line['bbox'][0] - block['bbox_fs'][0] > line_height / 2
            and abs(last_line['bbox'][0] - block['bbox_fs'][0]) < line_height / 2
            and block['bbox_fs'][2] - last_line['bbox'][2] > line_height
        ):
            multiple_para_flag = True

        block_text = ''

        for line in block['lines']:
            line_text = ''

            for span in line['spans']:
                span_type = span['type']
                if span_type == ContentType.TEXT:
                    line_text += span['content'].strip()
            # Add all text, including blank lines, keeping it with block['lines']Same length
            lines_text_list.append(line_text)
            block_text = ''.join(lines_text_list)

        block_lang = detect_lang(block_text)
        # logger.info(f"block_lang: {block_lang}")

        for line in block['lines']:
            line_mid_x = (line['bbox'][0] + line['bbox'][2]) / 2
            block_mid_x = (block['bbox_fs'][0] + block['bbox_fs'][2]) / 2
            if (
                line['bbox'][0] - block['bbox_fs'][0] > 0.7 * line_height
                and block['bbox_fs'][2] - line['bbox'][2] > 0.7 * line_height
            ):
                external_sides_not_close_num += 1
            if abs(line_mid_x - block_mid_x) < line_height / 2:
                center_close_num += 1

            # To calculate whether the number of top cells on the left side of the line is greater than 2, use abs(block['bbox_fs'][0] - line['bbox'][0]) < line_height/2 to judge
            if abs(block['bbox_fs'][0] - line['bbox'][0]) < line_height / 2:
                left_close_num += 1
            elif line['bbox'][0] - block['bbox_fs'][0] > line_height:
                left_not_close_num += 1

            # Calculate whether the right side is top grid
            if abs(block['bbox_fs'][2] - line['bbox'][2]) < line_height:
                right_close_num += 1
            else:
                # If there are no very long words in Chinese-like language, a unified threshold can be used.
                if block_lang in ['zh', 'ja', 'ko']:
                    closed_area = 0.26 * block_weight
                else:
                    # Is there a certain distance when there is no top on the right side? Use 0 when patting the head..3blockWidth threshold
                    # blockThe threshold for wide blocks can be smaller, and the threshold for narrow blocks should be larger.
                    if block_weight_radio >= 0.5:
                        closed_area = 0.26 * block_weight
                    else:
                        closed_area = 0.36 * block_weight
                if block['bbox_fs'][2] - line['bbox'][2] > closed_area:
                    right_not_close_num += 1

        # Determine whether more than 80% of the elements in lines_text_list end with LIST_END_FLAG
        line_end_flag = False
        # Determine whether more than 80% of the elements in lines_text_list start with a number or end with a number
        line_num_flag = False
        num_start_count = 0
        num_end_count = 0
        flag_end_count = 0

        if len(lines_text_list) > 0:
            for line_text in lines_text_list:
                if len(line_text) > 0:
                    if line_text[-1] in LIST_END_FLAG:
                        flag_end_count += 1
                    if line_text[0].isdigit():
                        num_start_count += 1
                    if line_text[-1].isdigit():
                        num_end_count += 1

            if (
                num_start_count / len(lines_text_list) >= 0.8
                or num_end_count / len(lines_text_list) >= 0.8
            ):
                line_num_flag = True
            if flag_end_count / len(lines_text_list) >= 0.8:
                line_end_flag = True

        # Some directories do not have a border on the right side, It is currently believed that the left or right side is fully welted, and it complies with the number rules and is extremely index.
        if (
            left_close_num / len(block['lines']) >= 0.8
            or right_close_num / len(block['lines']) >= 0.8
        ) and line_num_flag:
            for line in block['lines']:
                line[ListLineTag.IS_LIST_START_LINE] = True
            return BlockType.INDEX

        # Special list recognition where all lines are centered, each line needs to be wrapped, characterized by multiple lines, and most lines are not_closed before and after,The x coordinate of each line midpoint is close to
        # Supplementary conditions: There are requirements for the aspect ratio of the block
        elif (
            external_sides_not_close_num >= 2
            and center_close_num == len(block['lines'])
            and external_sides_not_close_num / len(block['lines']) >= 0.5
            and block_height / block_weight > 0.4
        ):
            for line in block['lines']:
                line[ListLineTag.IS_LIST_START_LINE] = True
            return BlockType.LIST

        elif (
            left_close_num >= 2
            and (right_not_close_num >= 2 or line_end_flag or left_not_close_num >= 2)
            and not multiple_para_flag
            # and block_weight_radio > 0.27
        ):
            # Process a special kind of list without indentation. All lines are pasted to the left, and the gap on the right is used to determine whether it is the end of the item.
            if left_close_num / len(block['lines']) > 0.8:
                # This type of short item has only one line for each item and is bordered on the left. list
                if flag_end_count == 0 and right_close_num / len(block['lines']) < 0.5:
                    for line in block['lines']:
                        if abs(block['bbox_fs'][0] - line['bbox'][0]) < line_height / 2:
                            line[ListLineTag.IS_LIST_START_LINE] = True
                # This is most of the lines item There are situations where there is an end identifier, and different items are distinguished according to the end identifier.
                elif line_end_flag:
                    for i, line in enumerate(block['lines']):
                        if (
                            len(lines_text_list[i]) > 0
                            and lines_text_list[i][-1] in LIST_END_FLAG
                        ):
                            line[ListLineTag.IS_LIST_END_LINE] = True
                            if i + 1 < len(block['lines']):
                                block['lines'][i + 1][
                                    ListLineTag.IS_LIST_START_LINE
                                ] = True
                # line itemThere is basically no end identifier, and there is no indentation. Use the space on the right to determine which items are end
                else:
                    line_start_flag = False
                    for i, line in enumerate(block['lines']):
                        if line_start_flag:
                            line[ListLineTag.IS_LIST_START_LINE] = True
                            line_start_flag = False

                        if (
                            abs(block['bbox_fs'][2] - line['bbox'][2])
                            > 0.1 * block_weight
                        ):
                            line[ListLineTag.IS_LIST_END_LINE] = True
                            line_start_flag = True
            # A special ordered list with indentation,start line The left side is not welted and starts with a number, end line by IS_LIST_END_FLAG end and quantity and start line consistent
            elif num_start_count >= 2 and num_start_count == flag_end_count:
                for i, line in enumerate(block['lines']):
                    if len(lines_text_list[i]) > 0:
                        if lines_text_list[i][0].isdigit():
                            line[ListLineTag.IS_LIST_START_LINE] = True
                        if lines_text_list[i][-1] in LIST_END_FLAG:
                            line[ListLineTag.IS_LIST_END_LINE] = True
            else:
                # Normal indented list processing
                for line in block['lines']:
                    if abs(block['bbox_fs'][0] - line['bbox'][0]) < line_height / 2:
                        line[ListLineTag.IS_LIST_START_LINE] = True
                    if abs(block['bbox_fs'][2] - line['bbox'][2]) > line_height:
                        line[ListLineTag.IS_LIST_END_LINE] = True

            return BlockType.LIST
        else:
            return BlockType.TEXT
    else:
        return BlockType.TEXT


def __merge_2_text_blocks(block1, block2):
    if len(block1['lines']) > 0:
        first_line = block1['lines'][0]
        line_height = first_line['bbox'][3] - first_line['bbox'][1]
        block1_weight = block1['bbox'][2] - block1['bbox'][0]
        block2_weight = block2['bbox'][2] - block2['bbox'][0]
        min_block_weight = min(block1_weight, block2_weight)
        if abs(block1['bbox_fs'][0] - first_line['bbox'][0]) < line_height / 2:
            last_line = block2['lines'][-1]
            if len(last_line['spans']) > 0:
                last_span = last_line['spans'][-1]
                line_height = last_line['bbox'][3] - last_line['bbox'][1]
                if len(first_line['spans']) > 0:
                    first_span = first_line['spans'][0]
                    if len(first_span['content']) > 0:
                        span_start_with_num = first_span['content'][0].isdigit()
                        span_start_with_big_char = first_span['content'][0].isupper()
                        if (
                            # The difference between the right boundary of the last line of the previous block and the right boundary of the block does not exceed line_height.
                            abs(block2['bbox_fs'][2] - last_line['bbox'][2]) < line_height
                            # The last span of the previous block does not end with a specific symbol
                            and not last_span['content'].endswith(LINE_STOP_FLAG)
                            # Two blocks will not be merged if the difference in width exceeds 2 times.
                            and abs(block1_weight - block2_weight) < min_block_weight
                            # The first character of the next block is a number
                            and not span_start_with_num
                            # The first character of the next block is an uppercase letter
                            and not span_start_with_big_char
                        ):
                            if block1['page_num'] != block2['page_num']:
                                for line in block1['lines']:
                                    for span in line['spans']:
                                        span[SplitFlag.CROSS_PAGE] = True
                            block2['lines'].extend(block1['lines'])
                            block1['lines'] = []
                            block1[SplitFlag.LINES_DELETED] = True

    return block1, block2


def __merge_2_list_blocks(block1, block2):
    if block1['page_num'] != block2['page_num']:
        for line in block1['lines']:
            for span in line['spans']:
                span[SplitFlag.CROSS_PAGE] = True
    block2['lines'].extend(block1['lines'])
    block1['lines'] = []
    block1[SplitFlag.LINES_DELETED] = True

    return block1, block2


def __is_list_group(text_blocks_group):
    # list groupThe characteristic is that all blocks in a group meet the following conditions
    # 1.Each block should not exceed 3 lines 2. each block The left boundaries of are relatively close (the logic is simple and we don’t add this rule for now)
    for block in text_blocks_group:
        if len(block['lines']) > 3:
            return False
    return True


def __para_merge_page(blocks):
    page_text_blocks_groups = __process_blocks(blocks)
    for text_blocks_group in page_text_blocks_groups:
        if len(text_blocks_group) > 0:
            # It is necessary to first determine whether all blocks are lists before merging. or index block
            for block in text_blocks_group:
                block_type = __is_list_or_index_block(block)
                block['type'] = block_type
                # logger.info(f"{block['type']}:{block}")

        if len(text_blocks_group) > 1:
            # Determine this group before merging whether it is a list group
            is_list_group = __is_list_group(text_blocks_group)

            # Traverse in reverse order
            for i in range(len(text_blocks_group) - 1, -1, -1):
                current_block = text_blocks_group[i]

                # Check if there is a previous block
                if i - 1 >= 0:
                    prev_block = text_blocks_group[i - 1]

                    if (
                        current_block['type'] == 'text'
                        and prev_block['type'] == 'text'
                        and not is_list_group
                    ):
                        __merge_2_text_blocks(current_block, prev_block)
                    elif (
                        current_block['type'] == BlockType.LIST
                        and prev_block['type'] == BlockType.LIST
                    ) or (
                        current_block['type'] == BlockType.INDEX
                        and prev_block['type'] == BlockType.INDEX
                    ):
                        __merge_2_list_blocks(current_block, prev_block)

        else:
            continue


def para_split(page_info_list):
    all_blocks = []
    for page_info in page_info_list:
        blocks = copy.deepcopy(page_info['preproc_blocks'])
        for block in blocks:
            block['page_num'] = page_info['page_idx']
            block['page_size'] = page_info['page_size']
        all_blocks.extend(blocks)

    __para_merge_page(all_blocks)
    for page_info in page_info_list:
        page_info['para_blocks'] = []
        for block in all_blocks:
            if 'page_num' in block:
                if block['page_num'] == page_info['page_idx']:
                    page_info['para_blocks'].append(block)
                    # Remove unnecessary page_num and page_size fields from block
                    del block['page_num']
                    del block['page_size']


if __name__ == '__main__':
    input_blocks = []
    # call function
    groups = __process_blocks(input_blocks)
    for group_index, group in enumerate(groups):
        print(f'Group {group_index}: {group}')
