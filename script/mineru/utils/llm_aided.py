# Copyright (c) Opendatalab. All rights reserved.
from loguru import logger
from openai import OpenAI
import json_repair

from mineru.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text


def llm_aided_title(page_info_list, title_aided_config):
    client = OpenAI(
        api_key=title_aided_config["api_key"],
        base_url=title_aided_config["base_url"],
    )
    title_dict = {}
    origin_title_list = []
    i = 0
    for page_info in page_info_list:
        blocks = page_info["para_blocks"]
        for block in blocks:
            if block["type"] == "title":
                origin_title_list.append(block)
                title_text = merge_para_with_text(block)

                if 'line_avg_height' in block:
                    line_avg_height = block['line_avg_height']
                else:
                    title_block_line_height_list = []
                    for line in block['lines']:
                        bbox = line['bbox']
                        title_block_line_height_list.append(int(bbox[3] - bbox[1]))
                    if len(title_block_line_height_list) > 0:
                        line_avg_height = sum(title_block_line_height_list) / len(title_block_line_height_list)
                    else:
                        line_avg_height = int(block['bbox'][3] - block['bbox'][1])

                title_dict[f"{i}"] = [title_text, line_avg_height, int(page_info['page_idx']) + 1]
                i += 1
    # logger.info(f"Title list: {title_dict}")

    title_optimize_prompt = f"""The input content is a dictionary composed of all titles in a document. Please optimize the title results according to the following guidelines so that the results conform to the hierarchical structure of the normal document:

1. Each value in the dictionary is a list, containing the following elements:
    - title text
    - Text line height is the average line height of the block in which the title appears
    - The page number where the title is located

2. Keep original content:
    - All elements in the input dictionary are valid, and no elements in the dictionary can be deleted.
    - Please make sure that the number of elements in the output dictionary is consistent with the number of inputs

3. Keep the key-value correspondence in the dictionary unchanged

4. Optimize hierarchy:
    - Add appropriate hierarchy to each heading element based on the semantics of the heading content
    - Headings with larger line heights are generally higher-level headings
    - The title level must be continuous from front to back, and levels cannot be skipped.
    - The title level should be up to 4 levels. Do not add too many levels.
    - The optimized title only retains the integer representing the level of the title, and does not retain other information.

5. Reasonability check and fine-tuning:
    - After completing the preliminary grading, carefully check the reasonableness of the grading results
    - Fine-tune unreasonable grading based on context and logical sequence
    - Ensure that the final classification result conforms to the actual structure and logic of the document

IMPORTANT: 
Please directly return the optimized dictionary composed of title levels in the format{{title id:Title level}}，as follows:
{{
  0:1,
  1:2,
  2:2,
  3:3
}}
There is no need to format the dictionary and no other information needs to be returned.

Input title list:
{title_dict}

Corrected title list:
"""
    #5.
    #- The dictionary may contain text that is mistaken for a title. You can do this by marking its level as 0 to exclude them

    retry_count = 0
    max_retries = 3
    dict_completion = None

    # Build API call parameters
    api_params = {
        "model": title_aided_config["model"],
        "messages": [{'role': 'user', 'content': title_optimize_prompt}],
        "temperature": 0.7,
        "stream": True,
    }

    # Only add extra_body when explicitly specified in config
    if "enable_thinking" in title_aided_config:
        api_params["extra_body"] = {"enable_thinking": title_aided_config["enable_thinking"]}

    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(**api_params)
            content_pieces = []
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content_pieces.append(chunk.choices[0].delta.content)
            content = "".join(content_pieces).strip()
            # logger.info(f"Title completion: {content}")
            if "</think>" in content:
                idx = content.index("</think>") + len("</think>")
                content = content[idx:].strip()
            dict_completion = json_repair.loads(content)
            dict_completion = {int(k): int(v) for k, v in dict_completion.items()}

            # logger.info(f"len(dict_completion): {len(dict_completion)}, len(title_dict): {len(title_dict)}")
            if len(dict_completion) == len(title_dict):
                for i, origin_title_block in enumerate(origin_title_list):
                    origin_title_block["level"] = int(dict_completion[i])
                break
            else:
                logger.warning(
                    "The number of titles in the optimized result is not equal to the number of titles in the input.")
                retry_count += 1
        except Exception as e:
            logger.exception(e)
            retry_count += 1

    if dict_completion is None:
        logger.error("Failed to decode dict after maximum retries.")
