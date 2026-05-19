"""
Contains methods and logic reused in the two MagicModel classes
"""
from typing import List, Dict, Any, Callable

from loguru import logger
from mineru.utils.boxbase import bbox_distance, bbox_center_distance, is_in


def reduct_overlap(bboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove overlapping bboxes and retain bboxes not included by other bboxes

    Args:
        bboxes: List of dictionaries containing bbox information

    Returns:
        bbox list after deduplication
    """
    N = len(bboxes)
    keep = [True] * N
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if is_in(bboxes[i]['bbox'], bboxes[j]['bbox']):
                keep[i] = False
    return [bboxes[i] for i in range(N) if keep[i]]


def tie_up_category_by_distance_v3(
        get_subjects_func: Callable,
        get_objects_func: Callable,
        extract_subject_func: Callable = None,
        extract_object_func: Callable = None
):
    """
    General category association method, used to associate subject objects with object objects

    parameter:
        get_subjects_func: Function, extract the main object
        get_objects_func: Function, extract object object
        extract_subject_func: Function, custom extraction of body attributes (default uses bbox and other attributes)
        extract_object_func: Function, customized to extract object attributes (bbox and other attributes are used by default)

    return:
        The associated object list
    """
    subjects = get_subjects_func()
    objects = get_objects_func()

    # If no custom extraction function is provided, the default function is used
    if extract_subject_func is None:
        extract_subject_func = lambda x: x
    if extract_object_func is None:
        extract_object_func = lambda x: x

    ret = []
    N, M = len(subjects), len(objects)
    subjects.sort(key=lambda x: x["bbox"][0] ** 2 + x["bbox"][1] ** 2)
    objects.sort(key=lambda x: x["bbox"][0] ** 2 + x["bbox"][1] ** 2)

    OBJ_IDX_OFFSET = 10000
    SUB_BIT_KIND, OBJ_BIT_KIND = 0, 1

    all_boxes_with_idx = [(i, SUB_BIT_KIND, sub["bbox"][0], sub["bbox"][1]) for i, sub in enumerate(subjects)] + [
        (i + OBJ_IDX_OFFSET, OBJ_BIT_KIND, obj["bbox"][0], obj["bbox"][1]) for i, obj in enumerate(objects)
    ]
    seen_idx = set()
    seen_sub_idx = set()

    while N > len(seen_sub_idx):
        candidates = []
        for idx, kind, x0, y0 in all_boxes_with_idx:
            if idx in seen_idx:
                continue
            candidates.append((idx, kind, x0, y0))

        if len(candidates) == 0:
            break
        left_x = min([v[2] for v in candidates])
        top_y = min([v[3] for v in candidates])

        candidates.sort(key=lambda x: (x[2] - left_x) ** 2 + (x[3] - top_y) ** 2)

        fst_idx, fst_kind, left_x, top_y = candidates[0]
        fst_bbox = subjects[fst_idx]['bbox'] if fst_kind == SUB_BIT_KIND else objects[fst_idx - OBJ_IDX_OFFSET]['bbox']
        candidates.sort(
            key=lambda x: bbox_distance(fst_bbox, subjects[x[0]]['bbox']) if x[1] == SUB_BIT_KIND else bbox_distance(
                fst_bbox, objects[x[0] - OBJ_IDX_OFFSET]['bbox']))
        nxt = None

        for i in range(1, len(candidates)):
            if candidates[i][1] ^ fst_kind == 1:
                nxt = candidates[i]
                break
        if nxt is None:
            break

        if fst_kind == SUB_BIT_KIND:
            sub_idx, obj_idx = fst_idx, nxt[0] - OBJ_IDX_OFFSET
        else:
            sub_idx, obj_idx = nxt[0], fst_idx - OBJ_IDX_OFFSET

        pair_dis = bbox_distance(subjects[sub_idx]["bbox"], objects[obj_idx]["bbox"])
        nearest_dis = float("inf")
        for i in range(N):
            # Cancel the original algorithm 1pair 1 Matching offset
            # if i in seen_idx or i == sub_idx:continue
            nearest_dis = min(nearest_dis, bbox_distance(subjects[i]["bbox"], objects[obj_idx]["bbox"]))

        if pair_dis >= 3 * nearest_dis:
            seen_idx.add(sub_idx)
            continue

        seen_idx.add(sub_idx)
        seen_idx.add(obj_idx + OBJ_IDX_OFFSET)
        seen_sub_idx.add(sub_idx)

        ret.append(
            {
                "sub_bbox": extract_subject_func(subjects[sub_idx]),
                "obj_bboxes": [extract_object_func(objects[obj_idx])],
                "sub_idx": sub_idx,
            }
        )

    for i in range(len(objects)):
        j = i + OBJ_IDX_OFFSET
        if j in seen_idx:
            continue
        seen_idx.add(j)
        nearest_dis, nearest_sub_idx = float("inf"), -1
        for k in range(len(subjects)):
            dis = bbox_distance(objects[i]["bbox"], subjects[k]["bbox"])
            if dis < nearest_dis:
                nearest_dis = dis
                nearest_sub_idx = k

        for k in range(len(subjects)):
            if k != nearest_sub_idx:
                continue
            if k in seen_sub_idx:
                for kk in range(len(ret)):
                    if ret[kk]["sub_idx"] == k:
                        ret[kk]["obj_bboxes"].append(extract_object_func(objects[i]))
                        break
            else:
                ret.append(
                    {
                        "sub_bbox": extract_subject_func(subjects[k]),
                        "obj_bboxes": [extract_object_func(objects[i])],
                        "sub_idx": k,
                    }
                )
            seen_sub_idx.add(k)
            seen_idx.add(k)

    for i in range(len(subjects)):
        if i in seen_sub_idx:
            continue
        ret.append(
            {
                "sub_bbox": extract_subject_func(subjects[i]),
                "obj_bboxes": [],
                "sub_idx": i,
            }
        )

    return ret


def tie_up_category_by_index(
        get_subjects_func: Callable,
        get_objects_func: Callable,
        extract_subject_func: Callable = None,
        extract_object_func: Callable = None,
        object_block_type: str = "object",
):
    """
    Index-based category association method, used to associate subject objects with object objects
    Objects are matched first to the subject with the closest index. The matching priority is:
    1. indexDifference (highest priority)
    2. bboxEdge distance (adjacent edge distance)
    3. bboxCenter point distance (lowest priority, as final tiebreaker)

    parameter:
        get_subjects_func: Function, extract the main object
        get_objects_func: Function, extract object object
        extract_subject_func: Function, custom extraction of body attributes (default uses bbox and other attributes)
        extract_object_func: Function, customized to extract object attributes (bbox and other attributes are used by default)

    return:
        The associated object list is sorted in ascending order by subject index.
    """
    subjects = get_subjects_func()
    objects = get_objects_func()

    # If no custom extraction function is provided, the default function is used
    if extract_subject_func is None:
        extract_subject_func = lambda x: x
    if extract_object_func is None:
        extract_object_func = lambda x: x

    # Initialize the result dictionary, key is the subject index, and value is the associated information
    result_dict = {}

    # Initialize all principals
    for i, subject in enumerate(subjects):
        result_dict[i] = {
            "sub_bbox": extract_subject_func(subject),
            "obj_bboxes": [],
            "sub_idx": i,
        }

    # Extract the index set of all objects and use it to calculate the effective index difference
    object_indices = set(obj["index"] for obj in objects)

    def calc_effective_index_diff(obj_index: int, sub_index: int) -> int:
        """
        Calculate the effective index difference
        effective difference = absolute difference - The number of other objects in the interval
        That is: if the difference between obj_index and sub_index is caused by other objects, this part of the difference should be deducted
        """
        if obj_index == sub_index:
            return 0

        start, end = min(obj_index, sub_index), max(obj_index, sub_index)
        abs_diff = end - start

        # Calculate interval (start, end)How many indexes of other objects are there?
        other_objects_count = 0
        for idx in range(start + 1, end):
            if idx in object_indices:
                other_objects_count += 1

        return abs_diff - other_objects_count

    # Find the best matching subject for each object
    for obj in objects:
        if len(subjects) == 0:
            # If there is no subject, skip the object
            continue

        obj_index = obj["index"]
        min_index_diff = float("inf")
        best_subject_indices = []

        # Find all subjects with the smallest effective index difference
        for i, subject in enumerate(subjects):
            sub_index = subject["index"]
            index_diff = calc_effective_index_diff(obj_index, sub_index)

            if index_diff < min_index_diff:
                min_index_diff = index_diff
                best_subject_indices = [i]
            elif index_diff == min_index_diff:
                best_subject_indices.append(i)

        if len(best_subject_indices) == 1:
            best_subject_idx = best_subject_indices[0]
        # If there are multiple subjects with the same index difference (up to two), filter based on edge distance.
        elif len(best_subject_indices) == 2:
            # Calculate the edge distance of all candidate subjects
            edge_distances = [(idx, bbox_distance(obj["bbox"], subjects[idx]["bbox"])) for idx in best_subject_indices]
            edge_dist_diff = abs(edge_distances[0][1] - edge_distances[1][1])

            for idx, edge_dist in edge_distances:
                logger.debug(f"Obj index: {obj_index}, Sub index: {subjects[idx]['index']}, Edge distance: {edge_dist}")

            if edge_dist_diff > 2:
                # If the edge distance difference is greater than 2, match the subject with a smaller edge distance.
                best_subject_idx = min(edge_distances, key=lambda x: x[1])[0]
                logger.debug(f"Obj index: {obj_index}, edge_dist_diff > 2, matching to subject with min edge distance, index: {subjects[best_subject_idx]['index']}")
            elif object_block_type == "table_caption":
                # edge distance difference<=2And it is table_caption, matching the subject with a larger index
                best_subject_idx = max(best_subject_indices, key=lambda idx: subjects[idx]["index"])
                logger.debug(f"Obj index: {obj_index}, edge_dist_diff <= 2 and table_caption, matching to later subject with index: {subjects[best_subject_idx]['index']}")
            elif object_block_type.endswith("footnote"):
                # edge distance difference<=2And it is a footnote, matching the subject with a smaller index
                best_subject_idx = min(best_subject_indices, key=lambda idx: subjects[idx]["index"])
                logger.debug(f"Obj index: {obj_index}, edge_dist_diff <= 2 and footnote, matching to earlier subject with index: {subjects[best_subject_idx]['index']}")
            else:
                # edge distance difference<=2 And special matching rules are not applicable, and center point distance matching is used.
                center_distances = [(idx, bbox_center_distance(obj["bbox"], subjects[idx]["bbox"])) for idx in best_subject_indices]
                for idx, center_dist in center_distances:
                    logger.debug(f"Obj index: {obj_index}, Sub index: {subjects[idx]['index']}, Center distance: {center_dist}")
                best_subject_idx = min(center_distances, key=lambda x: x[1])[0]
        else:
            raise ValueError("More than two subjects have the same minimal index difference, which is unexpected.")

        # Add objects to the best subject's obj_bboxes
        result_dict[best_subject_idx]["obj_bboxes"].append(extract_object_func(obj))

    # Convert to list and sort by body index
    ret = list(result_dict.values())
    ret.sort(key=lambda x: x["sub_idx"])

    return ret
