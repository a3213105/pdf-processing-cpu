import math


def is_in(box1, box2) -> bool:
    """box1Is it completely inside box2?."""
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    return (
        x0_1 >= x0_2  # box1The left boundary of is not outside the left side of box2
        and y0_1 >= y0_2  # box1The upper boundary of is not outside the upper edge of box2
        and x1_1 <= x1_2  # box1The right boundary of is not outside the right side of box2
        and y1_1 <= y1_2
    )  # box1The lower boundary of is not outside the lower edge of box2


def bbox_relative_pos(bbox1, bbox2):
    """Determine the relative position relationship between two rectangular boxes.

    Args:
        bbox1: A four-tuple representing the coordinates of the upper left corner and lower right corner of the first rectangular box, the format is (x1, y1, x1b, y1b)
        bbox2: A four-tuple representing the coordinates of the upper left corner and lower right corner of the second rectangular box, the format is (x2, y2, x2b, y2b)

    Returns:
        A four-tuple representing the positional relationship of rectangular box 1 relative to rectangular box 2. The format is (left, right, bottom, top)
        Among them, left indicates whether rectangular frame 1 is on the left side of rectangular frame 2, and right indicates whether rectangular frame 1 is on the right side of rectangular frame 2.
        bottomIndicates whether rectangular frame 1 is below rectangular frame 2, top indicates whether rectangular frame 1 is above rectangular frame 2
    """
    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    return left, right, bottom, top


def bbox_distance(bbox1, bbox2):
    """Calculate the distance between two rectangular boxes.

    Args:
        bbox1 (tuple): The coordinates of the first rectangular box, in the format (x1, y1, x2, y2)，in (x1, y1) is the coordinate of the upper left corner, (x2, y2) are the coordinates of the lower right corner.
        bbox2 (tuple): The coordinates of the second rectangular box, in the format (x1, y1, x2, y2)，in (x1, y1) is the coordinate of the upper left corner, (x2, y2) are the coordinates of the lower right corner.

    Returns:
        float: The distance between rectangular boxes.
    """

    def dist(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    left, right, bottom, top = bbox_relative_pos(bbox1, bbox2)

    if top and left:
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    return 0.0


def bbox_center_distance(bbox1, bbox2):
    """Calculate the Euclidean distance between the center points of two rectangular boxes.

    Args:
        bbox1 (tuple): The coordinates of the first rectangular box, in the format (x1, y1, x2, y2)
        bbox2 (tuple): The coordinates of the second rectangular box, in the format (x1, y1, x2, y2)

    Returns:
        float: The distance between the center points of two rectangular boxes
    """
    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    # Calculate center point
    center1_x = (x1 + x1b) / 2
    center1_y = (y1 + y1b) / 2
    center2_x = (x2 + x2b) / 2
    center2_y = (y2 + y2b) / 2

    # Calculate Euclidean distance
    return math.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)


def get_minbox_if_overlap_by_ratio(bbox1, bbox2, ratio):
    """Calculate the ratio of the overlapping area of ​​two bboxes to the smallest area box by calculate_overlap_area_2_minbox_area_ratio
    If the ratio is greater than ratio, return the smaller bbox, Otherwise return None."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    overlap_ratio = calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2)
    if overlap_ratio > ratio:
        if area1 <= area2:
            return bbox1
        else:
            return bbox2
    else:
        return None


def calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2):
    """Calculate the ratio of the overlapping area of ​​box1 and box2 to the smallest area box."""
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    min_box_area = min([(bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]),
                        (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0])])
    if min_box_area == 0:
        return 0
    else:
        return intersection_area / min_box_area


def calculate_iou(bbox1, bbox2):
    """Calculate the intersection-over-union ratio (IOU) of two bounding boxes.

    Args:
        bbox1 (list[float]): Coordinates of the first bounding box, in the format [x1, y1, x2, y2]，in (x1, y1) is the coordinate of the upper left corner, (x2, y2) are the coordinates of the lower right corner.
        bbox2 (list[float]): The coordinates of the second bounding box, in the format `bbox1` same.

    Returns:
        float: The intersection and union ratio (IOU) of two bounding boxes, the value range is [0, 1]。
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # The area of both rectangles
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    if any([bbox1_area == 0, bbox2_area == 0]):
        return 0

    # Compute the intersection over union by taking the intersection area
    # and dividing it by the sum of both areas minus the intersection area
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)

    return iou


def calculate_overlap_area_in_bbox1_area_ratio(bbox1, bbox2):
    """Calculate the ratio of the overlapping area of ​​box1 and box2 to bbox1."""
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    if bbox1_area == 0:
        return 0
    else:
        return intersection_area / bbox1_area


def calculate_vertical_projection_overlap_ratio(block1, block2):
    """
    Calculate the proportion of the x-axis covered by the vertical projection of two blocks.

    Args:
        block1 (tuple): Coordinates of the first block (x0, y0, x1, y1).
        block2 (tuple): Coordinates of the second block (x0, y0, x1, y1).

    Returns:
        float: The proportion of the x-axis covered by the vertical projection of the two blocks.
    """
    x0_1, _, x1_1, _ = block1
    x0_2, _, x1_2, _ = block2

    # Calculate the intersection of the x-coordinates
    x_left = max(x0_1, x0_2)
    x_right = min(x1_1, x1_2)

    if x_right < x_left:
        return 0.0

    # Length of the intersection
    intersection_length = x_right - x_left

    # Length of the x-axis projection of the first block
    block1_length = x1_1 - x0_1

    if block1_length == 0:
        return 0.0

    # Proportion of the x-axis covered by the intersection
    # logger.info(f"intersection_length: {intersection_length}, block1_length: {block1_length}")
    return intersection_length / block1_length