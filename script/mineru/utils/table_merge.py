# Copyright (c) Opendatalab. All rights reserved.
from copy import deepcopy

from loguru import logger
from bs4 import BeautifulSoup

from mineru.utils.char_utils import full_to_half
from mineru.utils.enum_class import BlockType, SplitFlag
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text

CONTINUATION_END_MARKERS = [
    "(Continued)",
    "(Continued table)",
    "((Continued from above table)",
    "(continued)",
    "(cont.)",
    "(cont’d)",
    "(…continued)",
    "Continuation table",
]

CONTINUATION_INLINE_MARKERS = [
    "(continued)",
]


def calculate_table_total_columns(soup):
    """Calculate the total number of columns of the table, processing rowspan and colspan by analyzing the entire table structure

    Args:
        soup: BeautifulSoupparsed table

    Returns:
        int: Total number of columns in the table
    """
    rows = soup.find_all("tr")
    if not rows:
        return 0

    # Create a matrix to track occupancy at each location
    max_cols = 0
    occupied = {}  # {row_idx: {col_idx: True}}

    for row_idx, row in enumerate(rows):
        col_idx = 0
        cells = row.find_all(["td", "th"])

        if row_idx not in occupied:
            occupied[row_idx] = {}

        for cell in cells:
            # Find the next unoccupied column position
            while col_idx in occupied[row_idx]:
                col_idx += 1

            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            # Mark all positions occupied by this cell
            for r in range(row_idx, row_idx + rowspan):
                if r not in occupied:
                    occupied[r] = {}
                for c in range(col_idx, col_idx + colspan):
                    occupied[r][c] = True

            col_idx += colspan
            max_cols = max(max_cols, col_idx)

    return max_cols


def build_table_occupied_matrix(soup):
    """Construct the occupancy matrix of the table and return the effective number of columns in each row

    Args:
        soup: BeautifulSoupparsed table

    Returns:
        dict: {row_idx: effective_columns} The effective number of columns per row (taking into account rowspan occupancy)
    """
    rows = soup.find_all("tr")
    if not rows:
        return {}

    occupied = {}  # {row_idx: {col_idx: True}}
    row_effective_cols = {}  # {row_idx: effective_columns}

    for row_idx, row in enumerate(rows):
        col_idx = 0
        cells = row.find_all(["td", "th"])

        if row_idx not in occupied:
            occupied[row_idx] = {}

        for cell in cells:
            # Find the next unoccupied column position
            while col_idx in occupied[row_idx]:
                col_idx += 1

            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            # Mark all positions occupied by this cell
            for r in range(row_idx, row_idx + rowspan):
                if r not in occupied:
                    occupied[r] = {}
                for c in range(col_idx, col_idx + colspan):
                    occupied[r][c] = True

            col_idx += colspan

        # The effective number of columns in this row is the largest occupied column index+1
        if occupied[row_idx]:
            row_effective_cols[row_idx] = max(occupied[row_idx].keys()) + 1
        else:
            row_effective_cols[row_idx] = 0

    return row_effective_cols


def calculate_row_effective_columns(soup, row_idx):
    """Calculate the effective number of columns in the specified row (considering rowspan occupation)

    Args:
        soup: BeautifulSoupparsed table
        row_idx: row index

    Returns:
        int: The number of valid columns in the row
    """
    row_effective_cols = build_table_occupied_matrix(soup)
    return row_effective_cols.get(row_idx, 0)


def calculate_row_columns(row):
    """
    Calculate the actual number of columns of table rows, taking into account the colspan attribute

    Args:
        row: BeautifulSouptr element object

    Returns:
        int: The actual number of columns in the row
    """
    cells = row.find_all(["td", "th"])
    column_count = 0

    for cell in cells:
        colspan = int(cell.get("colspan", 1))
        column_count += colspan

    return column_count


def calculate_visual_columns(row):
    """
    Calculate the number of visual columns for table rows (actual number of td/th cells, colspan not taken into account)

    Args:
        row: BeautifulSouptr element object

    Returns:
        int: Visual column number of rows (actual number of cells)
    """
    cells = row.find_all(["td", "th"])
    return len(cells)


def detect_table_headers(soup1, soup2, max_header_rows=5):
    """
    Detect and compare headers of two tables

    Args:
        soup1: BeautifulSoup object for the first table
        soup2: BeautifulSoup object for the second table
        max_header_rows: Maximum possible number of header rows

    Returns:
        tuple: (Number of header rows, Are the headers consistent?, header text list)
    """
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")

    # Construct a matrix of effective column numbers for two tables
    effective_cols1 = build_table_occupied_matrix(soup1)
    effective_cols2 = build_table_occupied_matrix(soup2)

    min_rows = min(len(rows1), len(rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    header_texts = []

    for i in range(min_rows):
        # Extract all cells of the current row
        cells1 = rows1[i].find_all(["td", "th"])
        cells2 = rows2[i].find_all(["td", "th"])

        # Check whether the structure and content of the two lines are consistent
        structure_match = True

        # First check the number of cells
        if len(cells1) != len(cells2):
            structure_match = False
        else:
            # Check whether the number of effective columns is consistent (consider the impact of rowspan)
            if effective_cols1.get(i, 0) != effective_cols2.get(i, 0):
                structure_match = False
            else:
                # Then check the properties and contents of the cell
                for cell1, cell2 in zip(cells1, cells2):
                    colspan1 = int(cell1.get("colspan", 1))
                    rowspan1 = int(cell1.get("rowspan", 1))
                    colspan2 = int(cell2.get("colspan", 1))
                    rowspan2 = int(cell2.get("rowspan", 1))

                    # Remove all whitespace characters (including spaces, newlines, tabs, etc.)
                    text1 = ''.join(full_to_half(cell1.get_text()).split())
                    text2 = ''.join(full_to_half(cell2.get_text()).split())

                    if colspan1 != colspan2 or rowspan1 != rowspan2 or text1 != text2:
                        structure_match = False
                        break

        if structure_match:
            header_rows += 1
            row_texts = [full_to_half(cell.get_text().strip()) for cell in cells1]
            header_texts.append(row_texts)  # Add header text
        else:
            headers_match = header_rows > 0  # The header is considered a match only if at least one row matches.
            break

    # If strict match fails, try visual consistency match (only text content is compared)
    if header_rows == 0:
        header_rows, headers_match, header_texts = _detect_table_headers_visual(soup1, soup2, rows1, rows2, max_header_rows)

    return header_rows, headers_match, header_texts


def _detect_table_headers_visual(soup1, soup2, rows1, rows2, max_header_rows=5):
    """
    Detect table headers based on visual consistency (only compare text content, ignore colspan/rowspan differences)

    Args:
        soup1: BeautifulSoup object for the first table
        soup2: BeautifulSoup object for the second table
        rows1: Row list of first table
        rows2: Row list of second table
        max_header_rows: Maximum possible number of header rows

    Returns:
        tuple: (Number of header rows, Are the headers consistent?, header text list)
    """
    # Construct a matrix of effective column numbers for two tables
    effective_cols1 = build_table_occupied_matrix(soup1)
    effective_cols2 = build_table_occupied_matrix(soup2)

    min_rows = min(len(rows1), len(rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    header_texts = []

    for i in range(min_rows):
        cells1 = rows1[i].find_all(["td", "th"])
        cells2 = rows2[i].find_all(["td", "th"])

        # Extract the text content list of each line (remove whitespace characters)
        texts1 = [''.join(full_to_half(cell.get_text()).split()) for cell in cells1]
        texts2 = [''.join(full_to_half(cell.get_text()).split()) for cell in cells2]

        # Check for visual consistency: the text content is exactly the same and the number of valid columns is the same
        effective_cols_match = effective_cols1.get(i, 0) == effective_cols2.get(i, 0)
        if texts1 == texts2 and effective_cols_match:
            header_rows += 1
            row_texts = [full_to_half(cell.get_text().strip()) for cell in cells1]
            header_texts.append(row_texts)
        else:
            headers_match = header_rows > 0
            break

    if header_rows == 0:
        headers_match = False

    return header_rows, headers_match, header_texts


def can_merge_tables(current_table_block, previous_table_block):
    """Determine whether two tables can be merged"""
    # Check whether the table has caption and footnote
    # Calculate the number of footnotes in previous_table_block
    footnote_count = sum(1 for block in previous_table_block["blocks"] if block["type"] == BlockType.TABLE_FOOTNOTE)
    # If there is a block of type TABLE_CAPTION,Check if there is at least one starting with"(Continued)"ending
    caption_blocks = [block for block in current_table_block["blocks"] if block["type"] == BlockType.TABLE_CAPTION]
    if caption_blocks:
        # Check whether at least one caption contains the continuation table identifier
        has_continuation_marker = False
        for block in caption_blocks:
            caption_text = full_to_half(merge_para_with_text(block).strip()).lower()
            if (
                    any(caption_text.endswith(marker.lower()) for marker in CONTINUATION_END_MARKERS)
                    or any(marker.lower() in caption_text for marker in CONTINUATION_INLINE_MARKERS)
            ):
                has_continuation_marker = True
                break

        # If all captions do not contain the continuation table identifier, merging is not allowed.
        if not has_continuation_marker:
            return False, None, None, None, None

        # If the caption of current_table_block has a continuation identifier,Relax footnote restrictions to allow previous_table_block to have at most one footnote
        if footnote_count > 1:
            return False, None, None, None, None
    else:
        if footnote_count > 0:
            return False, None, None, None, None

    # Get the HTML content of two tables
    current_html = ""
    previous_html = ""

    for block in current_table_block["blocks"]:
        if (block["type"] == BlockType.TABLE_BODY and block["lines"] and block["lines"][0]["spans"]):
            current_html = block["lines"][0]["spans"][0].get("html", "")

    for block in previous_table_block["blocks"]:
        if (block["type"] == BlockType.TABLE_BODY and block["lines"] and block["lines"][0]["spans"]):
            previous_html = block["lines"][0]["spans"][0].get("html", "")

    if not current_html or not previous_html:
        return False, None, None, None, None

    # Check table width differences
    x0_t1, y0_t1, x1_t1, y1_t1 = current_table_block["bbox"]
    x0_t2, y0_t2, x1_t2, y1_t2 = previous_table_block["bbox"]
    table1_width = x1_t1 - x0_t1
    table2_width = x1_t2 - x0_t2

    if abs(table1_width - table2_width) / min(table1_width, table2_width) >= 0.1:
        return False, None, None, None, None

    # Parse HTML and check table structure
    soup1 = BeautifulSoup(previous_html, "html.parser")
    soup2 = BeautifulSoup(current_html, "html.parser")

    # Check overall column count match
    table_cols1 = calculate_table_total_columns(soup1)
    table_cols2 = calculate_table_total_columns(soup2)
    # logger.debug(f"Table columns - Previous: {table_cols1}, Current: {table_cols2}")
    tables_match = table_cols1 == table_cols2

    # Check the matching of first and last row and column numbers
    rows_match = check_rows_match(soup1, soup2)

    return (tables_match or rows_match), soup1, soup2, current_html, previous_html


def check_rows_match(soup1, soup2):
    """Check if table rows match"""
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")

    if not (rows1 and rows2):
        return False

    # Get the row index of the last row of the first table
    last_row_idx = None
    last_row = None
    for idx in range(len(rows1) - 1, -1, -1):
        if rows1[idx].find_all(["td", "th"]):
            last_row_idx = idx
            last_row = rows1[idx]
            break

    # Detect the number of header rows in order to obtain the first data row of the second table
    header_count, _, _ = detect_table_headers(soup1, soup2)

    # Get the first data row of the second table
    first_data_row_idx = None
    first_data_row = None
    if len(rows2) > header_count:
        first_data_row_idx = header_count
        first_data_row = rows2[header_count]  # first non-header row

    if not (last_row and first_data_row):
        return False

    # Calculate the number of effective columns (consider rowspan and colspan)
    last_row_effective_cols = calculate_row_effective_columns(soup1, last_row_idx)
    first_row_effective_cols = calculate_row_effective_columns(soup2, first_data_row_idx)

    # Calculate the actual number of columns (only considering colspan) and the visual number of columns
    last_row_cols = calculate_row_columns(last_row)
    first_row_cols = calculate_row_columns(first_data_row)
    last_row_visual_cols = calculate_visual_columns(last_row)
    first_row_visual_cols = calculate_visual_columns(first_data_row)

    # logger.debug(f"Number of rows and columns - The last row of the previous table: {last_row_cols}(Valid number of columns:{last_row_effective_cols}, visual column number:{last_row_visual_cols}), The first row of the current table: {first_row_cols}(Valid number of columns:{first_row_effective_cols}, visual column number:{first_row_visual_cols})")

    # Considers effective column number matching, actual column number matching, and visual column number matching simultaneously
    return (last_row_effective_cols == first_row_effective_cols or
            last_row_cols == first_row_cols or
            last_row_visual_cols == first_row_visual_cols)


def check_row_columns_match(row1, row2):
    # Check whether the colspan attributes are consistent cell by cell
    cells1 = row1.find_all(["td", "th"])
    cells2 = row2.find_all(["td", "th"])
    if len(cells1) != len(cells2):
        return False
    for cell1, cell2 in zip(cells1, cells2):
        colspan1 = int(cell1.get("colspan", 1))
        colspan2 = int(cell2.get("colspan", 1))
        if colspan1 != colspan2:
            return False
    return True


def adjust_table_rows_colspan(soup, rows, start_idx, end_idx,
                              reference_structure, reference_visual_cols,
                              target_cols, current_cols, reference_row):
    """Adjust the colspan attribute of table rows to match the target number of columns

    Args:
        soup: BeautifulSoupParsed table object (used to calculate the number of valid columns)
        rows: Table row list
        start_idx: Starting row index
        end_idx: end row index (not included)
        reference_structure: List of colspan structures for reference rows
        reference_visual_cols: The visual column number of the reference row
        target_cols: Target total number of columns
        current_cols: Current total number of columns
        reference_row: Reference row object
    """
    reference_row_copy = deepcopy(reference_row)

    # Construct a matrix with valid number of columns
    effective_cols_matrix = build_table_occupied_matrix(soup)

    for i in range(start_idx, end_idx):
        row = rows[i]
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        # Use the effective number of columns (considering rowspan) to determine whether adjustment is needed
        current_row_effective_cols = effective_cols_matrix.get(i, 0)
        current_row_cols = calculate_row_columns(row)

        # Skip if the effective number of columns or the actual number of columns has reached the target
        if current_row_effective_cols >= target_cols or current_row_cols >= target_cols:
            continue

        # Check if it matches the reference row structure
        if calculate_visual_columns(row) == reference_visual_cols and check_row_columns_match(row, reference_row_copy):
            # Try applying a reference structure
            if len(cells) <= len(reference_structure):
                for j, cell in enumerate(cells):
                    if j < len(reference_structure) and reference_structure[j] > 1:
                        cell["colspan"] = str(reference_structure[j])
        else:
            # Extend last cell to fill column count difference
            # Use effective number of columns to calculate differences
            cols_diff = target_cols - current_row_effective_cols
            if cols_diff > 0:
                last_cell = cells[-1]
                current_last_span = int(last_cell.get("colspan", 1))
                last_cell["colspan"] = str(current_last_span + cols_diff)


def perform_table_merge(soup1, soup2, previous_table_block, wait_merge_table_footnotes):
    """Perform table merge operation"""
    # Check the number of rows in the header and confirm whether the contents of the header are consistent
    header_count, headers_match, header_texts = detect_table_headers(soup1, soup2)
    # logger.debug(f"Number of header rows detected: {header_count}, Header match: {headers_match}")
    # logger.debug(f"Header content: {header_texts}")

    # Find the tbody of the first table, if not find the table element
    tbody1 = soup1.find("tbody") or soup1.find("table")

    # Get all rows from table 1 and table 2
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")


    if rows1 and rows2 and header_count < len(rows2):
        # Get the last row of table 1 and the first non-header row of table 2
        last_row1 = rows1[-1]
        first_data_row2 = rows2[header_count]

        # Calculate the total number of columns in the table
        table_cols1 = calculate_table_total_columns(soup1)
        table_cols2 = calculate_table_total_columns(soup2)
        if table_cols1 >= table_cols2:
            reference_structure = [int(cell.get("colspan", 1)) for cell in last_row1.find_all(["td", "th"])]
            reference_visual_cols = calculate_visual_columns(last_row1)
            # Using the last row of Table 1 as a reference, adjust the rows of Table 2
            adjust_table_rows_colspan(
                soup2, rows2, header_count, len(rows2),
                reference_structure, reference_visual_cols,
                table_cols1, table_cols2, first_data_row2
            )

        else:  # table_cols2 > table_cols1
            reference_structure = [int(cell.get("colspan", 1)) for cell in first_data_row2.find_all(["td", "th"])]
            reference_visual_cols = calculate_visual_columns(first_data_row2)
            # Using the first data row of Table 2 as a reference, adjust the rows of Table 1
            adjust_table_rows_colspan(
                soup1, rows1, 0, len(rows1),
                reference_structure, reference_visual_cols,
                table_cols2, table_cols1, last_row1
            )

    # Add rows from second table to first table
    if tbody1:
        tbody2 = soup2.find("tbody") or soup2.find("table")
        if tbody2:
            # Add rows from second table to first table (skipping header rows)
            for row in rows2[header_count:]:
                row.extract()
                tbody1.append(row)

    # Clear the footnote of previous_table_block
    previous_table_block["blocks"] = [
        block for block in previous_table_block["blocks"]
        if block["type"] != BlockType.TABLE_FOOTNOTE
    ]
    # Add the footnote of the table to be merged to the previous table
    for table_footnote in wait_merge_table_footnotes:
        temp_table_footnote = table_footnote.copy()
        temp_table_footnote[SplitFlag.CROSS_PAGE] = True
        previous_table_block["blocks"].append(temp_table_footnote)

    return str(soup1)


def merge_table(page_info_list):
    """Merge spreadsheets"""
    # Traverse each page in reverse order
    for page_idx in range(len(page_info_list) - 1, -1, -1):
        # Skip the first page because it has no previous page
        if page_idx == 0:
            continue

        page_info = page_info_list[page_idx]
        previous_page_info = page_info_list[page_idx - 1]

        # Check if the current page has a table block
        if not (page_info["para_blocks"] and page_info["para_blocks"][0]["type"] == BlockType.TABLE):
            continue

        current_table_block = page_info["para_blocks"][0]

        # Check if there is a table block on the previous page
        if not (previous_page_info["para_blocks"] and previous_page_info["para_blocks"][-1]["type"] == BlockType.TABLE):
            continue

        previous_table_block = previous_page_info["para_blocks"][-1]

        # Collect footnotes of tables to be merged
        wait_merge_table_footnotes = [
            block for block in current_table_block["blocks"]
            if block["type"] == BlockType.TABLE_FOOTNOTE
        ]

        # Check if two tables can be merged
        can_merge, soup1, soup2, current_html, previous_html = can_merge_tables(
            current_table_block, previous_table_block
        )

        if not can_merge:
            continue

        # Perform table merge
        merged_html = perform_table_merge(
            soup1, soup2, previous_table_block, wait_merge_table_footnotes
        )

        # Update the html of previous_table_block
        for block in previous_table_block["blocks"]:
            if (block["type"] == BlockType.TABLE_BODY and block["lines"] and block["lines"][0]["spans"]):
                block["lines"][0]["spans"][0]["html"] = merged_html
                break

        # Delete the table of the current page
        for block in current_table_block["blocks"]:
            block['lines'] = []
            block[SplitFlag.LINES_DELETED] = True
