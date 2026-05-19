# Copyright (c) Opendatalab. All rights reserved.
import re
from io import BytesIO
import numpy as np
import pypdfium2 as pdfium
from loguru import logger
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.layout import LAParams, LTImage, LTFigure
from pdfminer.converter import PDFPageAggregator


def classify(pdf_bytes):
    """
    Determine whether a PDF file can directly extract text or requires OCR

    Args:
        pdf_bytes: PDFfile byte data

    Returns:
        str: 'txt' Indicates that text can be extracted directly,'ocr' Indicates the need for OCR
    """

    # Load PDF from byte data
    sample_pdf_bytes = extract_pages(pdf_bytes)
    pdf = pdfium.PdfDocument(sample_pdf_bytes)
    try:
        # Get the number of PDF pages
        page_count = len(pdf)

        # If the number of PDF pages is 0, return to OCR directly
        if page_count == 0:
            return 'ocr'

        # Number of pages checked (up to 10 pages checked)
        pages_to_check = min(page_count, 10)

        # Set the threshold: if there are less than 50 valid characters on average per page, OCR is considered required
        chars_threshold = 50

        # Check average character count and invalid characters
        if (get_avg_cleaned_chars_per_page(pdf, pages_to_check) < chars_threshold) or detect_invalid_chars(sample_pdf_bytes):
            return 'ocr'

        # Check image coverage
        if get_high_image_coverage_ratio(sample_pdf_bytes, pages_to_check) >= 0.8:
            return 'ocr'

        return 'txt'

    except Exception as e:
        logger.error(f"Error determining PDF type: {e}")
        # Use OCR by default when errors occur
        return 'ocr'

    finally:
        # Whichever path is executed, make sure the PDF is closed
        pdf.close()


def get_avg_cleaned_chars_per_page(pdf_doc, pages_to_check):
    # Total characters
    total_chars = 0
    # Total number of characters after cleaning
    cleaned_total_chars = 0

    # Check the text of previous pages
    for i in range(pages_to_check):
        page = pdf_doc[i]
        text_page = page.get_textpage()
        text = text_page.get_text_bounded()
        total_chars += len(text)

        # Clean extracted text, remove whitespace characters
        cleaned_text = re.sub(r'\s+', '', text)
        cleaned_total_chars += len(cleaned_text)

    # Calculate the average number of characters per page
    avg_cleaned_chars_per_page = cleaned_total_chars / pages_to_check

    # logger.debug(f"PDFanalyze: Average per page after cleaning{avg_cleaned_chars_per_page:.1f}character")

    return avg_cleaned_chars_per_page


def get_high_image_coverage_ratio(sample_pdf_bytes, pages_to_check):
    # Create memory file object
    pdf_stream = BytesIO(sample_pdf_bytes)

    # Create PDF parser
    parser = PDFParser(pdf_stream)

    # Create PDF document object
    document = PDFDocument(parser)

    # Check if document allows text extraction
    if not document.is_extractable:
        # logger.warning("PDFContent extraction is not allowed")
        return 1.0  # Defaults to high coverage because content cannot be extracted

    # Create resource manager and parameter objects
    rsrcmgr = PDFResourceManager()
    laparams = LAParams(
        line_overlap=0.5,
        char_margin=2.0,
        line_margin=0.5,
        word_margin=0.1,
        boxes_flow=None,
        detect_vertical=False,
        all_texts=False,
    )

    # Create aggregator
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)

    # Create an interpreter
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Record number of pages with high image coverage
    high_image_coverage_pages = 0
    page_count = 0

    # Traverse pages
    for page in PDFPage.create_pages(document):
        # Control the number of pages checked
        if page_count >= pages_to_check:
            break

        # Process page
        interpreter.process_page(page)
        layout = device.get_result()

        # page size
        page_width = layout.width
        page_height = layout.height
        page_area = page_width * page_height

        # Calculate the total area covered by the image
        image_area = 0

        # Traverse page elements
        for element in layout:
            # Check if it is an image or graphic element
            if isinstance(element, (LTImage, LTFigure)):
                # Calculate image bounding box area
                img_width = element.width
                img_height = element.height
                img_area = img_width * img_height
                image_area += img_area

        # Calculate coverage
        coverage_ratio = min(image_area / page_area, 1.0) if page_area > 0 else 0
        # logger.debug(f"PDFanalyze: page {page_count + 1} Image coverage: {coverage_ratio:.2f}")

        # Determine whether it is high coverage
        if coverage_ratio >= 0.8:  # Use 80% as the threshold for high coverage
            high_image_coverage_pages += 1

        page_count += 1

    # close resource
    pdf_stream.close()

    # If no page has been processed, returns 0
    if page_count == 0:
        return 0.0

    # Calculate proportion of pages with high image coverage
    high_coverage_ratio = high_image_coverage_pages / page_count
    # logger.debug(f"PDFanalyze: High image coverage page ratio: {high_coverage_ratio:.2f}")

    return high_coverage_ratio


def extract_pages(src_pdf_bytes: bytes) -> bytes:
    """
    Randomly extract up to 10 pages from PDF byte data and return new PDF byte data

    Args:
        src_pdf_bytes: PDFfile byte data

    Returns:
        bytes: PDF byte data after extracting the page
    """

    # Load PDF from byte data
    pdf = pdfium.PdfDocument(src_pdf_bytes)

    # Get the number of PDF pages
    total_page = len(pdf)
    if total_page == 0:
        # If the PDF has no pages, an empty document will be returned directly.
        logger.warning("PDF is empty, return empty document")
        return b''

    # Select up to 10 pages
    select_page_cnt = min(10, total_page)

    # Randomly select pages from the total number of pages
    page_indices = np.random.choice(total_page, select_page_cnt, replace=False).tolist()

    # Create a new PDF document
    sample_docs = pdfium.PdfDocument.new()

    try:
        # Import selected pages into a new document
        sample_docs.import_pages(pdf, page_indices)
        pdf.close()

        # Save new PDF to memory buffer
        output_buffer = BytesIO()
        sample_docs.save(output_buffer)

        # Get byte data
        return output_buffer.getvalue()
    except Exception as e:
        pdf.close()
        logger.exception(e)
        return b''  # Returns a null byte on error


def detect_invalid_chars(sample_pdf_bytes: bytes) -> bool:
    """"
    Detect if PDF contains illegal characters
    """
    '''pdfminerrelatively slow,You need to randomly select about 10 pages of samples first.'''
    # sample_pdf_bytes = extract_pages(src_pdf_bytes)
    sample_pdf_file_like_object = BytesIO(sample_pdf_bytes)
    laparams = LAParams(
        line_overlap=0.5,
        char_margin=2.0,
        line_margin=0.5,
        word_margin=0.1,
        boxes_flow=None,
        detect_vertical=False,
        all_texts=False,
    )
    text = extract_text(pdf_file=sample_pdf_file_like_object, laparams=laparams)
    text = text.replace("\n", "")
    # logger.info(text)
    '''The text features extracted from garbled text using pdfminer are (cid:xxx)'''
    cid_pattern = re.compile(r'\(cid:\d+\)')
    matches = cid_pattern.findall(text)
    cid_count = len(matches)
    cid_len = sum(len(match) for match in matches)
    text_len = len(text)
    if text_len == 0:
        cid_chars_radio = 0
    else:
        cid_chars_radio = cid_count/(cid_count + text_len - cid_len)
    # logger.debug(f"cid_count: {cid_count}, text_len: {text_len}, cid_chars_radio: {cid_chars_radio}")
    '''When more than 5% of the text in an article is garbled,The document is considered to be a garbled document'''
    if cid_chars_radio > 0.05:
        return True  # Garbled document
    else:
        return False   # normal document


if __name__ == '__main__':
    with open('/Users/myhloli/pdf/luanma2x10.pdf', 'rb') as f:
        p_bytes = f.read()
        logger.info(f"PDFClassification results: {classify(p_bytes)}")