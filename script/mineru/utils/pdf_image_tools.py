# Copyright (c) Opendatalab. All rights reserved.
import os
import signal
import time
from io import BytesIO

import numpy as np
import pypdfium2 as pdfium
from loguru import logger
from PIL import Image, ImageOps

from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.check_sys_env import is_windows_environment
from mineru.utils.os_env_config import get_load_images_timeout, get_load_images_threads
from mineru.utils.pdf_reader import image_to_b64str, image_to_bytes, page_to_image
from mineru.utils.enum_class import ImageType
from mineru.utils.hash_utils import str_sha256
from mineru.utils.pdf_page_id import get_end_page_id

from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED


def _trace_enabled() -> bool:
    return False


def _trace_stage(stage: str, **kwargs):
    return


def pdf_page_to_image(page: pdfium.PdfPage, dpi=200, image_type=ImageType.PIL) -> dict:
    """Convert pdfium.PdfDocument to image, Then convert the image to base64.

    Args:
        page (_type_): pdfium.PdfPage
        dpi (int, optional): reset the dpi of dpi. Defaults to 200.
        image_type (ImageType, optional): The type of image to return. Defaults to ImageType.PIL.

    Returns:
        dict:  {'img_base64': str, 'img_pil': pil_img, 'scale': float }
    """
    pil_img, scale = page_to_image(page, dpi=dpi)
    image_dict = {
        "scale": scale,
    }
    if image_type == ImageType.BASE64:
        image_dict["img_base64"] = image_to_b64str(pil_img)
    else:
        image_dict["img_pil"] = pil_img

    return image_dict


def _load_images_from_pdf_worker(
    pdf_bytes, dpi, start_page_id, end_page_id, image_type
):
    """Wrapper functions for process pools"""
    return load_images_from_pdf_core(
        pdf_bytes, dpi, start_page_id, end_page_id, image_type
    )


def load_images_from_pdf(
    pdf_bytes: bytes,
    dpi=200,
    start_page_id=0,
    end_page_id=None,
    image_type=ImageType.PIL,
    timeout=None,
    threads=None,
):
    """With timeout control PDF Convert image function,Support multi-process acceleration

    Args:
        pdf_bytes (bytes): PDF Documentary bytes
        dpi (int, optional): reset the dpi of dpi. Defaults to 200.
        start_page_id (int, optional): Starting page number. Defaults to 0.
        end_page_id (int | None, optional): end page number. Defaults to None.
        image_type (ImageType, optional): Picture type. Defaults to ImageType.PIL.
        timeout (int | None, optional): Timeout (seconds). if for None，Then from the environment variable MINERU_PDF_RENDER_TIMEOUT Read, if not set, defaults to 300 Second.
        threads (int): Number of processes, if for None，Then from the environment variable MINERU_PDF_RENDER_THREADS Read, if not set, defaults to 4.

    Raises:
        TimeoutError: Thrown when conversion times out
    """
    _trace_stage("pdf_image.load.begin", start=start_page_id, end=end_page_id)
    pdf_doc = pdfium.PdfDocument(pdf_bytes)

    if timeout is None:
        timeout = get_load_images_timeout()
    if threads is None:
        threads = get_load_images_threads()

    end_page_id = get_end_page_id(end_page_id, len(pdf_doc))

    # Calculate total number of pages
    total_pages = end_page_id - start_page_id + 1

    # The actual number of processes used does not exceed the total number of pages
    actual_threads = min(os.cpu_count() or 1, threads, total_pages)
    _trace_stage(
        "pdf_image.load.plan",
        total_pages=total_pages,
        threads=threads,
        actual_threads=actual_threads,
        timeout=timeout,
    )

    logger.debug(f"PDF to images using {actual_threads} processes, page ranges: {start_page_id}-{end_page_id}")

    if is_windows_environment() or actual_threads <= 1:
        # Windows Do not use multiple processes in the environment
        _trace_stage("pdf_image.load.single_process", actual_threads=actual_threads)
        return load_images_from_pdf_core(
            pdf_bytes,
            dpi,
            start_page_id,
            get_end_page_id(end_page_id, len(pdf_doc)),
            image_type,
        ), pdf_doc
    else:
        # Group page ranges based on actual number of processes
        pages_per_thread = max(1, total_pages // actual_threads)
        page_ranges = []
        for i in range(actual_threads):
            range_start = start_page_id + i * pages_per_thread
            if i == actual_threads - 1:
                # The last process handles all remaining pages
                range_end = end_page_id
            else:
                range_end = start_page_id + (i + 1) * pages_per_thread - 1

            page_ranges.append((range_start, range_end))

        _trace_stage("pdf_image.load.pool_ranges", ranges=len(page_ranges))

        executor = ProcessPoolExecutor(max_workers=actual_threads)
        try:
            # Submit all tasks
            futures = []
            future_to_range = {}
            for range_start, range_end in page_ranges:
                future = executor.submit(
                    _load_images_from_pdf_worker,
                    pdf_bytes,
                    dpi,
                    range_start,
                    range_end,
                    image_type,
                )
                futures.append(future)
                future_to_range[future] = range_start
            _trace_stage("pdf_image.load.pool_submitted", futures=len(futures))

            # use wait() Set a single global timeout
            done, not_done = wait(futures, timeout=timeout, return_when=ALL_COMPLETED)
            _trace_stage("pdf_image.load.pool_wait_done", done=len(done), not_done=len(not_done))

            # Check if there are any unfinished tasks (timeout situation)
            if not_done:
                # Timeout: Forcefully terminate all child processes
                _terminate_executor_processes(executor)
                pdf_doc.close()
                _trace_stage("pdf_image.load.pool_timeout", not_done=len(not_done))
                raise TimeoutError(f"PDF to images conversion timeout after {timeout}s")

            # All tasks completed, collect results
            all_results = []
            for future in futures:
                range_start = future_to_range[future]
                # Not needed here timeout，Because the task has been completed
                images_list = future.result()
                all_results.append((range_start, images_list))
            _trace_stage("pdf_image.load.pool_collected", chunks=len(all_results))

            # Sort and merge results by starting page number
            all_results.sort(key=lambda x: x[0])
            images_list = []
            for _, imgs in all_results:
                images_list.extend(imgs)

            _trace_stage("pdf_image.load.pool_merged", pages=len(images_list))

            return images_list, pdf_doc

        except Exception as e:
            # Make sure to clean up the child process when any exception occurs
            _terminate_executor_processes(executor)
            pdf_doc.close()
            _trace_stage("pdf_image.load.error", error=type(e).__name__)
            if isinstance(e, TimeoutError):
                raise
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            _trace_stage("pdf_image.load.pool_shutdown")

def load_image_from_pdf(
    pdf_bytes: bytes,
    pdf_doc,
    dpi=200,
    start_page_id=0,
    end_page_id=None,
    image_type=ImageType.PIL,
    timeout=None,
    threads=None,
):
    """With timeout control PDF Convert image function,Support multi-process acceleration

    Args:
        pdf_bytes (bytes): PDF Documentary bytes
        dpi (int, optional): reset the dpi of dpi. Defaults to 200.
        start_page_id (int, optional): Starting page number. Defaults to 0.
        end_page_id (int | None, optional): end page number. Defaults to None.
        image_type (ImageType, optional): Picture type. Defaults to ImageType.PIL.
        timeout (int | None, optional): Timeout (seconds). if for None，Then from the environment variable MINERU_PDF_RENDER_TIMEOUT Read, if not set, defaults to 300 Second.
        threads (int): Number of processes, if for None，Then from the environment variable MINERU_PDF_RENDER_THREADS Read, if not set, defaults to 4.

    Raises:
        TimeoutError: Thrown when conversion times out
    """
    _trace_stage("pdf_image.load_one.begin", start=start_page_id, end=end_page_id)
    if timeout is None:
        timeout = get_load_images_timeout()
    if threads is None:
        threads = get_load_images_threads()

    end_page_id = get_end_page_id(end_page_id, len(pdf_doc))

    # Calculate total number of pages
    total_pages = end_page_id - start_page_id + 1

    # The actual number of processes used does not exceed the total number of pages
    actual_threads = min(os.cpu_count() or 1, threads, total_pages)
    _trace_stage(
        "pdf_image.load_one.plan",
        total_pages=total_pages,
        threads=threads,
        actual_threads=actual_threads,
        timeout=timeout,
    )


    logger.debug(f"PDF to images using {actual_threads} processes, page ranges: {start_page_id}-{end_page_id}")

    if is_windows_environment() or actual_threads <= 1:
        # Windows Do not use multiple processes in the environment
        _trace_stage("pdf_image.load_one.single_process", actual_threads=actual_threads)
        return load_images_from_pdf_core(
            pdf_bytes,
            dpi,
            start_page_id,
            get_end_page_id(end_page_id, len(pdf_doc)),
            image_type,
        )
    else:
        # Group page ranges based on actual number of processes
        pages_per_thread = max(1, total_pages // actual_threads)
        page_ranges = []

        for i in range(actual_threads):
            range_start = start_page_id + i * pages_per_thread
            if i == actual_threads - 1:
                # The last process handles all remaining pages
                range_end = end_page_id
            else:
                range_end = start_page_id + (i + 1) * pages_per_thread - 1

            page_ranges.append((range_start, range_end))

        _trace_stage("pdf_image.load_one.pool_ranges", ranges=len(page_ranges))

        executor = ProcessPoolExecutor(max_workers=actual_threads)
        try:
            # Submit all tasks
            futures = []
            future_to_range = {}
            for range_start, range_end in page_ranges:
                future = executor.submit(
                    _load_images_from_pdf_worker,
                    pdf_bytes,
                    dpi,
                    range_start,
                    range_end,
                    image_type,
                )
                futures.append(future)
                future_to_range[future] = range_start
            _trace_stage("pdf_image.load_one.pool_submitted", futures=len(futures))

            # use wait() Set a single global timeout
            done, not_done = wait(futures, timeout=timeout, return_when=ALL_COMPLETED)
            _trace_stage("pdf_image.load_one.pool_wait_done", done=len(done), not_done=len(not_done))

            # Check if there are any unfinished tasks (timeout situation)
            if not_done:
                # Timeout: Forcefully terminate all child processes
                _terminate_executor_processes(executor)
                pdf_doc.close()
                _trace_stage("pdf_image.load_one.pool_timeout", not_done=len(not_done))
                raise TimeoutError(f"PDF to images conversion timeout after {timeout}s")

            # All tasks completed, collect results
            all_results = []
            for future in futures:
                range_start = future_to_range[future]
                # Not needed here timeout，Because the task has been completed
                images_list = future.result()
                all_results.append((range_start, images_list))
            _trace_stage("pdf_image.load_one.pool_collected", chunks=len(all_results))

            # Sort and merge results by starting page number
            all_results.sort(key=lambda x: x[0])
            images_list = []
            for _, imgs in all_results:
                images_list.extend(imgs)

            _trace_stage("pdf_image.load_one.pool_merged", pages=len(images_list))

            return images_list

        except Exception as e:
            # Make sure to clean up the child process when any exception occurs
            _terminate_executor_processes(executor)
            pdf_doc.close()
            _trace_stage("pdf_image.load_one.error", error=type(e).__name__)
            if isinstance(e, TimeoutError):
                raise
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            _trace_stage("pdf_image.load_one.pool_shutdown")

def _terminate_executor_processes(executor):
    """Forced termination ProcessPoolExecutor all child processes in"""
    _trace_stage("pdf_image.terminate.begin")
    if hasattr(executor, '_processes'):
        _trace_stage("pdf_image.terminate.process_count", count=len(executor._processes))
        for pid, process in executor._processes.items():
            if process.is_alive():
                try:
                    # Send first SIGTERM Allow graceful exit
                    os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass

        # Give the child process some time to respond SIGTERM
        time.sleep(0.1)

        # Sent to still alive processes SIGKILL Forced termination
        for pid, process in executor._processes.items():
            if process.is_alive():
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
    _trace_stage("pdf_image.terminate.end")


def load_images_from_pdf_core(
    pdf_bytes: bytes,
    dpi=200,
    start_page_id=0,
    end_page_id=None,
    image_type=ImageType.PIL,  # PIL or BASE64
):
    _trace_stage("pdf_image.core.begin", start=start_page_id, end=end_page_id)
    images_list = []
    pdf_doc = pdfium.PdfDocument(pdf_bytes)
    pdf_page_num = len(pdf_doc)
    end_page_id = get_end_page_id(end_page_id, pdf_page_num)

    for index in range(start_page_id, end_page_id + 1):
        # logger.debug(f"Converting page {index}/{pdf_page_num} to image")
        page = pdf_doc[index]
        try:
            image_dict = pdf_page_to_image(page, dpi=dpi, image_type=image_type)
            images_list.append(image_dict)
        finally:
            try:
                page.close()
            except Exception:
                pass

        if _trace_enabled() and ((index - start_page_id + 1) % 10 == 0):
            _trace_stage("pdf_image.core.progress", done=index - start_page_id + 1)

    pdf_doc.close()

    _trace_stage("pdf_image.core.end", pages=len(images_list))

    return images_list


def cut_image(
    bbox: tuple,
    page_num: int,
    page_pil_img,
    return_path,
    image_writer: FileBasedDataWriter,
    scale=2,
):
    """From the page of page_num, crop a jpg picture according to the bbox and return the picture path. save_path：Need to support both s3 and local,
    The picture is stored under save_path, and the file name is:
    {page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg , bboxInternal numbers are rounded."""

    # Splice file names
    filename = f"{page_num}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"

    # The old version returns the path without bucket
    img_path = f"{return_path}_{filename}" if return_path is not None else None

    # New version generates tile path
    img_hash256_path = f"{str_sha256(img_path)}.jpg"
    # img_hash256_path = f'{img_path}.jpg'

    crop_img = get_crop_img(bbox, page_pil_img, scale=scale)

    img_bytes = image_to_bytes(crop_img, image_format="JPEG")

    image_writer.write(img_hash256_path, img_bytes)
    return img_hash256_path


def get_crop_img(bbox: tuple, pil_img, scale=2):
    scale_bbox = (
        int(bbox[0] * scale),
        int(bbox[1] * scale),
        int(bbox[2] * scale),
        int(bbox[3] * scale),
    )
    return pil_img.crop(scale_bbox)


def get_crop_np_img(bbox: tuple, input_img, scale=2):
    if isinstance(input_img, Image.Image):
        np_img = np.asarray(input_img)
    elif isinstance(input_img, np.ndarray):
        np_img = input_img
    else:
        raise ValueError("Input must be a pillow object or a numpy array.")

    scale_bbox = (
        int(bbox[0] * scale),
        int(bbox[1] * scale),
        int(bbox[2] * scale),
        int(bbox[3] * scale),
    )

    return np_img[scale_bbox[1] : scale_bbox[3], scale_bbox[0] : scale_bbox[2]]


def images_bytes_to_pdf_bytes(image_bytes):
    # memory buffer
    pdf_buffer = BytesIO()

    # Load and convert all images to RGB model
    image = Image.open(BytesIO(image_bytes))
    # according to EXIF Information is automatically corrected (processing the tapes taken by mobile phones) Orientation tagged pictures)
    image = ImageOps.exif_transpose(image) or image
    # Convert only when necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save the first picture as PDF，The rest are appended
    image.save(
        pdf_buffer,
        format="PDF",
        # save_all=True
    )

    # get PDF bytes and reset the pointer (optional)
    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()
    return pdf_bytes
