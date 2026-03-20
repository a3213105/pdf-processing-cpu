import json
import os
os.environ["YOLO_VERBOSE"] = 'False'
from pathlib import Path
from pdf_rec import PDF_Instance, process_pdf_file
from utils import parse_args



def main(args, pdf_instance):
    output_md_list = process_pdf_file(pdf_instance, args.input, args.return_md, args.return_json, args.return_layout, args.return_span, args.output_dir)
    outputs = []
    output_info = f"Processed {len(output_md_list)} PDF files successfully."
    for input_name, output_md_filename, *_ in output_md_list:
        outputs.append({
            "input_name": str(input_name),
            "output_path": str(output_md_filename),
            })

    return json.dumps({
            "success": True,
            "message": output_info,
            "outputs": outputs,
            }, ensure_ascii=False, default=str)

if __name__ == '__main__':
    # try :
        args = parse_args()
        if args.all is not None :
            args.all = args.all.lower()
            args.layout_type = args.all
            args.mfd_type = args.all
            args.mfr_enc_type = args.all
            args.mfr_dec_type = args.all
            args.ocr_det_type = args.all
            args.ocr_rec_type = args.all
            args.table_type = args.all
            args.lang_type = args.all
            args.page_type = args.all
        else :
            args.layout_type = args.layout_type.lower()
            args.mfd_type = args.mfd_type.lower()
            args.mfr_enc_type = args.mfr_enc_type.lower()
            args.mfr_dec_type = args.mfr_dec_type.lower()
            args.ocr_det_type = args.ocr_det_type.lower()
            args.ocr_rec_type = args.ocr_rec_type.lower()
            args.table_type = args.table_type.lower()
            args.lang_type = args.lang_type.lower()
            args.page_type = args.page_type.lower()
        pdf_instance = PDF_Instance(args)
        if args.verify :
            output_json_str = json.dumps([{
                "success": True,
                "message": "Installation Verified Successfully",
                "outputs": [],
            }])
        else :
            output_json_str = main(args, pdf_instance)
    # except Exception as e:
    #     output_json_str = json.dumps([{
    #         "success": False,
    #         "message": str(e),
    #         "outputs": [],
    #     }])
    # print(output_json_str)