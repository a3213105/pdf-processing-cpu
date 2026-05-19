import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDF processing entry")
    parser.add_argument("--disable-ov", "-o", action="store_true", default=False, help="disable_ov")
    parser.add_argument("--layout-type", type=str, default="bf16", help="layout detection infer type")
    parser.add_argument("--mfd-type", type=str, default="bf16", help="formula detection infer type")
    parser.add_argument("--mfr-enc-type", type=str, default="bf16", help="formula recognition enc infer type")
    parser.add_argument("--mfr-dec-type", type=str, default="bf16", help="formula recognition dec infer type")
    parser.add_argument("--ocr-det-type", type=str, default="bf16", help="ocr detection infer type")
    parser.add_argument("--ocr-rec-type", type=str, default="bf16", help="ocr recognition infer type")
    parser.add_argument("--wired-table-type", type=str, default="bf16", help="wired table infer type")
    parser.add_argument("--wireless-table-type", type=str, default="bf16", help="wireless table infer type")
    parser.add_argument("--table-cls-type", type=str, default="bf16", help="page layout infer type")
    parser.add_argument("--img-cls-type", type=str, default="bf16", help="image orientation classification infer type")
    parser.add_argument("--layoutreader-type", type=str, default="bf16", help="page layout infer type")
    parser.add_argument("--all", "-a", type=str, default=None, help="set all infer type")
    parser.add_argument("--input", "-i", metavar="INPUT", nargs="+", help="Filenames of input pdfs")
    parser.add_argument("--nstreams", "-n", type=int, default=1, help="Number of ov streams")
    parser.add_argument("--app", "-p", action="store_true", default=False, help="True for app mode (offline run)")
    parser.add_argument("--benchmark", action="store_true", default=False, help="Enable benchmark in app mode only")
    parser.add_argument("--repeat", "-r", type=int, default=20, help="Number of measured benchmark rounds")
    parser.add_argument("--warmup", "-w", type=int, default=3, help="Number of warmup rounds")
    parser.add_argument("--benchmark-json", type=str, default=None, help="Optional benchmark summary output path")
    parser.add_argument("--enable-json", dest="enable_json", action="store_true", default=False, help="enable json output")
    parser.add_argument("--disable-json", "-j", dest="enable_json", action="store_false", help="disable json output (default)")
    parser.add_argument("--disable-cache", "-c", dest="enable_cache", default=True, action="store_false", help="disable caching")
    parser.add_argument("--config", type=str, default="./mineru.json", help="Path to mineru.json configuration file")
    parser.add_argument("--verify", "-v", action="store_true", default=False, help="Verify installation only")
    return parser.parse_args()


def normalize_infer_types(args: argparse.Namespace) -> argparse.Namespace:
    if args.all is not None:
        value = args.all.lower()
        args.all = value
        args.layout_type = value
        args.mfd_type = value
        args.mfr_enc_type = value
        args.mfr_dec_type = value
        args.ocr_det_type = value
        args.ocr_rec_type = value
        args.wired_table_type = value
        args.wireless_table_type = value
        args.img_cls_type = value
        args.table_cls_type = value
        args.layoutreader_type = value
        return args

    args.layout_type = args.layout_type.lower()
    args.mfd_type = args.mfd_type.lower()
    args.mfr_enc_type = args.mfr_enc_type.lower()
    args.mfr_dec_type = args.mfr_dec_type.lower()
    args.ocr_det_type = args.ocr_det_type.lower()
    args.ocr_rec_type = args.ocr_rec_type.lower()
    args.wired_table_type = args.wired_table_type.lower()
    args.wireless_table_type = args.wireless_table_type.lower()
    args.img_cls_type = args.img_cls_type.lower()
    args.table_cls_type = args.table_cls_type.lower()
    args.layoutreader_type = args.layoutreader_type.lower()
    return args
