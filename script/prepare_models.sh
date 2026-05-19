#！/bin/bash
# Usage: prepare_models.sh <output_dir> <source>
# <output_dir>: Directory to save the downloaded models.
# <source>: Model source, e.g., "huggingface" or "modelscope".
OUTPUT=$1
SOURCE=$2 
python -m mineru.cli.models_download -m pipeline -s $SOURCE -o $OUTPUT
python -m mineru.main --init
