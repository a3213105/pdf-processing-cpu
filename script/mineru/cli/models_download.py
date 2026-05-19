import json
import os
import sys
import click
import requests
from loguru import logger

from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path, auto_download_and_get_model_root_path_2_local


def download_json(url):
    """Download JSON file"""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def download_and_modify_json(url, local_filename, modifications):
    """Download JSON and modify content"""
    if os.path.exists(local_filename):
        data = json.load(open(local_filename))
        config_version = data.get('config_version', '0.0.0')
        if config_version < '1.3.1':
            data = download_json(url)
    else:
        data = download_json(url)

    # Modify content
    for key, value in modifications.items():
        if key in data:
            if isinstance(data[key], dict):
                # If it is a dictionary, merge the new values
                data[key].update(value)
            else:
                # Otherwise, replace directly
                data[key] = value

    # Save modified content
    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def configure_model(model_dir, model_type, config_file_path=None):
    """Configure model"""
    json_url = 'https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/mineru.template.json'
    if config_file_path:
        config_file = os.path.abspath(os.path.expanduser(config_file_path))
    else:
        config_file_name = os.getenv('MINERU_TOOLS_CONFIG_JSON', './mineru.json')
        if os.path.isabs(config_file_name):
            config_file = config_file_name
        elif config_file_name.startswith('./') or config_file_name.startswith('../'):
            config_file = os.path.abspath(os.path.expanduser(config_file_name))
        else:
            home_dir = os.path.expanduser('~')
            config_file = os.path.join(home_dir, config_file_name)

    os.makedirs(os.path.dirname(config_file), exist_ok=True)

    json_mods = {
        'models-dir': {
            f'{model_type}': model_dir
        }
    }

    download_and_modify_json(json_url, config_file, json_mods)
    logger.info(f'The configuration file has been successfully configured, the path is: {config_file}')


def download_pipeline_models(outputs, config_file_path=None):
    """Download Pipeline model"""
    model_paths = [
        ModelPath.doclayout_yolo,
        ModelPath.yolo_v8_mfd,
        ModelPath.unimernet_small,
        ModelPath.pytorch_paddle,
        ModelPath.layout_reader,
        ModelPath.slanet_plus,
        ModelPath.unet_structure,
        ModelPath.paddle_table_cls,
        ModelPath.paddle_orientation_classification,
        ModelPath.pp_formulanet_plus_m,
    ]
    download_finish_path = ""
    for model_path in model_paths:
        logger.info(f"Downloading model: {model_path}")
        download_finish_path = auto_download_and_get_model_root_path_2_local(model_path, repo_mode='pipeline', cache_dir=outputs)
    if not os.path.isabs(download_finish_path):
        download_finish_path = os.path.abspath(download_finish_path)
    logger.info(f"Pipeline models downloaded successfully to: {download_finish_path}")
    configure_model(download_finish_path, "pipeline", config_file_path)


def download_vlm_models(config_file_path=None):
    """Download VLM model"""
    download_finish_path = auto_download_and_get_model_root_path("/", repo_mode='vlm')
    if not os.path.isabs(download_finish_path):
        download_finish_path = os.path.abspath(download_finish_path)
    logger.info(f"VLM models downloaded successfully to: {download_finish_path}")
    configure_model(download_finish_path, "vlm", config_file_path)


@click.command()
@click.option(
    '-s',
    '--source',
    'model_source',
    type=click.Choice(['huggingface', 'modelscope']),
    help="""
        The source of the model repository. 
        """,
    default='modelscope',
)
@click.option(
    '-m',
    '--model_type',
    'model_type',
    type=click.Choice(['pipeline', 'vlm', 'all']),
    help="""
        The type of the model to download.
        """,
    default='pipeline',
)
@click.option(
    '-o',
    '--outputs',
    'outputs',
    type=click.Path(),
    help="""
        The output directory for the downloaded models.
        """,
    default='./models_cache'
)
@click.option(
    '--config',
    '-c',
    'config_file_path',
    type=click.Path(),
    help="""
        The output file path for mineru.json.
        """,
    default='./mineru.json',
)

def download_models(model_source, model_type, outputs, config_file_path):
    """Download MinerU model files.

    Supports downloading pipeline or VLM models from ModelScope or HuggingFace.
    """
    # Interactively enter the download source if not explicitly specified
    if model_source is None:
        model_source = click.prompt(
            "Please select the model download source: ",
            type=click.Choice(['huggingface', 'modelscope']),
            default='modelscope'
        )

    # if os.getenv('MINERU_MODEL_SOURCE', None) is None:
    #     os.environ['MINERU_MODEL_SOURCE'] = model_source

    # Enter model type interactively if not specified explicitly
    if model_type is None:
        model_type = click.prompt(
            "Please select the model type to download: ",
            type=click.Choice(['pipeline', 'vlm', 'all']),
            default='pipeline'
        )

    if outputs is None:
        outputs = click.prompt(
            "Please specify the output directory: ",
            type=click.Path(),
            default='./models_cache'
        )

    logger.info(f"Downloading {model_type} model from {model_source} to {outputs} ...")

    try:
        if model_type == 'pipeline':
            download_pipeline_models(outputs, config_file_path)
        elif model_type == 'vlm':
            download_vlm_models(config_file_path)
        elif model_type == 'all':
            download_pipeline_models(outputs, config_file_path)
            download_vlm_models(config_file_path)
        else:
            click.echo(f"Unsupported model type: {model_type}", err=True)
            sys.exit(1)

    except Exception as e:
        logger.exception(f"An error occurred while downloading models: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    download_models()
