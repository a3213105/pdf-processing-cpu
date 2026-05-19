import os
from huggingface_hub import snapshot_download as hf_snapshot_download
from modelscope import snapshot_download as ms_snapshot_download

from mineru.utils.config_reader import get_local_models_dir
from mineru.utils.enum_class import ModelPath

def auto_download_and_get_model_root_path_2_local(relative_path: str, repo_mode='pipeline', model_source="modelscope", cache_dir=None) -> str:
    if model_source == 'local':
        local_models_config = get_local_models_dir()
        root_path = local_models_config.get(repo_mode, None)
        if not root_path:
            raise ValueError(f"Local path for repo_mode '{repo_mode}' is not configured.")
        return root_path

    # Establish a mapping of warehouse schema to paths
    repo_mapping = {
        'pipeline': {
            'huggingface': ModelPath.pipeline_root_hf,
            'modelscope': ModelPath.pipeline_root_modelscope,
            'default': ModelPath.pipeline_root_hf
        },
        'vlm': {
            'huggingface': ModelPath.vlm_root_hf,
            'modelscope': ModelPath.vlm_root_modelscope,
            'default': ModelPath.vlm_root_hf
        }
    }

    if repo_mode not in repo_mapping:
        raise ValueError(f"Unsupported repo_mode: {repo_mode}, must be 'pipeline' or 'vlm'")

    # If model_source is not specified or the value is not'modelscope'，then use the default value
    repo = repo_mapping[repo_mode].get(model_source, repo_mapping[repo_mode]['default'])


    if model_source == "huggingface":
        snapshot_download = hf_snapshot_download
    elif model_source == "modelscope":
        snapshot_download = ms_snapshot_download
    else:
        raise ValueError(f"Unknown warehouse type: {model_source}")


    if repo_mode == 'pipeline':
        relative_path = relative_path.strip('/')
        cache_dir = snapshot_download(repo, cache_dir=cache_dir, allow_patterns=[relative_path, relative_path+"/*"])
    elif repo_mode == 'vlm':
        # VLM mode, according to relative_path different ways of handling
        if relative_path == "/":
            cache_dir = snapshot_download(repo, cache_dir=cache_dir)
        else:
            relative_path = relative_path.strip('/')
            cache_dir = snapshot_download(repo, cache_dir=cache_dir, allow_patterns=[relative_path, relative_path+"/*"])

    if not cache_dir:
        raise FileNotFoundError(f"Failed to download model: {relative_path} from {repo}")
    return cache_dir


def auto_download_and_get_model_root_path(relative_path: str, repo_mode='pipeline') -> str:
    """
    Supports reliable downloading of files or directories.
    - If the input file: Returns the absolute path of the local file
    - If you enter a directory: Return to local cache with relative_path Relative path string with the same structure
    :param repo_mode: Specify the warehouse mode,'pipeline' or 'vlm'
    :param relative_path: File or directory relative path
    :return: Absolute or relative path to local file
    """
    model_source = os.getenv('MINERU_MODEL_SOURCE', "huggingface")

    if model_source == 'local':
        local_models_config = get_local_models_dir()
        root_path = local_models_config.get(repo_mode, None)
        if not root_path:
            raise ValueError(f"Local path for repo_mode '{repo_mode}' is not configured.")
        return root_path

    # Establish a mapping of warehouse schema to paths
    repo_mapping = {
        'pipeline': {
            'huggingface': ModelPath.pipeline_root_hf,
            'modelscope': ModelPath.pipeline_root_modelscope,
            'default': ModelPath.pipeline_root_hf
        },
        'vlm': {
            'huggingface': ModelPath.vlm_root_hf,
            'modelscope': ModelPath.vlm_root_modelscope,
            'default': ModelPath.vlm_root_hf
        }
    }

    if repo_mode not in repo_mapping:
        raise ValueError(f"Unsupported repo_mode: {repo_mode}, must be 'pipeline' or 'vlm'")

    # If model_source is not specified or the value is not'modelscope'，then use the default value
    repo = repo_mapping[repo_mode].get(model_source, repo_mapping[repo_mode]['default'])


    if model_source == "huggingface":
        snapshot_download = hf_snapshot_download
    elif model_source == "modelscope":
        snapshot_download = ms_snapshot_download
    else:
        raise ValueError(f"Unknown warehouse type: {model_source}")

    cache_dir = None

    if repo_mode == 'pipeline':
        relative_path = relative_path.strip('/')
        cache_dir = snapshot_download(repo, allow_patterns=[relative_path, relative_path+"/*"])
    elif repo_mode == 'vlm':
        # VLM mode, according to relative_path different ways of handling
        if relative_path == "/":
            cache_dir = snapshot_download(repo)
        else:
            relative_path = relative_path.strip('/')
            cache_dir = snapshot_download(repo, allow_patterns=[relative_path, relative_path+"/*"])

    if not cache_dir:
        raise FileNotFoundError(f"Failed to download model: {relative_path} from {repo}")
    return cache_dir


if __name__ == '__main__':
    path1 = "models/README.md"
    cache_dir = "./models_cache"
    root =  auto_download_and_get_model_root_path_2_local(path1, cache_dir=cache_dir)
    print("Absolute path to local file:", os.path.join(root, path1))