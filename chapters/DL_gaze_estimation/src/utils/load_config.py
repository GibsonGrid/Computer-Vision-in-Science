import os
import sys
import importlib
import numpy as np
import torch
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set environment variables for path expansion
    os.environ["HOME_DIR"] = config.get("home_dir", "")
    os.environ["EXPERIMENT_NAME"] = config.get("experiment_name", "")
    
    # Function to expand environment variables
    def expand_vars(obj):
        if isinstance(obj, dict):
            return {k: expand_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_vars(item) for item in obj]
        elif isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj
    
    return expand_vars(config)

def script_init_common(mode, config_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config_dir = os.path.dirname(os.path.abspath(config_path))
    project_root = os.path.abspath(os.path.join(config_dir, ".."))

    with open(config_path, 'r') as f:
        config_content = f.read()

    project_root = project_root.replace('\\', '/') # for Windows
    config_content = config_content.replace("{PROJECT_ROOT}", project_root)

    os.environ["PROJECT_ROOT"] = project_root
    config = yaml.safe_load(config_content)

    full_config = config
    
    readername = config["dataloader"]
    bs = config["batch_size"]
    nw = config["num_workers"]
    dataloader = importlib.import_module(readername)

    torch.manual_seed(0xC0FFEE)
    torch.cuda.manual_seed(0xC0FFEE)
    if config["fully_reproducible"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(0xC0FFEE)
    
    dataset_name = config["dataset"]
    
    if mode == "train":
        config = config["train"]
        imagepath = config["data"]["image"]
        labelpath = config["data"]["label"]
        
        modelname = config["save"]["model_name"]
        savepath = config["save"]["save_path"]
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        return full_config, config, dataloader, savepath, imagepath, labelpath, modelname, bs, nw, device
        
    elif mode == "test":
        config = config["test"]
        imagepath = config["data"]["image"]
        labelpath = config["data"]["label"]
        
        modelname = config["load"]["model_name"]
        loadpath = config["load"]["load_path"]
        
        return full_config, config, dataloader, loadpath, imagepath, labelpath, modelname, bs, nw, device