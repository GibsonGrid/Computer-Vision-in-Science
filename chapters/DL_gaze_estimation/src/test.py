import torch
import sys
import os
import copy
import yaml
import gc
import glob
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

import utils.load_config as training
from model import FullFace_pl


if __name__ == "__main__":
    print(os.getcwd())
    mode = "test"
    
    config_path = f"{project_root}/src/config.yaml"

    full_config, config, dataloader, loadpath, imagepath, labelpath, modelname, bs, nw, device = training.script_init_common(mode, config_path)

    i = full_config["leaveout_folder"]
    folder = os.listdir(labelpath)
    folder.sort()

    if i in list(range(len(folder))):
        trains = copy.deepcopy(folder)
        tests = trains.pop(i)
        
        vals = trains[int(len(folder)*0.7):]
        trains = trains[:int(len(folder)*0.7)]

        print(f"Train Set: {trains}")
        print(f"Val Set: {vals}")
        print(f"Test Set: {tests}")

        trainlabelpath = [os.path.join(labelpath, j) for j in trains] 
        vallabelpath = [os.path.join(labelpath, j) for j in vals] 
        testlabelpath = os.path.join(labelpath, tests)  
        
        train_dataset, train_loader = dataloader.create_dataset(full_config, trainlabelpath, imagepath, shuffle=True, num_workers=nw)
        val_dataset, val_loader = dataloader.create_dataset(full_config, vallabelpath, imagepath, shuffle=False, num_workers=nw)
        test_dataset, test_loader = dataloader.create_dataset(full_config, testlabelpath, imagepath, shuffle=False, num_workers=nw)

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    all_files = glob.glob(f"{loadpath}/*.ckpt")
    pt_files = sorted([f for f in all_files if f.endswith(".ckpt")])
    print(pt_files)
    latest_checkpoint = pt_files[-1]
    
    model =  FullFace_pl.load_from_checkpoint(latest_checkpoint)
    
    trainer = pl.Trainer(logger=TensorBoardLogger(save_dir=f'{project_root}/outputs/tensorboard_logs/{mode}_FullFace_{full_config["dataset"]}', name=""), enable_progress_bar=True)

    trainer.test(model, dataloaders=test_loader)   
    gc.collect()
    torch.cuda.empty_cache() 