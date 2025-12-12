import torch
import sys
import os
import copy
import yaml
import gc
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

import utils.load_config as training
from model import FullFace_pl


if __name__ == "__main__":
    mode = "train"
    
    config_path = f"{project_root}/src/config.yaml"

    full_config, config, dataloader, savepath, imagepath, labelpath, modelname, bs, nw, device = training.script_init_common(mode, config_path)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

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
    print(full_config["train"]["params"]["epoch"])
    
    with open(os.path.join(savepath, "config.yaml"), "w") as file:
        yaml.dump(full_config, file)

    model =  FullFace_pl(full_config["train"]["params"]["lr"], full_config["img_size"])
    trainer = pl.Trainer(precision=16, accelerator="gpu", devices=1, logger=TensorBoardLogger(save_dir=f'{project_root}/outputs/tensorboard_logs/{mode}_FullFace_{full_config["dataset"]}', name=""), 
                            max_epochs=full_config["train"]["params"]["epoch"], enable_progress_bar=True,
                            callbacks=[pl.callbacks.LearningRateMonitor("step"), 
                                        pl.callbacks.ModelCheckpoint(dirpath=savepath, filename="{epoch}-{val_ang_error:.2f}-{train_ang_error:.2f}", every_n_epochs = 1, save_top_k=-1)])

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    gc.collect()
    torch.cuda.empty_cache()