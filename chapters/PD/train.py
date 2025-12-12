#https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
import os
import time
from glob import glob
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import customDataset
from model import build_unet, UNet, get_model, EarlyStopping
from utils import append2csv, seeding, create_dir, epoch_time, get_images_masks_paths
import pathlib
from tqdm import tqdm
import numpy as np
from PIL import Image



import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd


import segmentation_models_pytorch




def train_one_epoch(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0


    model.train()
    for idx, (x, y) in enumerate(tqdm(loader)):
        # print(x)
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        y_pred = model(x)
        optimizer.zero_grad()
        loss = loss_fn(y_pred, y)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(len(loader), idx + 1)
    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def validate_one_epoch(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

	    #acc = 
	    #epoch_acc += acc

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss




def create_parser_disease_model_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATASET_NAME', type=str,)
    parser.add_argument('--checkpoint_path_w', type=str, default = "")
    parser.add_argument('--checkpoint_path_full', type=str, default = "")

    parser.add_argument('--MODEL_ARCH', type=str)
    parser.add_argument('--ENCODER_NAME', type=str)

    parser.add_argument('--W', type=int)
    parser.add_argument('--H', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--CLASSES_NUM', type=int)

    parser.add_argument('--N_SAMPLES', type=float)
    parser.add_argument('--USE_AUGMENTED', type=int)

    


    #---------------------------MODIFY
    return parser

if __name__ == "__main__":
    parser = create_parser_disease_model_train()
    args = parser.parse_args()
    

    DATASET_NAME = args.DATASET_NAME
    checkpoint_path_w = args.checkpoint_path_w
    checkpoint_path_full = args.checkpoint_path_full
    MODEL_ARCH = args.MODEL_ARCH
    ENCODER_NAME = args.ENCODER_NAME
    W = args.W
    H = args.H

    CLASSES_NUM = int(args.CLASSES_NUM)
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    N_SAMPLES = args.N_SAMPLES
    USE_AUGMENTED = args.USE_AUGMENTED
    #---------------------------MODIFY



    from datetime import datetime

    now_str = datetime.now().strftime("%Y-%m-%d %H_%M_%S")


    """ Seeding """
    seeding(42)
    where = "Results"
    create_dir(f"{where}")

    subfolders = [ f.path for f in os.scandir(where) if f.is_dir() ]
    where2save = f"{where}//Exp{len(subfolders)+1}_{now_str}"

    create_dir(f"{where2save}")
    """ Directories """
    checkpoint_path = f"{where2save}//checkpoint.pth"


    size = (H, W)
    #---------------------------MODIFY
    # Load eye segmentation dataset
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_images_masks_paths(root_dir=f"data/{DATASET_NAME}", sefics="_aug" if USE_AUGMENTED else "")


    print("Before wanted partition: ", np.array(train_x).shape, np.array(train_y).shape, np.array(valid_x).shape, np.array(valid_y).shape, np.array(test_x).shape, np.array(test_y).shape)
    train_x, train_y = train_x[0:round(len(train_x)*N_SAMPLES)], train_y[0:round(len(train_y)*N_SAMPLES)]
    valid_x, valid_y = valid_x[0:round(len(valid_x)*N_SAMPLES)], valid_y[0:round(len(valid_y)*N_SAMPLES)]
    test_x, test_y = test_x[0:round(len(test_x)*N_SAMPLES)], test_y[0:round(len(test_y)*N_SAMPLES)]
    print("After wanted partition: ", np.array(train_x).shape, np.array(train_y).shape, np.array(valid_x).shape, np.array(valid_y).shape, np.array(test_x).shape, np.array(test_y).shape)
    

    """ Dataset and loader """
    train_dataset = customDataset(train_x, train_y, size)
    valid_dataset = customDataset(valid_x, valid_y, size)

    # generator =  torch.Generator().manual_seed(42)
    # train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2], generator = generator)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=2
    )

    
    # for x, y in valid_loader:
    #     print(x.shape, y.shape)
    epoch_init = 0
    train_loss_history = []

    val_loss_history = []

    early_stopping = EarlyStopping(patience=5, delta=0.01)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("----------------------------------------------------------------------",device)
    if 1:
        #---------------------------MODIFY
        
        model = get_model(MODEL_ARCH, ENCODER_NAME, 1)
        # if checkpoint_path_w != "":
        #     model.load_state_dict(torch.load(checkpoint_path_w, map_location=device))


        # print("Weights are loaded............")
        # model = UNet(3, 2)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-4, eps=1e-08, verbose='deprecated')


        if checkpoint_path_full != "":
            print("-----------------", checkpoint_path_full)
            checkpoint_obj = torch.load(f"{checkpoint_path_full}/checkpoint_full.pt", map_location=device)
            model.load_state_dict(checkpoint_obj['model_state_dict'])
            model.to(device)
            optimizer.load_state_dict(checkpoint_obj['optimizer_state_dict'])
            epoch_init = checkpoint_obj['epoch'] + 1

            print('epoch_init: ', epoch_init)
            # loss = checkpoint_obj['loss']
            train_loss_history = list(np.load(f"{checkpoint_path_full}/train_loss_history.npy"))
            val_loss_history = list(np.load(f"{checkpoint_path_full}/val_loss_history.npy"))
            print("_______", train_loss_history)
            epoch_for_best = checkpoint_obj['epoch_for_best']
            early_stopping_best = checkpoint_obj['early_stopping_best']
            # optimizer, 'min', patience=5, verbose=True)
        # loss_fn = nn.BCEWithLogitsLoss()
        from loss import DiceLoss, DiceBCELoss

        loss_fn = DiceBCELoss()
        # loss_fn = segmentation_models_pytorch.losses.MCCLoss(eps=1e-05)
        # loss_fn = segmentation_models_pytorch.losses.LovaszLoss('binary', per_image=False, ignore_index=None, from_logits=True)
        loss_fn = segmentation_models_pytorch.losses.FocalLoss('binary', alpha=None, gamma=2.0, ignore_index=None, reduction='mean', normalized=False, reduced_threshold=None)
        
    # else:
    #     checkpoint = torch.load( '/content/drive/MyDrive/checkpoint.pt')
    #     print(checkpoint)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     model.to(device)
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     epoch = checkpoint['epoch']
    #     loss = checkpoint['loss']


    print(len(train_x), len(train_y), len(valid_x), len(valid_y), len(test_x), len(test_y))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Training the model """
    best_valid_loss = float("inf")

    epoch_for_best = -1
    early_stopping_best = -1
    # early_stopping = EarlyStopping(tolerance=5, min_delta=10)
    epoch = epoch_init
    # for epoch in tqdm(range(epoch_init, num_epochs)):
    for epoch in tqdm(range(num_epochs)):

        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_loss = validate_one_epoch(model, valid_loader, loss_fn, device)


        train_loss_history.append(train_loss)

        val_loss_history.append(valid_loss)


        """ Saving the model """
        if valid_loss < best_valid_loss:
            epoch_for_best = epoch
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(model.state_dict(), "checkpoint.pth")
        

        # early_stopping(valid_loss, model)
        # if 0:#early_stopping.early_stop:
        #     print("...................................................Early stopping")
        #     early_stopping_best = epoch
        #     # break
# 
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)

        if epoch !=0 and epoch%3 == 0:
            res = eval_(test_x, test_y, checkpoint_path, (MODEL_ARCH, ENCODER_NAME), model, where2save + f"\\samples {epoch}",size ,test_yd = valid_yd if DATASET_NAME == "MRI" else None)
            res = eval_([test_x[0]], [test_y[0]], checkpoint_path, (MODEL_ARCH, ENCODER_NAME), model, where2save + f"\\samples {epoch}",size ,test_yd = valid_yd if DATASET_NAME == "MRI" else None)

    # early stopping
    # early_stopping(train_loss, valid_loss)
    # if early_stopping.early_stop:
    #   print("We are at epoch:", i)
    #   break
       
    np.save(f"{where2save}//train_loss_history.npy", np.array(train_loss_history))
    np.save(f"{where2save}//val_loss_history.npy", np.array(val_loss_history))
    torch.save(model.state_dict(), checkpoint_path)


    # save model
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),
                'epoch_for_best':epoch_for_best, 'early_stopping_best': early_stopping_best,
                }, f'{where2save}/checkpoint_full.pt')


    fig, axs = plt.subplots(1, 1, layout='constrained')
    # axs.plot(range(1,num_epochs+1), train_loss_history, label = "train")
    # axs.plot(range(1,num_epochs+1), val_loss_history, label = "val")
    print(len(train_loss_history), train_loss_history)
    axs.plot(list(range(len(train_loss_history))), train_loss_history, label = "train")
    axs.plot(list(range(len(val_loss_history))), val_loss_history, label = "val")
    # if early_stopping_best != -1 :
    #     plt.axvline(x = early_stopping_best, color = 'g', linestyle='-.' , linewidth=1, label = 'Early stopping');
    # axs.set_xlim(0, 2)
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Loss')
    axs.grid(True)
    plt.xticks(list(range(len(val_loss_history))))
    axs.set_title("Loss")
    plt.legend()
    plt.savefig(f"{where2save}//Loss.png")
    
    # os.system(F'python test.py --checkpoint_path {checkpoint_path} --where2save {where2save}')



    # res = eval_(test_x, test_y, checkpoint_path, (MODEL_ARCH, ENCODER_NAME), None, where2save + "\\samples",size ,test_yd = valid_yd if DATASET_NAME == "MRI" else None)
    # res.to_csv(f"{where2save}//metrics.csv")

    res["Exp"] = len(subfolders)+1
    res["Dataset"] = DATASET_NAME
    res['lr'] = lr
    res['num_epochs'] = num_epochs
    res["MODEL_ARCH"] = MODEL_ARCH
    res["ENCODER_NAME"] = ENCODER_NAME

    res['epoch_for_best'] = epoch_for_best

    res['early_stopping_best'] = early_stopping_best
    
    res['model_size'] = os.path.getsize(checkpoint_path)


    total_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters: {total_params}")
    

    
    # if checkpoint_path_full != "":
    #     if epoch == 0:
    #         res['model_size'] = os.path.getsize(f"{checkpoint_path_full}/checkpoint.pth")
    #     else:
    #         res['model_size'] = os.path.getsize(checkpoint_path)
    # else:
    #     res['model_size'] = os.path.getsize(checkpoint_path)

    res = pd.DataFrame(res)
    append2csv(f"final_results.csv", res.round(4))