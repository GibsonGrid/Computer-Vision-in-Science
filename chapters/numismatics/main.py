from data_processing import process_last
from model2 import modified_classifiers, all_resnets
from train import epoch_train_valid, test_function
import time
import torch
from itertools import product
import argparse
import timm
import torch.nn as nn
import random
import numpy as np

#---------------------------------------------------------------------------------------------------------------------------------------------------
# train models from scratch with no pretrained weight
# train model with gray scale and RGB data
#---------------------------------------------------------------------------------------------------------------------------------------------------

def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saveas', type=str, default='default')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gray', type=bool, default=False)
    parser.add_argument('--weight', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cnn', type=bool, default=False)
    parser.add_argument('--freeze', type=bool, default=False)
    parser.add_argument('--side', type=str, default='single')
    args = parser.parse_args()
    head_path = '../Dataset/OnlyObverse' if args.side == 'single' else '../Dataset/ObverseReverse'
    batch_size = args.batch_size
    lr = 1e-3
    momentum = 0.9
    loss_fun = nn.CrossEntropyLoss()
    n_epochs = args.epoch
    in_chan_size = 1 if args.gray else 3   
    fix_seeds(args.seed)
    model_type = ['vgg16', 'cnn', 'resnet50', 'mobilenetv2']
    model_types = ['vgg16', 'resnet50', 'mobilenetv2_110d', 'densenet121', 'inception_v3'] 
    optimizer = ['sgd'] #['adam', 'sgd']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, valid_loader, test_loader = process_last(root=head_path, batch_size=batch_size, gray=args.gray, side=args.side)
    # train_loader, valid_loader = process_last(root=head_path, batch_size=batch_size, gray=args.gray, side=args.side)
    
    '''
    New training procedure: the below training is done with model where:
        - weight is included with the parameter default (up to date weight)
        - classifier architecture is modified
        - activation function is added
    '''
    # new_models = ['vgg16', 'vgg19', 'mobilenetv2', 'mobilenetv3small', 'inception3', 'resnet50', 'densenet121']
    resnet_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    for clf in resnet_models:
        # model, optim = modified_classifiers(model_name=clf, freeze=args.freeze)
        model, optim = all_resnets(model_name=clf)
        model = model.to(device)
        st = time.time()
        print('Model type: ', clf)
        epoch_train_valid(train_loader, valid_loader, model, loss_fun, optim, device, n_epochs, f'{clf}_{args.saveas}')
        test_function(test_loader=test_loader, model=model, model_path=f"{clf}_{args.saveas}", criterion=loss_fun, device=device)
        en = time.time()
        print('Elapsed time: ', round((en-st)/60,2), ' minutes')
        print("*"*100)
    
    """
    if args.cnn is not True:
        for clf in model_types:
            model = timm.create_model(model_name=clf, pretrained=args.weight, in_chans=in_chan_size, num_classes=3)
            optim = torch.optim.Adam(model.parameters(), lr=lr) if clf != 'vgg16' else torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            model = model.to(device)
            st = time.time()
            print('Model type: ', clf)
            epoch_train_valid(train_loader, valid_loader, model, loss_fun, optim, device, n_epochs, f'{clf}_{args.saveas}')
            test_function(test_loader=test_loader, model=model, model_path=f"{clf}_{args.saveas}", criterion=loss_fun, device=device)
            en = time.time()
            print('Elapsed time: ', round((en-st)/60,2), ' minutes')
            print("*"*100)
    else:
        clf = 'cnn'
        model, optim, loss_fun = classifier_detail(model_choice=clf, lr=lr, gray_scale=args.gray, pretrain=args.weight)
        model = model.to(device)
        st = time.time()
        print('Model type: ', clf)
        epoch_train_valid(train_loader, valid_loader, model, loss_fun, optim, device, n_epochs, f'{clf}_{args.saveas}')
        test_function(test_loader=test_loader, model=model, model_path=f"{clf}_{args.saveas}", criterion=loss_fun, device=device)
        en = time.time()
        print('Elapsed time: ', round((en-st)/60,2), ' minutes')
        print("*"*100)
    """