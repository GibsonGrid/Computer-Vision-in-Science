
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch.nn as nn


def data_transformer(gray_scale, side):
    # tensor([0.6260, 0.6123, 0.5762]), tensor([0.2012, 0.2040, 0.2096])) with crop (tensor([0.6123]), tensor([0.2032]))
    # (tensor([0.6699, 0.6577, 0.6269]), tensor([0.2365, 0.2410, 0.2503])) without crop (tensor([0.6579]), tensor([0.2401]))
    reshape_size = (400, 195) #(200, 195) #(400, 195) 
    crop_size = 350 # 180
    # both sides
    # (tensor([0.5972, 0.5815, 0.5387]), tensor([0.1629, 0.1638, 0.1650])) # crop 180
    # (tensor([0.3611, 0.3537, 0.3344]), tensor([0.5787, 0.5722, 0.5547])) # crop 350
    mean = [0.3611, 0.3537, 0.3344]  #[0.6260, 0.6123, 0.5762] #[0.3611, 0.3537, 0.3344]  
    std = [0.5787, 0.5722, 0.5547] #[0.2012, 0.2040, 0.2096] #[0.5787, 0.5722, 0.5547]  
    if side == 'single':
        reshape_size = (200, 195)
        crop_size = 180
        mean = [0.6260, 0.6123, 0.5762] 
        std = [0.2012, 0.2040, 0.2096] 
    else:
        reshape_size = (400, 195)
        crop_size = 350
        mean = [0.3611, 0.3537, 0.3344]  
        std = [0.5787, 0.5722, 0.5547] 
        
    image_transforms = {
        'train':
        transforms.Compose([
            # transforms.RandomResizedCro/p(size=(200,195), scale=(0.95, 1.0)),
            transforms.Resize(size=reshape_size),  # (200, 195)
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std) #[0.5, 0.5, 0.5]
        ]),
        'val':
        transforms.Compose([
            transforms.Resize(size=reshape_size),
            transforms.CenterCrop(size=crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test':
        transforms.Compose([
            transforms.Resize(size=reshape_size),
            transforms.CenterCrop(size=crop_size),            
            transforms.ToTensor(),
            transforms.Normalize(mean, std)  
        ]),
    }
    image_transforms2 = {
        'train':
        transforms.Compose([
            transforms.Resize((200,195)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=180),
            transforms.Grayscale(1),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.6123], std=[0.2032]) 
        ]),
        'val':
        transforms.Compose([
            transforms.Resize(size=(200,195)),
            transforms.CenterCrop(size=180),
            transforms.Grayscale(1),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.6123], std=[0.2032])
        ]),
        'test':
        transforms.Compose([
            transforms.Resize(size=(200,195)),
            transforms.CenterCrop(size=180),            
            transforms.Grayscale(1),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.6123], std=[0.2032])  # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ]),
    }
    return image_transforms if not gray_scale else image_transforms2

def process_last(root, batch_size, gray, side='single'):
    all_data = datasets.ImageFolder(root=root)
    train_data_len = int(len(all_data)*0.8)
    valid_data_len = int((len(all_data) - train_data_len)/2)
    test_data_len = int(len(all_data) - train_data_len - valid_data_len)
    generator = torch.Generator().manual_seed(42)
    # train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len], generator=generator)
    train_data, test_data = random_split(all_data, [train_data_len+valid_data_len, test_data_len], generator=generator)
    image_transforms = data_transformer(gray, side)
    train_data.dataset.transform = image_transforms['train']
    # val_data.dataset.transform = image_transforms['val']
    test_data.dataset.transform = image_transforms['test']
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
    # return train_loader, test_loader    

if __name__ == '__main__':
    head_path = '../Dataset/OnlyObverse'
    batch_size = 16
    a, b, c = process_last(head_path, batch_size, True)
    for x, y, in c:
        print(x.shape, y.shape)





