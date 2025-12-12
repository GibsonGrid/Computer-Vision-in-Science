import os
import numpy as np
import cv2 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BGR2RGB:
    def __call__(self, image):
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return frame

class MPIIFaceGaze_dataset(Dataset): 
    def __init__(self, path, root, config, header=True):
        self.lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    line = f.readlines()
                    if header: line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                self.lines = f.readlines()
                if header: self.lines.pop(0)

        self.root = root
        self.img_size = config["img_size"]
                
        self.img_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size), antialias=True),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
            ]
        )
        self.bgr2rgb_converter = BGR2RGB()

    def __len__(self):
        return len(self.lines)
 
    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        # For more deatils on annotations, please refer to dataset itself or documeentation provided by GazeHub@Phi-ai Lab 
        face = line[0]
        lefteye = line[1]
        righteye = line[2]
        name = line[3]
        gaze2d = line[7]
        head2d = line[8]
    
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)
    
        headpose = np.array(head2d.split(",")).astype("float")
        headpose = torch.from_numpy(headpose).type(torch.FloatTensor)
    
        rimg = self.bgr2rgb_converter(cv2.imread(os.path.join(self.root, righteye)))
        rimg = rimg.transpose(2, 0, 1)
    
        limg = self.bgr2rgb_converter(cv2.imread(os.path.join(self.root, lefteye)))
        limg = limg.transpose(2, 0, 1)
    
        fimg = self.bgr2rgb_converter(cv2.imread(os.path.join(self.root, face)))
        fimg = fimg.transpose(2, 0, 1)
    
        img = {
            "face":self.img_transform(torch.from_numpy(fimg).type(torch.FloatTensor)),
            "leye":self.img_transform(torch.from_numpy(limg).type(torch.FloatTensor)),
            "reye":self.img_transform(torch.from_numpy(rimg).type(torch.FloatTensor)),
        }

        return img, label
        
def create_dataset(config, labelpath, imagepath, shuffle=True, num_workers=0):
    dataset = MPIIFaceGaze_dataset(labelpath, imagepath, config)  
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return dataset, loader