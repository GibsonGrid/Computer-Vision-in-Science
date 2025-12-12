import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl

def gazeto3d(gaze):
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt

def angular(gaze, label):
    total = np.sum(gaze * label)
    return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi
    

class FullFace_pl(pl.LightningModule):
    def __init__(self, lr, img_size):
        super().__init__()
        self.save_hyperparameters() 
        self.lr = lr
        self.img_size = img_size

        alexnet = torchvision.models.alexnet(pretrained=True)

        self.convNet = alexnet.features

        self.weightStream = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.img_size, self.img_size)
            features = self.convNet(dummy_input)
            weighted_features = self.weightStream(features) * features
            flattened_size = weighted_features.view(1, -1).shape[1]

        self.FC = nn.Sequential(
            nn.Linear(flattened_size, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )

    def forward(self, x_in):
        faceFeature = self.convNet(x_in["face"])
        weight = self.weightStream(faceFeature)

        faceFeature = weight * faceFeature

        faceFeature = torch.flatten(faceFeature, start_dim=1)
        gaze = self.FC(faceFeature)

        return gaze

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9,0.95))
        sched = optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.1)
        return [opt], [{                                                      
                "scheduler": sched,                                            
                "interval": "step"    
            }]

    def training_step(self, batch):
        y_hat = self(batch[0])
        loss = F.l1_loss(y_hat, batch[1])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        count = 0
        metric_ang_g = 0
        for k, g in enumerate(y_hat):
            metric_ang_g += angular(gazeto3d(g.cpu().detach().numpy()), gazeto3d(batch[1].cpu().detach().numpy()[k]))
            count += 1
        
        metric_ang_g = metric_ang_g / count
        self.log("train_ang_error", metric_ang_g, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch):
        y_hat = self(batch[0])
        val_loss = F.l1_loss(y_hat, batch[1])
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        count = 0
        metric_ang_g = 0
        for k, g in enumerate(y_hat):
            metric_ang_g += angular(gazeto3d(g.cpu().detach().numpy()), gazeto3d(batch[1].cpu().detach().numpy()[k]))
            count += 1
        
        metric_ang_g = metric_ang_g / count
        self.log("val_ang_error", metric_ang_g, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch):
        y_hat = self(batch[0])
        loss = F.l1_loss(y_hat, batch[1])
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        count = 0
        metric_ang_g = 0
        for k, g in enumerate(y_hat):
            metric_ang_g += angular(gazeto3d(g.cpu().detach().numpy()), gazeto3d(batch[1].cpu().detach().numpy()[k]))
            count += 1
        
        metric_ang_g = metric_ang_g / count
        self.log("test_ang_error", metric_ang_g, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss