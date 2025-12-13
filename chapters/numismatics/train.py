import torch
import numpy as np
from utils import metrics_computation

def epoch_train_valid(train_loader, valid_loader, model, criterion, opt, device, epoch, model_type):    
    
    max_valid_acc = 0
    min_valid_loss = np.inf
    for i in range(epoch):
        truth = []
        prediction = []
        final_train_loss = 0
        final_train_accuracy = 0
        final_valid_loss = 0
        final_valid_accuracy = 0
        model.train()
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)
            train_pred = model(image)
            train_loss = criterion(train_pred, label)
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            
            final_train_loss += train_loss.item() * len(image)
            value, max_idx = torch.max(train_pred, dim=1)
            final_train_accuracy += torch.sum(max_idx == label)
        
        with torch.no_grad():
            model.eval()    
            for image, label in valid_loader:
                image, label = image.to(device), label.to(device)    
                valid_pred = model(image)
                valid_loss = criterion(valid_pred, label)
                final_valid_loss += valid_loss.item() * len(image)
                value, max_idx = torch.max(valid_pred, dim=1)
                truth.extend(label.cpu())
                prediction.extend(max_idx.cpu())
                final_valid_accuracy += torch.sum(max_idx==label)
                
        final_train_loss = final_train_loss / len(train_loader.dataset)
        final_train_accuracy = final_train_accuracy / len(train_loader.dataset)
        final_valid_loss = final_valid_loss / len(valid_loader.dataset)
        final_valid_accuracy = final_valid_accuracy / len(valid_loader.dataset)
        
        if final_valid_accuracy > max_valid_acc:
            max_valid_acc = final_valid_accuracy
            max_epoch = i + 1
            torch.save(model.state_dict(), "checkpoints/{}_model.pt".format(model_type))
            metrics_computation(truth, prediction, model_type, type=model_type.split('_')[-1])
        
        # if final_valid_loss < min_valid_loss:
        #     min_valid_loss = final_valid_loss
        #     max_epoch = i + 1
        #     torch.save(model.state_dict(), "checkpoints/{}_model.pt".format(model_type))
        if (i+1) % 10 == 0:
            print(f'[Epoch {i + 1}] train loss: {final_train_loss:.3f}; train acc: {final_train_accuracy:.2f}; valid loss: {final_valid_loss:.3f}; valid acc: {final_valid_accuracy:.2f}')
    print('Maximum Validation accuracy: ', max_valid_acc, ' at epoch: ', max_epoch)
    # return final_train_loss, final_train_accuracy, final_valid_loss, final_valid_accuracy


def test_function(test_loader, model, model_path, criterion, device):    
    final_loss = 0
    final_accuracy = 0
    model.load_state_dict(torch.load("checkpoints/{}_model.pt".format(model_path)))
    with torch.no_grad():
        model.eval()
        model.to(device)
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            pred = model(image)
            loss = criterion(pred, label)
            final_loss += loss.item() * len(image)
            value, max_idx = torch.max(pred, dim=1)
            final_accuracy += torch.sum(max_idx == label)
    final_loss = final_loss / len(test_loader.dataset)
    final_accuracy = final_accuracy / len(test_loader.dataset)
    print(f"Test Loss: {final_loss:.3f}; Test Accuracy: {final_accuracy:.3f}")
    # return final_loss, final_accuracy
        


if __name__ == "__main__":
    print('Check')






  
