import torch
import torch.nn as nn
from torchvision import models
import timm


def top_model(inp_feat, out_feat):
    # with a possibilit of further modification
    new_classifier = nn.Sequential(
        nn.Linear(in_features=inp_feat, out_features=out_feat, bias=True),
        nn.Softmax(dim=1)
        )
    return new_classifier

def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model

def modified_classifiers(model_name, lr=1e-3, pretrain=True, freeze=False, num_classes=30):
    if model_name == 'vgg19':
        model = models.vgg19(pretrained=pretrain) #  weights='VGG19_Weights.DEFAULT'
        model = model if not freeze  else freeze_layers(model)
        model.classifier = top_model(model.classifier[0].in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrain) #  weights='VGG19_Weights.DEFAULT'
        model = model if not freeze  else freeze_layers(model)
        model.classifier = top_model(model.classifier[0].in_features, num_classes) 
    elif model_name == 'mobilenetv2':
        model = models.mobilenet_v2(weights=pretrain)
        model.classifier = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    elif model_name == 'mobilenetv3small':
        model = models.mobilenet_v3_small(pretrained=pretrain)
        model.classifier[3] = nn.Linear(in_features=model.classifier[3].in_features, out_features=num_classes)
    elif model_name == 'inception3':
        model = timm.create_model(model_name='inception_v3', pretrained=pretrain, in_chans=3, num_classes=num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrain) # weights='ResNet50_Weights.DEFAULT'
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        new_classifier = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True),
            nn.Softmax(dim=1)
            )
        model.fc = new_classifier
        # model.fc = top_model(model.fc.in_features, model.fc.in_features//2)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrain) # weights='DenseNet121_Weights.DEFAULT'
        model = model if not freeze  else freeze_layers(model)
        new_classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=num_classes, bias=True),
            nn.Softmax(dim=1)
        )
        model.classifier = new_classifier
    opt = torch.optim.Adam(model.parameters(), lr=lr) if model_name not in ['vgg16', 'vgg19'] else torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    return model, opt


resnet_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

def top_model_for_resnet(feature_size):
    # with a possibilit of further modification
    new_classifier = nn.Sequential(
        nn.Linear(in_features=feature_size, out_features=num_classes, bias=True),
        nn.Softmax(dim=1)
        )
    return new_classifier

clf_changer = lambda in_size: nn.Sequential(nn.Linear(in_features=in_size, out_features=num_classes), nn.Softmax(dim=1))

def all_resnets(model_name, lr=1e-3):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = top_model_for_resnet(model.fc.in_features)
    elif model_name ==  'resnet34':
        model = models.resnet34(pretrained=True)
        model.fc = top_model_for_resnet(model.fc.in_features)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = top_model_for_resnet(model.fc.in_features)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        model.fc = top_model_for_resnet(model.fc.in_features)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        model.fc = top_model_for_resnet(model.fc.in_features)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, opt


if __name__ == "__main__":
    inp = torch.zeros((4, 3, 200, 195))
    print(inp.shape)
    # model = CNN()
    for mo in ['inception3']:
        print('Model', mo)
        model, _ = modified_classifiers(model_name=mo, pretrain=False, freeze=False)
        out = model(inp)
        print(out.shape)
