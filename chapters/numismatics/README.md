# Transfer Learning Based Coin Grading
This folder contains files for preprocessing coin dataset and train and test a coin grading model.

### List of models for training. These models can be trained from scratch, or they can fine-tuned with all their weight or only part of the layers frozen.
- VGG
- ResNet (all versions)
- MobileNet
- Inception

### Install the required libraries to successfully run these files.
```bash
pip install -r requirements.txt
```

### The main file runs both the training and the testing pipeline one after the other.
```bash
python main.py --batch_size 4 --epochs 30 --type coin1
```
