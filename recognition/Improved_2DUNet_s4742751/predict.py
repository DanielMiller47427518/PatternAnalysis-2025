import torch
import numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import get_loader
from modules import UNet2D
from utils import MODEL_PATH, SEG_TEST_PATH, IMAGE_TEST_PATH, class_map, IMAGE_PATH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 16
num_classes = 6
in_channels = 1
latent_channels = 64

test_loader = get_loader(IMAGE_TEST_PATH, SEG_TEST_PATH)

class DiceScorePredict(nn.Module):
    def __init__(self, smooth=1e-5, num_classes=6):
        super(DiceScorePredict, self).__init__()
        self.smooth = smooth
        self.softmax = nn.Softmax(dim=1)

    def forward(self, preds, targets):
        """
            Params:
                preds: output from model on a given inference
                targets: ground truth labels for a given input image
        """
        # apply the softmax function to the predictions to get class probs
        pred = self.softmax(preds)
        targets = targets.squeeze(1).long()
        # need target to be one-hot encoded to match output dimensions of model
        target_onehot = F.one_hot(targets.long(), num_classes=num_classes)
        # need to rearrange indicies to match format of [batch size, channels, height, width]
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        pred = pred.flatten(start_dim=2)
        target = target_onehot.flatten(start_dim=2)

        # calculate pred intersection with targets
        intersection = (pred * target).sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (pred.sum(dim=2) + target.sum(dim=2) + self.smooth)

        dice_class = dice.mean(dim=0)
        return dice_class

model = UNet2D(in_channels=in_channels, latent_channels=latent_channels, num_classes=num_classes)
model = model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
criterion = DiceScorePredict()

def evaluate():
    """
    Run inference of the model using test data
    Outputs Dice Loss per class
    """
    model.eval()
    batch_num = 0
    dice_scores = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            scores = criterion(outputs, masks)
            
            dice_scores += scores
            batch_num += 1

    dice_per_class = dice_scores / batch_num
    dice_per_class = dice_per_class.cpu()

    print("Performance of each class on the test set:")
    for i, score in enumerate(dice_per_class.cpu()):
        print(f"{class_map[i]}: {score:.3f}")

def plot_image(image, mask, pred, idx):
    """
    Takes an example image and its index in a given batch and saves plot of image, seg mask and prediction
    """
    # tensors so bring back to cpu for plotting
    image = image.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(10,5))
    
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Raw Image")
    ax[0].axis('off')

    ax[1].imshow(mask)
    ax[1].set_title("Segmentation Mask")
    ax[1].axis('off')

    ax[2].imshow(pred)
    ax[2].set_title("Model Prediction")
    ax[2].axis('off')
   
    name = "output_" + str(idx) + ".png"
    plt.savefig(os.path.join(IMAGE_PATH, name))
    plt.show

def plot_samples(model, loader, nth_batch=0, num_samples=3):
    """
    Runs inference on the model and plots the predictions along with images and segmentation masks
    Saves images in current directory
    """
    model.eval()
    # list batches so we can extract desired batch
    batches = list(loader)
    images, masks = batches[nth_batch]

    images = images.to(device)
    masks = masks.to(device)

    # run inference to get predicted mask
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

    # plot each mask and save
    for i in range(min(num_samples,images.shape[0])):
        plot_image(images[i], masks[i], preds[i], i)

if __name__ == "__main__":

    evaluate()
    plot_samples(model, test_loader)