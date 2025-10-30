import torch
# import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from dataset import get_loader
from modules import UNet2D
# from train import DiceLoss
from utils import MODEL_PATH, SEG_TEST_PATH, IMAGE_TEST_PATH, class_map

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
    # for i, score in enumerate(dice_per_class.cpu()):
    #     print("Class {}: {:.3f}".format(i, score))

    for i, score in enumerate(dice_per_class.cpu()):
        print(f"{class_map[i]}: {score:.3f}")


if __name__ == "__main__":
    evaluate()