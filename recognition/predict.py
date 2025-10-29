import torch
from dataset import get_loader
from modules import UNet2D
from train import DiceLoss
from utils import MODEL_PATH, SEG_TEST_PATH, IMAGE_TEST_PATH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 16
num_classes = 6

in_channels = 1
latent_channels = 64


test_loader = get_loader(IMAGE_TEST_PATH, SEG_TEST_PATH)


model = UNet2D(in_channels=in_channels, latent_channels=latent_channels, num_classes=num_classes)
model = model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
criterion = DiceLoss()

def evaluate():

    dice_scores = []
    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            score = 1 - loss
            dice_scores.append(score)

    print("Dice Score on test set: {}".format(sum(dice_scores)/len(dice_scores)))


if __name__ == "__main__":
    evaluate()