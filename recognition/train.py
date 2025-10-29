import torch
from dataset import get_loader
from modules import UNet2D
from utils import SEG_TRAIN_PATH, IMAGE_TRAIN_PATH, SEG_TEST_PATH, IMAGE_TEST_PATH, SEG_VAL_PATH, IMAGE_VAL_PATH
from torchvision.transforms import v2

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# model hyperparameters
learning_rate = 10e-3
epochs = 5
batch_size = 16

latent_channels = 64
in_channels = 1

# hipmri data contains 6 classes total
num_classes = 6

# can use transforms if results do not meet criteria
transforms = v2.Compose()

train_loader = get_loader(IMAGE_TRAIN_PATH, SEG_TRAIN_PATH)
model = UNet2D(in_channels=in_channels, latent_channels=latent_channels, num_classes=num_classes)
model = model.to(device)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
            Params:
                preds: output from model on a given inference
                targets: ground truth labels for a given input image
        """
        preds = preds.reshape(-1)
        targets = targets.reshape(-1).float()

        # calculate pred intersection with targets
        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice

criterion = DiceLoss()
total_step = len(train_loader)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



# Loss function




