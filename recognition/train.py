import torch
from dataset import get_loader
from modules import UNet2D
from utils import SEG_TRAIN_PATH, IMAGE_TRAIN_PATH, SEG_TEST_PATH, IMAGE_TEST_PATH, SEG_VAL_PATH, IMAGE_VAL_PATH
from torchvision.transforms import v2
import torch.nn as nn
import torch.nn.functional as F
import time 

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(device)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.__version__)

# model hyperparameters
learning_rate = 10e-3
epochs = 30
batch_size = 16

latent_channels = 64
in_channels = 1

# hipmri data contains 6 classes total
num_classes = 6

# can use transforms if results do not meet criteria
# transforms = v2.Compose()

train_loader = get_loader(IMAGE_TRAIN_PATH, SEG_TRAIN_PATH)
model = UNet2D(in_channels=in_channels, latent_channels=latent_channels, num_classes=num_classes)
model = model.to(device)

# Loss function
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, num_classes=6):
        super(DiceLoss, self).__init__()
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
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

criterion = DiceLoss()
total_step = len(train_loader)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train():
    model.train()

    print(" -- Training -- ")

    start = time.time()
    for epoch in range(epochs):
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # backprop
            optimizer.zero_grad() # reset optimizer for each iteration
            loss.backward()
            optimizer.step()

            # print("Batch {}, Loss {:.3f}".format(i, loss.item()))
    
        print("Epoch [{}/{}], Loss: {:.5f}"
                    .format(epoch+1, epochs, loss.item()))
                
    end = time.time()
    elapsed = end - start
    print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    print("Daniel Miller s4742751")

    torch.save(model.state_dict(), "2DUnet_trained.pth")


if __name__ == "__main__":
    print(device)
    train()

