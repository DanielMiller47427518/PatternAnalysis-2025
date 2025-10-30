import torch
from dataset import get_loader
from modules import UNet2D
from utils import SEG_TRAIN_PATH, IMAGE_TRAIN_PATH, SEG_TEST_PATH, IMAGE_TEST_PATH, SEG_VAL_PATH, IMAGE_VAL_PATH
from torchvision.transforms import v2
from torch.nn.utils import clip_grad_norm_
from predict import DiceScorePredict
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time 

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(device)

# model hyperparameters
learning_rate = 1e-4
epochs = 30
batch_size = 16
momentum = 0.9
weight_decay = 1e-3

latent_channels = 64
in_channels = 1
# add extra weighting for under-represented classes
class_weights = torch.tensor([0.3, 0.3, 1.0, 1.0, 1.5, 1.5]).to(device) 
# hipmri data contains 6 classes total
num_classes = 6

# can use transforms if results do not meet criteria
transforms = v2.Compose([
    v2.RandomVerticalFlip(),
    v2.RandomRotation(10)
]
)

train_loader = get_loader(IMAGE_TRAIN_PATH, SEG_TRAIN_PATH, transforms=transforms)

validation_loader = get_loader(IMAGE_VAL_PATH, SEG_VAL_PATH)

model = UNet2D(in_channels=in_channels, latent_channels=latent_channels, num_classes=num_classes)
model = model.to(device)

# Loss function
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-4, num_classes=6, class_weights=torch.ones(num_classes)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.softmax = nn.Softmax(dim=1)
        self.weights = class_weights

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
        dice = dice.mean(dim=0)



        dice_loss = self.weights * (1 - dice)

        return dice_loss.mean()

criterion = DiceLoss(class_weights=class_weights)
total_step = len(train_loader)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# losses for each epoch, used for plot
train_losses = []
validation_losses = []


def train():
    model.train()

    print(" -- Training -- ")

    start = time.time()
    for epoch in range(epochs):
        loss_total = 0

        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # backprop
            optimizer.zero_grad() # reset optimizer for each iteration
            loss.backward()

            # gradient clip
            clip_grad_norm_(model.parameters(), max_norm=1.0)


            optimizer.step()

            loss_total += loss.item()
        
        # Convert summed loss across all batches to average loss for entire epoch
        epoch_loss = loss_total / len(train_loader)
        train_losses.append(epoch_loss)

        print("Epoch [{}/{}], Loss: {:.5f}"
                    .format(epoch+1, epochs, epoch_loss))


        validate(model, validation_loader, criterion, epoch)

        
        
    end = time.time()
    elapsed = end - start
    print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    print("Daniel Miller s4742751")

    torch.save(model.state_dict(), "2DUnet_trained.pth")


def validate(model, val_loader, criterion, epoch):
    model.eval()
    batch_num = 0
    total_score = 0
    with torch.no_grad():
        for idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            score = criterion(outputs, masks)

            total_score += score

        avg_loss = total_score / len(val_loader)
        validation_losses.append(avg_loss)


        print(f"Validation set loss at epoch: {epoch}/{epochs}: {avg_loss}")

def plot_losses(train_losses, val_losses):
    """
    Plots Training loss and validation loss on same set of axes

    Params:
        train_losses: losses for each epoch on training data
        val_losses: losses for validation set recorded at each epoch throughout training
    
    """
    plt.figure(figsize=(16,10))
    epoch_arr = list(range(1, epochs+1))

    plt.plot(epoch_arr, train_losses, label="Training Dice Loss")
    plt.plot(epoch_arr, val_losses, label="Validation Dice Loss")
    
    plt.title("Training and Validation Dice Loss", fontsize=20)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Dice Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss_plot.png")


if __name__ == "__main__":
    print(device)
  
    # GFG
    train()

    print(train_losses)
    print(validation_losses)

    plot_losses(train_losses, validation_losses)

