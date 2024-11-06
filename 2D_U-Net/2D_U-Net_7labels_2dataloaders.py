########################################################################################################################
#                                                  IMPORTS                                                             #
########################################################################################################################

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import nibabel as nib
import sys
import time as time
from scipy import ndimage
import random


########################################################################################################################
#                                                 VARIABLES                                                            #
########################################################################################################################

config = {
    # Paths
    'data_path': '',   # Path to the MRI data
    'labels_path': '',   # Path to the segmentations
    'weights_path': '',   # Path to save the model weights

    # Model
    'input_ch': 1,
    'output_ch': 8,

    # Hyper-parameters
    'lr': 0.01,
    'test_split': 0.2,
    'val_split': 0.1,
    'smooth': 1e-5,
    'ee': sys.float_info.epsilon,

    # Training settings
    'num_epochs': 500,
    'batch_size': 8,  # If using two dataloaders, need batch size to be a multiple of 2
    'device': "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

}


########################################################################################################################
#                                                 FUNCTIONS & CLASSES                                                  #
########################################################################################################################

def find_label(data, label=None):
    '''Find in the date the objects associated to the specific label.'''
    data = data.astype(int)
    if label is not None:
        return ndimage.find_objects(data == label)
    else:
        return ndimage.find_objects(data)


def convert_to_binary_mask(segmentation, target_label=4):
    """Converts a segmentation to a binary mask."""
    binary_mask = (segmentation == target_label).astype(np.float32)
    return binary_mask


def get_loader(dataset, batch_size):
    """Builds and returns Dataloader."""
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
    return data_loader


def evaluate(dataloader, model, criterion, threshold=0.5):
    """Evaluate the model on the validation set."""
    model.eval()
    losses = []
    scores = []

    for batch in dataloader:
        x1, y1 = batch[0]
        x2, y2 = batch[1]

        # Extract tensors and filenames
        tensor1_data = x1[0][0]
        tensor1_labels = x1[0][1]
        tensor1_filenames = x1[1]
        tensor1_seg_data = y1[0][0]
        tensor1_seg_labels = y1[0][1]
        tensor1_seg_filenames = y1[1]

        tensor2_data = x2[0][0]
        tensor2_labels = x2[0][1]
        tensor2_filenames = x2[1]
        tensor2_seg_data = y2[0][0]
        tensor2_seg_labels = y2[0][1]
        tensor2_seg_filenames = y2[1]

        # Concatenate the tensors along the first dimension (dimension 1 for labels)
        concatenated_data = torch.cat((tensor1_data, tensor2_data), dim=0)
        concatenated_labels = torch.cat((tensor1_labels, tensor2_labels), dim=0)
        concatenated_seg_data = torch.cat((tensor1_seg_data, tensor2_seg_data), dim=0)
        concatenated_seg_labels = torch.cat((tensor1_seg_labels, tensor2_seg_labels), dim=0)

        # Combine the filenames
        concatenated_filenames = tensor1_filenames + tensor2_filenames
        concatenated_seg_filenames = tensor1_seg_filenames + tensor2_seg_filenames

        # Create the final concatenated structure
        x = [[concatenated_data, concatenated_labels], concatenated_filenames]
        y = [[concatenated_seg_data, concatenated_seg_labels], concatenated_seg_filenames]

        x = x[0][0]
        y = y[0][0]
        x = x.unsqueeze(1).float().to(device)   # adding a channel dimension to the x tensor
        y = y.long().to(device)

        pred = model(x)

        loss = criterion(pred, y)
        losses.append(loss.item())
        scores.append(dice_score(pred, y).cpu().detach().numpy())

    return np.mean(losses), np.mean(scores)


class DiceLoss(nn.Module):
    """ Dice loss."""
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        target = F.one_hot(target, num_classes=config['output_ch']).permute(0, 3, 1, 2)
        target = target.float()

        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice.mean(dim=1)  # to have the mean per structure
        dice_loss = 1 - dice_loss.mean()

        return dice_loss


class CombinedLoss(nn.Module):
    """ Combined loss of crossentropy and dice. """
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(CombinedLoss, self).__init__()
        # we already apply a softmax at the end of the network, thus we have to do the crossentropy loss "by hand"
        self.cross_entropy = nn.NLLLoss()
        self.dice_loss = DiceLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        ce_loss = self.cross_entropy(torch.log(pred + config['ee']), target)
        dice_loss = self.dice_loss(pred, target)
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss


def dice_score(pred, target, smooth=1e-5):
    """ Computes the dice score. """
    target = F.one_hot(target, num_classes=config['output_ch']).permute(0, 3, 1, 2)
    target = target.float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    dice = dice.mean(dim=1)

    return dice.mean()


########################################################################################################################
#                                                   DATA                                                               #
########################################################################################################################

# Find all the files in the data and segmentation folders which end with .nii.gz
data_files = [file for file in os.listdir(config['data_path']) if file.endswith('.nii.gz')]
segmentation_files = [f for f in os.listdir(config['labels_path']) if f.endswith('.nii.gz')]

# print("Number of MRI files:", len(data_files))
# print("Number of segmentation files:", len(segmentation_files))

# Load all the data and labels
# Each element of the list is a tuple containing the data/label and the file name
mri = [((nib.as_closest_canonical(nib.load(config['data_path'] + file)).get_fdata(),
         nib.load(config['data_path'] + file).affine), file) for file in data_files]
segmentation = [((nib.as_closest_canonical(nib.load(config['labels_path'] + seg)).get_fdata(),
                  nib.load(config['labels_path'] + seg).affine), seg) for seg in segmentation_files]

mri_number = len(mri)
print("Number of MRI files loaded:", len(mri))
print("Number of segmentation files loaded:", len(segmentation))

# Find the images that contain ventricles
non_empty_mri = []
non_empty_segmentation = []

for seg in segmentation:
    # Retrieve only the segmentations and corresponding MRI that have ventricles label (4)
    if len(find_label(seg[0][0], label=4)) > 0:
        non_empty_segmentation.append(seg)
        # Find the corresponding MRI
        for m in mri:
            if m[1][3:] == seg[1][12:]:   # To modify depending on the file names
                non_empty_mri.append(m)
                break

print("Number of segmentation files containing ventricles:", len(non_empty_segmentation))

# Shuffle non_empty_mri and non_empty_segmentation
c = list(zip(non_empty_mri, non_empty_segmentation))
random.shuffle(c)
non_empty_mri, non_empty_segmentation = zip(*c)


# Split the data into training, validation and test sets
# Keep in non_empty_mri and non_empty_segmentation only half of the total data
non_empty_mri = non_empty_mri[:mri_number // 2]
non_empty_segmentation = non_empty_segmentation[:mri_number // 2]


non_empty_segmentation_names = [seg[1][13:] for seg in non_empty_segmentation]   # To modify depending on the file names

# Create the second dataset that contains the images without ventricles and the images not used in the first dataset
mri_other = []
for m in mri:
    if m[1][4:] not in non_empty_segmentation_names:   # To modify depending on the file names
        mri_other.append(m)

for mri in non_empty_mri[mri_number // 2:]:
    mri_other.append(mri)

segmentation_other = []
for m in mri_other:
    for seg in segmentation:
        if m[1][3:] == seg[1][12:]:   # To modify depending on the file names
            segmentation_other.append(seg)
            break

data_train1, data_val1, labels_train1, label_val1 = train_test_split(non_empty_mri, non_empty_segmentation,
                                                                     test_size=config['val_split'], random_state=42)
data_train2, data_val2, labels_train2, label_val2 = train_test_split(mri_other, segmentation_other,
                                                                     test_size=config['val_split'], random_state=42)

# Create train and val datasets of tuples containing the data and the associated labels
train_dataset1 = []
val_dataset1 = []
train_dataset2 = []
val_dataset2 = []

for mri_train in data_train1:
    mri_name = mri_train[1]
    for seg in non_empty_segmentation:
        seg_name = seg[1]
        if mri_name[3:] == seg_name[12:]:   # To modify depending on the file names
            train_dataset1.append((mri_train, seg))

for mri_val in data_val1:
    mri_name = mri_val[1]
    for seg in non_empty_segmentation:
        seg_name = seg[1]
        if mri_name[3:] == seg_name[12:]:   # To modify depending on the file names
            val_dataset1.append((mri_val, seg))

for mri_train in data_train2:
    mri_name = mri_train[1]
    for seg in segmentation_other:
        seg_name = seg[1]
        if mri_name[3:] == seg_name[12:]:   # To modify depending on the file names
            train_dataset2.append((mri_train, seg))

for mri_val in data_val2:
    mri_name = mri_val[1]
    for seg in segmentation_other:
        seg_name = seg[1]
        if mri_name[3:] == seg_name[12:]:   # To modify depending on the file names
            val_dataset2.append((mri_val, seg))

print("Train dataset 1 length:", len(train_dataset1))
print("Validation dataset 1 length:", len(val_dataset1))

print("Train dataset 2 length:", len(train_dataset2))
print("Validation dataset 2 length:", len(val_dataset2))


########################################################################################################################
#                                                    MODEL                                                             #
########################################################################################################################

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        d0 = self.softmax(d1)

        return d0


########################################################################################################################
#                                                   MAIN                                                               #
########################################################################################################################

if __name__ == '__main__':
    # Device
    device = torch.device(config['device'])
    print(f"----- Device: {device} -----")

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Model instantiation
    model = U_Net(config['input_ch'], config['output_ch']).to(device)

    # Loss function
    # loss_fn = DiceLoss()
    loss_fn = CombinedLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # Data loaders
    # Need batch size to be a multiple of 2
    if config['batch_size'] < 2:
        print("Batch size must be a multiple at least 2")
        config['batch_size'] = 2
    train_loader1 = get_loader(train_dataset1, int(config['batch_size'] / 2))
    val_loader1 = get_loader(val_dataset1, int(config['batch_size'] / 2))
    train_loader2 = get_loader(train_dataset2, int(config['batch_size'] / 2))
    val_loader2 = get_loader(val_dataset2, int(config['batch_size'] / 2))

    ################################################################################################################

    # TRAINING LOOP #

    train_dice_scores = []
    train_losses = []

    val_dice_scores = []
    val_losses = []

    for epoch in tqdm(range(config['num_epochs'])):
        print(f"----- Epoch {epoch} -----")
        losses = []
        scores = []

        # Train over the two dataloaders
        for batch in zip(train_loader1, train_loader2):
            model.train()
            optimizer.zero_grad()
            x1, y1 = batch[0]
            x2, y2 = batch[1]

            # Extract tensors and filenames
            tensor1_data = x1[0][0]
            tensor1_labels = x1[0][1]
            tensor1_filenames = x1[1]
            tensor1_seg_data = y1[0][0]
            tensor1_seg_labels = y1[0][1]
            tensor1_seg_filenames = y1[1]

            tensor2_data = x2[0][0]
            tensor2_labels = x2[0][1]
            tensor2_filenames = x2[1]
            tensor2_seg_data = y2[0][0]
            tensor2_seg_labels = y2[0][1]
            tensor2_seg_filenames = y2[1]

            # Concatenate the tensors along the first dimension 
            concatenated_data = torch.cat((tensor1_data, tensor2_data), dim=0)
            concatenated_labels = torch.cat((tensor1_labels, tensor2_labels), dim=0)
            concatenated_seg_data = torch.cat((tensor1_seg_data, tensor2_seg_data), dim=0)
            concatenated_seg_labels = torch.cat((tensor1_seg_labels, tensor2_seg_labels), dim=0)

            # Combine the filenames
            concatenated_filenames = tensor1_filenames + tensor2_filenames
            concatenated_seg_filenames = tensor1_seg_filenames + tensor2_seg_filenames

            # Create the final concatenated structure
            x = [[concatenated_data, concatenated_labels], concatenated_filenames]
            y = [[concatenated_seg_data, concatenated_seg_labels], concatenated_seg_filenames]

            x = x[0][0]
            y = y[0][0]

            x = x.unsqueeze(1).float().to(device)   # adding a channel dimension to the x tensor
            y = y.long().to(device)

            pred = model(x)

            loss = loss_fn(pred, y)
            loss.backward()

            optimizer.step()

            losses.append(loss.item())
            scores.append(dice_score(pred, y).cpu().detach().numpy())

        print("Train loss: ", np.mean(losses))
        print("Train dice score: ", np.mean(scores))

        eval_loss, eval_dice = evaluate(zip(val_loader1, val_loader2), model, loss_fn)
        print("Val loss: ", eval_loss)
        print("Val dice score: ", eval_dice)
        val_losses.append(eval_loss)
        val_dice_scores.append(eval_dice)

        # Write train and validation losses and accuracies to TensorBoard
        writer.add_scalar('Loss/Train', np.mean(losses), epoch)
        writer.add_scalar('Loss/Validation', eval_loss, epoch)
        writer.add_scalar('Accuracy/Train', np.mean(scores), epoch)
        writer.add_scalar('Accuracy/Validation', eval_dice, epoch)

        scheduler.step()

    # Close TensorBoard writer
    writer.close()

    # Save model weights
    weights_name = f"model_weights_{time.strftime('%Y%m%d-%H%M%S')}.pth"
    weights_path = os.path.join(config['weights_path'], weights_name)
    torch.save(model.state_dict(), weights_path)