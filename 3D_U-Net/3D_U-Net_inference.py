########################################################################################################################
#                                                  IMPORTS                                                             #
########################################################################################################################

from tqdm import tqdm
import numpy as np
import os
import torch
import torch.nn as nn
import nibabel as nib
import torch.utils.data as d
from torch.utils.data import Dataset
import torch.nn.functional as F

import csv
import math
from scipy import ndimage


########################################################################################################################
#                                                 VARIABLES                                                            #
########################################################################################################################

nb_labels = 7  # Number of labels in the segmentation
# nb_labels = 1

# Paths
data_path = ''   # Path to the data
segmentation_path = ''   # Path to the segmentations
weights_path = ''   # Path to the weights
seg_results_path = ''   # Path to save the segmentations
scores_results_path = ''   # Path to save the scores

# Device
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
# device = "cpu"
print("Device:", device)

# Parameters
input_ch, output_ch = 1, 8
# input_ch, output_ch = 1, 2


########################################################################################################################
#                                                   DATA                                                               #
########################################################################################################################

# Find all the files in the data folder which end with .nii.gz
data_files = [file for file in os.listdir(data_path) if file.endswith('.nii.gz')]
segmentation_files = [file for file in os.listdir(segmentation_path) if file.endswith('.nii.gz')]

print("Number of MRI files:", len(data_files))
print("Number of segmentation files:", len(segmentation_files))

# Load all the data
# Each element of the list is a tuple containing the data and the file name
data = [((nib.as_closest_canonical(nib.load(data_path + file)).get_fdata(), nib.load(data_path + file).affine), file)
        for file in data_files]
segmentation = [((nib.as_closest_canonical(nib.load(segmentation_path + seg)).get_fdata(),
                  nib.load(segmentation_path + seg).affine), seg) for seg in segmentation_files]

# Create a list that contains the data and the segmentation
list_data = []
for mri in data:
    mri_name = mri[1][mri[1].find('_') + 1:mri[1].find('.')]   # To modify depending on the file names
    # retrieve the segmentation file associated with the mri file which has the same name
    for seg in segmentation:
        seg_name = seg[1][seg[1].find('_') + 1:seg[1].find('.')]   # To modify depending on the file names
        if mri_name == seg_name:
            list_data.append((mri, seg))
            break

# Sort list_data by the patient number
list_data = sorted(list_data, key=lambda x: x[0][1][4:])   # To modify depending on the file names

nb_patients_test = len(list_data)

# print("Number of patients:", len(list_data))


########################################################################################################################
#                                                    MODEL                                                             #
########################################################################################################################

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
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
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

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

        self.Conv_1x1 = nn.Conv3d(64, output_ch, kernel_size=1, stride=1, padding=0)

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


def get_loader(dataset, batch_size):
    """Builds and returns Dataloader."""

    data_loader = d.DataLoader(dataset=dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               drop_last=False)
    return data_loader


def dice_score_multiclass(prediction, ground_truth, num_classes, smooth=1e-6, class_=None):
    """
    Calculate the Dice score for multi-class segmentation.

    Args:
    - prediction (numpy array): The predicted segmentation mask.
    - ground_truth (numpy array): The ground truth segmentation mask.
    - num_classes (int): The number of classes in the segmentation.
    - smooth (float): Smoothing constant to avoid division by zero.

    Returns:
    - float: Average Dice score across all classes.
    """
    dice_scores = []

    for c in range(1, num_classes + 1):
        prediction_c = (prediction == c).astype(int)
        if class_ is None:
            ground_truth_c = (ground_truth == c).astype(int)
        else:
            ground_truth_c = (ground_truth == class_).astype(int)

        intersection = np.sum(prediction_c * ground_truth_c)
        union = np.sum(prediction_c) + np.sum(ground_truth_c)

        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)

        # print("Dice score for class", c, ":", dice)

    return dice_scores, np.mean(dice_scores)


def precision_score_multiclass(prediction, ground_truth, num_classes, smooth=1e-6, class_=None):
    """
    Calculate the precision score for multi-class segmentation.

    Args:
    - prediction (numpy array): The predicted segmentation mask.
    - ground_truth (numpy array): The ground truth segmentation mask.
    - num_classes (int): The number of classes in the segmentation.
    - smooth (float): Smoothing constant to avoid division by zero.

    Returns:
    - float: Average precision score across all classes.
    """
    precision_scores = []

    for c in range(1, num_classes + 1):
        prediction_c = (prediction == c).astype(int)
        if class_ is None:
            ground_truth_c = (ground_truth == c).astype(int)
        else:
            ground_truth_c = (ground_truth == class_).astype(int)

        true_positives = np.sum(prediction_c * ground_truth_c)
        predicted_positives = np.sum(prediction_c)

        precision = (true_positives + smooth) / (predicted_positives + smooth)
        precision_scores.append(precision)

    return np.mean(precision_scores)


def recall_score_multiclass(prediction, ground_truth, num_classes, smooth=1e-6, class_=None):
    """
    Calculate the recall score for multi-class segmentation.

    Args:
    - prediction (numpy array): The predicted segmentation mask.
    - ground_truth (numpy array): The ground truth segmentation mask.
    - num_classes (int): The number of classes in the segmentation.
    - smooth (float): Smoothing constant to avoid division by zero.

    Returns:
    - float: Average recall score across all classes.
    """
    recall_scores = []

    for c in range(1, num_classes + 1):
        prediction_c = (prediction == c).astype(int)
        if class_ is None:
            ground_truth_c = (ground_truth == c).astype(int)
        else:
            ground_truth_c = (ground_truth == class_).astype(int)

        true_positives = np.sum(prediction_c * ground_truth_c)
        actual_positives = np.sum(ground_truth_c)

        recall = (true_positives + smooth) / (actual_positives + smooth)
        recall_scores.append(recall)

    return np.mean(recall_scores)


def volume_ventricles(segmentation, segmentation_gt, v_label=1):
    """
    Calculate the volume of the Ventricles for a given segmentation.

    Returns:
    - float: Volume of the Ventricles
    - float: Error (difference between the volume of the Ventricles and the expected volume)
    """
    # count the number of voxels that have the value class_
    ventricles = (segmentation == v_label).astype(int)
    ventricles_volume = np.sum(ventricles)

    # count the number of voxels that have the value class_ in the ground truth segmentation
    ventricles_gt = (segmentation_gt == v_label).astype(int)
    ventricles_volume_gt = np.sum(ventricles_gt)

    # calculate the error
    error = (ventricles_volume - ventricles_volume_gt)/1000  # divide by 1000 to convert from mm^3 to cm^3

    return ventricles_volume, error


########################################################################################################################
#                                                  PATCHS                                                              #
########################################################################################################################

def get_patch(image, patch_size, stride):
    """
    Extract patches from a 3D image.

    Args:
        image (np.array): 3D image.
        patch_size (tuple): Size of the patch to extract.
        stride (int): Stride to use when extracting the patches.

    Returns:
        list: List of patches.
    """

    patches = []
    positions = []
    patch_size = (patch_size[0], patch_size[1], patch_size[2])

    for i in range(0, image.shape[0] - patch_size[0] + 1, stride):
        for j in range(0, image.shape[1] - patch_size[1] + 1, stride):
            for k in range(0, image.shape[2] - patch_size[2] + 1, stride):
                patch = image[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]
                patches.append((patch))
                positions.append([i, j, k])

    print("Numbe of patches:", len(patches))

    return patches, positions


########################################################################################################################

# Parameters for the patches
patch_size = (128, 128, 128)
stride = 64

# Create an average image to store the number of patches that overlap at each pixel
avg_image = np.zeros((256, 256, 256))
for i in range(0, avg_image.shape[0] - patch_size[0] + 1, stride):
    for j in range(0, avg_image.shape[1] - patch_size[1] + 1, stride):
        for k in range(0, avg_image.shape[2] - patch_size[2] + 1, stride):
            avg_image[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += 1

# Get patches for the test
patchs_data = [None] * len(list_data)

for i in range(len(list_data)):
    mri_number = list_data[i][0][1][list_data[i][0][1].find('_') + 1:list_data[i][0][1].find('.')]
    mri = list_data[i][0][0][0]
    affine = list_data[i][0][0][1]
    seg = list_data[i][1][0][0]
    mri_patches, positions = get_patch(mri, patch_size, stride)
    patchs_data[i] = ((mri_patches, positions), seg, mri_number, mri.shape, affine)

print("Total number of patches:", sum([len(patch[0][0]) for patch in patchs_data]))


########################################################################################################################
#                                                   MAIN                                                               #
########################################################################################################################

if __name__ == '__main__':

    # Create a csv file to store the scores
    with open(scores_results_path + 'patient_scores.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        if nb_labels == 7:
            writer.writerow(
                ['Patient', 'Dice_score', 'Precision_score', 'Recall_score', 'ECF_Dice_score', 'GM_Dice_score',
                 'WM_Dice_score', 'V_Dice_score', 'C_Dice_score', 'DGM_Dice_score', 'B_Dice_score', 'Ventricles_volume', 'Ventricles_error'])
        if nb_labels == 1:
            writer.writerow(['Patient', 'Dice_score', 'Precision_score', 'Recall_score', 'V_Dice_score', 'Ventricles_volume', 'Ventricles_error'])

    # Defining model
    model = U_Net(input_ch, output_ch).to(device)

    # Initialize scores
    dice_scores = np.zeros(nb_patients_test)
    ECF_dice_scores = np.zeros(nb_patients_test)  # Dice for External Cerebrospinal Fluid
    GM_dice_scores = np.zeros(nb_patients_test)  # Dice for Grey Matter
    WM_dice_scores = np.zeros(nb_patients_test)  # Dice for White Matter
    V_dice_scores = np.zeros(nb_patients_test)  # Dice for Ventricles
    C_dice_scores = np.zeros(nb_patients_test)  # Dice for Cerebellum
    DGM_dice_scores = np.zeros(nb_patients_test)  # Dice for Deep Grey Matter
    B_dice_scores = np.zeros(nb_patients_test)  # Dice for Brainstem
    precision_scores = np.zeros(nb_patients_test)
    recall_scores = np.zeros(nb_patients_test)
    V_volumes = np.zeros(nb_patients_test)
    V_errors = np.zeros(nb_patients_test)
    nb_patients = np.zeros(nb_patients_test)

    # Load pre-trained model
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.eval()

    numero = 0
    print(len(list_data))
    for ii in tqdm(range(len(list_data))):
        test_patches, segm, patient_number, mri_shape, affine = patchs_data[ii]
        print("Patient number:", patient_number)

        # Initialize the 3D image
        pred3d = np.zeros((1, output_ch, mri_shape[0], mri_shape[1], mri_shape[2]))

        list_test = []
        # Get the patches and the positions
        for mri, pos in zip(*test_patches):
            list_test.append((mri, pos))

        # Data loaders
        dataloader = get_loader(list_test, 1)

        for data in dataloader:
            x = data[0]
            x = x.unsqueeze(1).float().to(device)

            pred = model(x)

            pred = pred.cpu().detach().numpy()

            # Get the position of the patch in the 3D image
            pos = data[1]

            # Add the patch to the 3D image
            pred3d[:, :, pos[0]:pos[0] + patch_size[0], pos[1]:pos[1] + patch_size[1],
            pos[2]:pos[2] + patch_size[2]] += pred

        # Do the average for the voxels that overlap
        pred3d = pred3d.squeeze()
        pred3d = pred3d / avg_image
        pred3d = np.argmax(pred3d, axis=0)  # Get the class with the highest probability

        # Save the segmentation of the patient
        pred_to_save = nib.Nifti1Image(np.expand_dims(pred3d, -1).astype(np.int8), affine.squeeze())
        nib.save(pred_to_save, seg_results_path + "patient_" + patient_number)

        # Compute scores
        if nb_labels == 1:
            c = 1  # class_ = 1 for the Ventricles
            v_label = 1
            segm[segm != 4] = 0
            segm[segm == 4] = 1
        if nb_labels == 7:
            c = None
            v_label = 4

        scores, dice = dice_score_multiclass(pred3d, segm, nb_labels, class_=c)
        precision = precision_score_multiclass(pred3d, segm, nb_labels, class_=c)
        recall = recall_score_multiclass(pred3d, segm, nb_labels, class_=c)
        ventricles_volume, ventricles_error = volume_ventricles(pred3d, segm, v_label)

        print("Dice score:", dice)
        print("Precision score:", precision)
        print("Recall score:", recall)
        print("Ventricles volume:", ventricles_volume)
        print("Ventricles error:", ventricles_error)

        if nb_labels == 7:
            ECF_dice_scores[int(numero) - 1] += scores[0]
            GM_dice_scores[int(numero) - 1] += scores[1]
            WM_dice_scores[int(numero) - 1] += scores[2]
            V_dice_scores[int(numero) - 1] += scores[3]
            C_dice_scores[int(numero) - 1] += scores[4]
            DGM_dice_scores[int(numero) - 1] += scores[5]
            B_dice_scores[int(numero) - 1] += scores[6]
        if nb_labels == 1:
            V_dice_scores[int(numero) - 1] += scores[0]

        nb_patients[int(numero) - 1] += 1
        dice_scores[int(numero) - 1] += dice
        precision_scores[int(numero) - 1] += precision
        recall_scores[int(numero) - 1] += recall
        V_volumes[int(numero)-1]+=ventricles_volume
        V_errors[int(numero)-1]+=ventricles_error

        # After processing the 10th subject, write the mean and std of the first 10 subjects (first test dataset)
        if numero == 9:
            means_dice = np.mean(dice_scores[:10])
            std_dice = np.std(dice_scores[:10])
            means_precision = np.mean(precision_scores[:10])
            std_precision = np.std(precision_scores[:10])
            means_recall = np.mean(recall_scores[:10])
            std_recall = np.std(recall_scores[:10])
            mean_ventricles_volume = np.mean(V_volumes[:10])
            std_ventricles_volume = np.std(V_volumes[:10])
            mean_ventricles_error = np.mean(V_errors[:10])
            std_ventricles_error = np.std(V_errors[:10])

            with open(scores_results_path + 'patient_scores.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                if nb_labels == 7:
                    writer.writerow(['Mean of FeTA Zurich dataset', round(means_dice, 3), round(means_precision, 3),
                                     round(means_recall, 3), round(np.mean(ECF_dice_scores[:10]), 3),
                                     round(np.mean(GM_dice_scores[:10]), 3), round(np.mean(WM_dice_scores[:10]), 3),
                                     round(np.mean(V_dice_scores[:10]), 3), round(np.mean(C_dice_scores[:10]), 3),
                                     round(np.mean(DGM_dice_scores[:10]), 3), round(np.mean(B_dice_scores[:10]), 3), 
                                    round(mean_ventricles_volume, 3), round(mean_ventricles_error, 3)])
                    writer.writerow(['Std Dev of FeTA Zurich dataset', round(std_dice, 3), round(std_precision, 3),
                                     round(std_recall, 3), round(np.std(ECF_dice_scores[:10]), 3),
                                     round(np.std(GM_dice_scores[:10]), 3), round(np.std(WM_dice_scores[:10]), 3),
                                     round(np.std(V_dice_scores[:10]), 3), round(np.std(C_dice_scores[:10]), 3),
                                     round(np.std(DGM_dice_scores[:10]), 3), round(np.std(B_dice_scores[:10]), 3),
                                    round(std_ventricles_volume, 3), round(std_ventricles_error, 3)])
                if nb_labels == 1:
                    writer.writerow(['Mean of FeTA Zurich dataset', round(means_dice, 3), round(means_precision, 3),
                                     round(means_recall, 3), round(np.mean(V_dice_scores[:10]), 3), round(mean_ventricles_volume, 3),
                                    round(mean_ventricles_error, 3)])
                    writer.writerow(['Std Dev FeTA Zurich dataset', round(std_dice, 3), round(std_precision, 3),
                                     round(std_recall, 3), round(np.std(V_dice_scores[:10]), 3), round(std_ventricles_volume, 3),
                                    round(std_ventricles_error, 3)])

        # After processing the 20th subject, write the mean and std of the last 10 subjects (second test dataset)
        if numero == 19:
            means_dice_last10 = np.mean(dice_scores[10:])
            std_dice_last10 = np.std(dice_scores[10:])
            means_precision_last10 = np.mean(precision_scores[10:])
            std_precision_last10 = np.std(precision_scores[10:])
            means_recall_last10 = np.mean(recall_scores[10:])
            std_recall_last10 = np.std(recall_scores[10:])
            mean_ventricles_volume_last10 = np.mean(V_volumes[10:])
            std_ventricles_volume_last10 = np.std(V_volumes[10:])
            mean_ventricles_error_last10 = np.mean(V_errors[10:])
            std_ventricles_error_last10 = np.std(V_errors[10:])

            with open(scores_results_path + 'patient_scores.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                if nb_labels == 7:
                    writer.writerow(
                        ['Mean of FeTA Vienna dataset', round(means_dice_last10, 3), round(means_precision_last10, 3),
                         round(means_recall_last10, 3), round(np.mean(ECF_dice_scores[10:]), 3),
                         round(np.mean(GM_dice_scores[10:]), 3), round(np.mean(WM_dice_scores[10:]), 3),
                         round(np.mean(V_dice_scores[10:]), 3), round(np.mean(C_dice_scores[10:]), 3),
                         round(np.mean(DGM_dice_scores[10:]), 3), round(np.mean(B_dice_scores[10:]), 3),
                        round(mean_ventricles_volume_last10, 3), round(mean_ventricles_error_last10, 3)])
                    writer.writerow(
                        ['Std Dev of FeTA Vienna dataset', round(std_dice_last10, 3),
                         round(std_precision_last10, 3), round(std_recall_last10, 3),
                         round(np.std(ECF_dice_scores[10:]), 3), round(np.std(GM_dice_scores[10:]), 3),
                         round(np.std(WM_dice_scores[10:]), 3), round(np.std(V_dice_scores[10:]), 3),
                         round(np.std(C_dice_scores[10:]), 3), round(np.std(DGM_dice_scores[10:]), 3),
                         round(np.std(B_dice_scores[10:]), 3),
                        round(std_ventricles_volume_last10, 3), round(std_ventricles_error_last10, 3)])
                if nb_labels == 1:
                    writer.writerow(['Mean of FeTA Vienna dataset', round(means_dice_last10, 3),
                                     round(means_precision_last10, 3), round(means_recall_last10, 3),
                                     round(np.mean(V_dice_scores[10:]), 3), round(mean_ventricles_volume_last10, 3),
                                    round(mean_ventricles_error_last10, 3)])
                    writer.writerow(['Std Dev of FeTA Vienna dataset', round(std_dice_last10, 3),
                                     round(std_precision_last10, 3), round(std_recall_last10, 3),
                                     round(np.std(V_dice_scores[10:]), 3), round(std_ventricles_volume_last10, 3),
                                    round(std_ventricles_error_last10, 3)])

        numero += 1


    # Write the scores for the Necker dataset
    means_dice_Necker = np.mean(dice_scores[20:])
    std_dice_Necker = np.std(dice_scores[20:])
    means_precision_Necker = np.mean(precision_scores[20:])
    std_precision_Necker = np.std(precision_scores[20:])
    means_recall_Necker = np.mean(recall_scores[20:])
    std_recall_Necker = np.std(recall_scores[20:])
    means_V_volume_Necker = np.mean(V_volumes[20:])
    std_V_volume_Necker = np.std(V_volumes[20:])
    means_V_error_Necker = np.mean(V_errors[20:])
    std_V_error_Necker = np.std(V_errors[20:])

    with open(scores_results_path + 'patient_scores.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if nb_labels == 7:
            writer.writerow(['Mean of Necker dataset', round(means_dice_Necker, 3), round(means_precision_Necker, 3), round(means_recall_Necker, 3), round(np.mean(ECF_dice_scores[20:]), 3), round(np.mean(GM_dice_scores[20:]), 3), round(np.mean(WM_dice_scores[20:]), 3), round(np.mean(V_dice_scores[20:]), 3), round(np.mean(C_dice_scores[20:]), 3), round(np.mean(DGM_dice_scores[20:]), 3), round(np.mean(B_dice_scores[20:]), 3), round(means_V_volume_Necker, 3), round(means_V_error_Necker, 3)])
            writer.writerow(['Std Dev of Necker dataset', round(std_dice_Necker, 3), round(std_precision_Necker, 3), round(std_recall_Necker, 3), round(np.std(ECF_dice_scores[20:]), 3), round(np.std(GM_dice_scores[20:]), 3), round(np.std(WM_dice_scores[20:]), 3), round(np.std(V_dice_scores[20:]), 3), round(np.std(C_dice_scores[20:]), 3), round(np.std(DGM_dice_scores[20:]), 3), round(np.std(B_dice_scores[20:]), 3), round(std_V_volume_Necker, 3), round(std_V_error_Necker, 3)])
        if nb_labels == 1:
            writer.writerow(['Mean of Necker dataset', round(means_dice_Necker, 3), round(means_precision_Necker, 3), round(means_recall_Necker, 3), round(np.mean(V_dice_scores[20:]), 3), round(means_V_volume_Necker, 3), round(means_V_error_Necker, 3)])
            writer.writerow(['Std Dev of Necker dataset', round(std_dice_Necker, 3), round(std_precision_Necker, 3), round(std_recall_Necker, 3), round(np.std(V_dice_scores[20:]), 3), round(std_V_volume_Necker, 3), round(std_V_error_Necker, 3)])


    # Print the scores per patient
    print("#############################################")
    for i in range(nb_patients_test):
        patient_number = list_data[i][0][1][list_data[i][0][1].find('_') + 1:list_data[i][0][1].find('.')]
        print("Patient", patient_number, "Dice score:", dice_scores[i])
        print("Patient", patient_number, "Precision score:", precision_scores[i])
        print("Patient", patient_number, "Recall score:", recall_scores[i])
        print("Patient", patient_number, "Ventricles volume:", V_volumes[i])
        print("Patient", patient_number, "Ventricles error:", V_errors[i])
        print("#############################################")

        with open(scores_results_path + 'patient_scores.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if nb_labels == 7:
                writer.writerow([patient_number, round(dice_scores[i], 3), round(precision_scores[i], 3),
                                 round(recall_scores[i], 3), round(ECF_dice_scores[i], 3), round(GM_dice_scores[i], 3),
                                 round(WM_dice_scores[i], 3), round(V_dice_scores[i], 3), round(C_dice_scores[i], 3),
                                 round(DGM_dice_scores[i], 3), round(B_dice_scores[i], 3), round(V_volumes[i], 3),
                                round(V_errors[i], 3)])
            if nb_labels == 1:
                writer.writerow([patient_number, round(dice_scores[i], 3), round(precision_scores[i], 3),
                                 round(recall_scores[i], 3), round(V_dice_scores[i], 3), round(V_volumes[i], 3),
                                round(V_errors[i], 3)])

    # Calculate the average scores
    average_dice = np.mean(dice_scores)
    average_precision = np.mean(precision_scores)
    average_recall = np.mean(recall_scores)
    average_V_volume = np.mean(V_volumes)
    average_V_error = np.mean(V_errors)

    # Prepare means row and standard deviations row
    means_row = ["Total mean", round(average_dice, 3), round(average_precision, 3), round(average_recall, 3)]
    std_row = ["Total standard deviation", round(np.std(dice_scores), 3), round(np.std(precision_scores), 3),
               round(np.std(recall_scores), 3)]

    if nb_labels == 7:
        means_row.extend([
            round(np.mean(ECF_dice_scores), 3),
            round(np.mean(GM_dice_scores), 3),
            round(np.mean(WM_dice_scores), 3),
            round(np.mean(V_dice_scores), 3),
            round(np.mean(C_dice_scores), 3),
            round(np.mean(DGM_dice_scores), 3),
            round(np.mean(B_dice_scores), 3),
            round(np.mean(V_volumes), 3),
            round(np.mean(V_errors), 3)
        ])
        std_row.extend([
            round(np.std(ECF_dice_scores), 3),
            round(np.std(GM_dice_scores), 3),
            round(np.std(WM_dice_scores), 3),
            round(np.std(V_dice_scores), 3),
            round(np.std(C_dice_scores), 3),
            round(np.std(DGM_dice_scores), 3),
            round(np.std(B_dice_scores), 3),
            round(np.std(V_volumes), 3),
            round(np.std(V_errors), 3)
        ])

    if nb_labels == 1:
        means_row.extend([
            round(np.mean(V_dice_scores), 3),
            round(np.mean(V_volumes), 3),
            round(np.mean(V_errors), 3)
        ])
        std_row.extend([
            round(np.std(V_dice_scores), 3),
            round(np.std(V_volumes), 3),
            round(np.std(V_errors), 3)
        ])

    # Write the means row to the CSV file
    with open(scores_results_path + 'patient_scores.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(means_row)
        writer.writerow(std_row)

    print("Average Dice score:", average_dice)
    print("Average Precision score:", average_precision)
    print("Average Recall score:", average_recall)
    print("Average Ventricles volume:", np.mean(V_volumes))
    print("Average Ventricles error:", np.mean(V_errors))
