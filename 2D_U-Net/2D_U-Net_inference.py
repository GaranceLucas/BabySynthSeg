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
import csv


########################################################################################################################
#                                                 VARIABLES                                                            #
########################################################################################################################

nb_labels = 7  # Number of labels in the segmentation
# nb_labels = 1

# Paths
data_path = ''   # Path to the MRI data
segmentation_path = ''   # Path to the segmentation data
weights_path = ''   # Path to the weights of the model
seg_results_path = ''   # Path to save the segmentation results
scores_results_path = ''   # Path to save the scores

# Device
# device ="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"

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
data = [((nib.as_closest_canonical(nib.load(data_path + file)).get_fdata(), nib.load(data_path + file).affine), file) for file in data_files]
segmentation = [nib.as_closest_canonical(nib.load(segmentation_path + file)).get_fdata() for file in segmentation_files]

nb_patients_test = len(data)

patient_dic = {}

for i in range(nb_patients_test):
    patient = data[i][1][data[i][1].find('_') + 1:data[i][1].rfind('_')]   # To modify depending on the file names
    if patient in patient_dic:
        patient_dic[patient].append(data[i])
    else:
        patient_dic[patient] = [data[i]]

list_patients = []
for patient, slices in patient_dic.items():
    list_patients.append(slices)

print('Number of test patients', len(list_patients))

# Sort the list of patients by patient number
list_patients = sorted(list_patients, key=lambda x: x[0][1][4:])   # To modify depending on the file names


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


def get_loader(dataset, batch_size):
    """Builds and returns Dataloader."""

    data_loader = d.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=False)
    return data_loader


def dice_score_multiclass(prediction, ground_truth, num_classes, smooth=1e-6, class_ = None):
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
        else :
            ground_truth_c = (ground_truth == class_).astype(int)
        
        intersection = np.sum(prediction_c * ground_truth_c)
        union = np.sum(prediction_c) + np.sum(ground_truth_c)
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)

        # print("Dice score for class", c, ":", dice)
    
    return dice_scores, np.mean(dice_scores)


def precision_score_multiclass(prediction, ground_truth, num_classes, smooth=1e-6, class_ = None):
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


def recall_score_multiclass(prediction, ground_truth, num_classes, smooth=1e-6, class_ = None):
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
#                                                   MAIN                                                               #
########################################################################################################################

if __name__ == '__main__':

    # Create a csv file to store the scores
    with open(scores_results_path + 'patient_scores.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        if nb_labels == 7:
            writer.writerow(['Patient', 'Dice_score', 'Precision_score', 'Recall_score', 'ECF_Dice_score', 'GM_Dice_score', 'WM_Dice_score', 'V_Dice_score', 'C_Dice_score', 'DGM_Dice_score', 'B_Dice_score', 'Ventricles_volume', 'Ventricles_error'])
        if nb_labels == 1:
            writer.writerow(['Patient', 'Dice_score', 'Precision_score', 'Recall_score', 'V_Dice_score', 'Ventricles_volume', 'Ventricles_error'])

    # Defining model
    model = U_Net(input_ch, output_ch).to(device)

    # Initialize scores 
    dice_scores = np.zeros(nb_patients_test)  # Dice score
    ECF_dice_scores = np.zeros(nb_patients_test)  # Dice for External Cerebrospinal Fluid
    GM_dice_scores = np.zeros(nb_patients_test)  # Dice for Grey Matter
    WM_dice_scores = np.zeros(nb_patients_test)  # Dice for White Matter
    V_dice_scores = np.zeros(nb_patients_test)  # Dice for Ventricles
    C_dice_scores = np.zeros(nb_patients_test)  # Dice for Cerebellum
    DGM_dice_scores = np.zeros(nb_patients_test)  # Dice for Deep Grey Matter
    B_dice_scores = np.zeros(nb_patients_test)  # Dice for Brainstem
    precision_scores = np.zeros(nb_patients_test)  # Precision score
    recall_scores = np.zeros(nb_patients_test)  # Recall score
    nb_patients = np.zeros(nb_patients_test)  # Number of patients
    V_volumes = np.zeros(nb_patients_test)  # Volume of the Ventricles
    V_errors = np.zeros(nb_patients_test)  # Error for the volume of the Ventricles


    # Load pre-trained model
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    # Data loaders
    # dataloader = get_loader(data, 1)

    list_dataloaders = []
    for patient in list_patients:
        list_dataloaders.append(get_loader(patient, 1))

    numero = 0
    for loader in tqdm(list_dataloaders):
        list_pred_slices = []

        for mri in loader:
            num_patient = mri[1][0][4:]   # To modify depending on the file names
            # print("Patient number:", num_patient)

            x = mri[0][0]
            x = x.unsqueeze(1).float()

            pred = model(x)
            pred = torch.argmax(pred, dim=1)  # Get the class with the highest probability

            # Remove the batch dimension
            pred = np.squeeze(pred)

            # Convert the tensor to a numpy array
            pred = pred.detach().numpy()

            # Save the segmentation result
            pred_to_save = nib.Nifti1Image(np.expand_dims(pred, -1).astype(np.int8), mri[0][1].squeeze())
            nib.save(pred_to_save, seg_results_path + mri[1][0])

            list_pred_slices.append((pred, mri[1]))

        # Sort the slices in the right order
        list_pred_slices = sorted(list_pred_slices, key=lambda x: x[1][0][4:])   # To modify depending on the file names

        # Load ground truth segmentations
        list_seg_slices = []
        scores_patient = []
        dice_patient = []
        precision_patient = []
        recall_patient = []
        ventricles_volume_patient = []
        ventricles_error_patient = []

        for pred in list_pred_slices:
            prediction = pred[0]
            mri_name = pred[1][0]
            mri_number = mri_name[4:]
            seg_name = 'segmentation_' + mri_number
            seg = [segmentation[i] for i in range(len(segmentation)) if segmentation_files[i] == seg_name][0]
            if nb_labels == 1:
                # put all the labels different from 4 to 0 and the label 4 to 1
                seg[seg != 4] = 0
                seg[seg == 4] = 1

            # Compute scores
            if nb_labels == 1:
                c = 1  # class_ = 1 for the Ventricles
                v_label = 1
            if nb_labels == 7:
                v_label = 4
                c = None

            scores, dice = dice_score_multiclass(prediction, seg, nb_labels, class_=c)  
            precision = precision_score_multiclass(prediction, seg, nb_labels, class_=c)
            recall = recall_score_multiclass(prediction, seg, nb_labels, class_=c)
            ventricles_volume, ventricles_error = volume_ventricles(prediction, seg, v_label)
            scores_patient.append(scores)
            dice_patient.append(dice)
            precision_patient.append(precision)
            recall_patient.append(recall)
            ventricles_volume_patient.append(ventricles_volume)
            ventricles_error_patient.append(ventricles_error)

        scores_patient = list(zip(*scores_patient))  # Transpose the list in order to have the scores for each class in the same list
        scores_patient = [np.mean(score_type) for score_type in scores_patient]
        # scores_patient = np.mean(scores_patient, axis=0)
        dice_patient = np.mean(dice_patient)

        precision_patient = np.mean(precision_patient)
        recall_patient = np.mean(recall_patient)
        ventricles_volume_patient = np.mean(ventricles_volume_patient)
        ventricles_error_patient = np.mean(ventricles_error_patient)

        print("Patient", num_patient, "Dice score:", dice_patient)
        print("Patient", num_patient, "Precision score:", precision_patient)
        print("Patient", num_patient, "Recall score:", recall_patient)
        print("Patient", num_patient, "Ventricles volume:", ventricles_volume_patient)
        print("Patient", num_patient, "Ventricles error:", ventricles_error_patient)
        print("#############################################")

        if nb_labels == 7:
            ECF_dice_scores[numero] = scores_patient[0]
            GM_dice_scores[numero] = scores_patient[1]
            WM_dice_scores[numero] = scores_patient[2]
            V_dice_scores[numero] = scores_patient[3]
            C_dice_scores[numero] = scores_patient[4]
            DGM_dice_scores[numero] = scores_patient[5]
            B_dice_scores[numero] = scores_patient[6]
        if nb_labels == 1:
            V_dice_scores[numero] = scores_patient[0]

        dice_scores[numero] = dice_patient
        precision_scores[numero] = precision_patient
        recall_scores[numero] = recall_patient
        V_volumes[numero] = ventricles_volume_patient
        V_errors[numero] = ventricles_error_patient

        # Write the scores of the patient in a csv file
        with open(scores_results_path + 'patient_scores.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if nb_labels == 7:
                writer.writerow([num_patient, round(dice_scores[numero], 3), round(precision_scores[numero], 3), round(recall_scores[numero], 3), round(ECF_dice_scores[numero], 3), round(GM_dice_scores[numero], 3), round(WM_dice_scores[numero], 3), round(V_dice_scores[numero], 3), round(C_dice_scores[numero], 3), round(DGM_dice_scores[numero], 3), round(B_dice_scores[numero], 3), round(V_volumes[numero], 3), round(V_errors[numero], 3)])
            if nb_labels == 1:
                writer.writerow([num_patient, round(dice_scores[numero], 3), round(precision_scores[numero], 3), round(recall_scores[numero], 3), round(V_dice_scores[numero], 3), round(V_volumes[numero], 3), round(V_errors[numero], 3)])


        # Write the scores for the Zurich dataset
        if numero == 9:
            means_dice_Zurich = np.mean(dice_scores[:10])
            std_dice_Zurich = np.std(dice_scores[:10])
            means_precision_Zurich = np.mean(precision_scores[:10])
            std_precision_Zurich = np.std(precision_scores[:10])
            means_recall_Zurich = np.mean(recall_scores[:10])
            std_recall_Zurich = np.std(recall_scores[:10])
            means_V_volume_Zurich = np.mean(V_volumes[:10])
            std_V_volume_Zurich = np.std(V_volumes[:10])
            means_V_error_Zurich = np.mean(V_errors[:10])
            std_V_error_Zurich = np.std(V_errors[:10])

            with open(scores_results_path + 'patient_scores.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                if nb_labels == 7:
                    writer.writerow(['Mean of FeTA Zurich dataset', round(means_dice_Zurich, 3), round(means_precision_Zurich, 3), round(means_recall_Zurich, 3), round(np.mean(ECF_dice_scores[:10]), 3), round(np.mean(GM_dice_scores[:10]), 3), round(np.mean(WM_dice_scores[:10]), 3), round(np.mean(V_dice_scores[:10]), 3), round(np.mean(C_dice_scores[:10]), 3), round(np.mean(DGM_dice_scores[:10]), 3), round(np.mean(B_dice_scores[:10]), 3), round(means_V_volume_Zurich, 3), round(means_V_error_Zurich, 3)])
                    writer.writerow(['Std Dev of FeTA Zurich dataset', round(std_dice_Zurich, 3), round(std_precision_Zurich, 3), round(std_recall_Zurich, 3), round(np.std(ECF_dice_scores[:10]), 3), round(np.std(GM_dice_scores[:10]), 3), round(np.std(WM_dice_scores[:10]), 3), round(np.std(V_dice_scores[:10]), 3), round(np.std(C_dice_scores[:10]), 3), round(np.std(DGM_dice_scores[:10]), 3), round(np.std(B_dice_scores[:10]), 3), round(std_V_volume_Zurich, 3), round(std_V_error_Zurich, 3)])
                if nb_labels == 1:
                    writer.writerow(['Mean of FeTA Zurich dataset', round(means_dice_Zurich, 3), round(means_precision_Zurich, 3), round(means_recall_Zurich, 3), round(np.mean(V_dice_scores[:10]), 3), round(means_V_volume_Zurich, 3), round(means_V_error_Zurich, 3)])
                    writer.writerow(['Std Dev of FeTA Zurich dataset', round(std_dice_Zurich, 3), round(std_precision_Zurich, 3), round(std_recall_Zurich, 3), round(np.std(V_dice_scores[:10]), 3), round(std_V_volume_Zurich, 3), round(std_V_error_Zurich, 3)])


        # Write the scores for the Vienna dataset
        if numero == 19:
            means_dice_Vienna = np.mean(dice_scores[10:20])
            std_dice_Vienna = np.std(dice_scores[10:20])
            means_precision_Vienna = np.mean(precision_scores[10:20])
            std_precision_Vienna = np.std(precision_scores[10:20])
            means_recall_Vienna= np.mean(recall_scores[10:20])
            std_recall_Vienna = np.std(recall_scores[10:20])
            means_V_volume_Vienna = np.mean(V_volumes[10:20])
            std_V_volume_Vienna = np.std(V_volumes[10:20])
            means_V_error_Vienna = np.mean(V_errors[10:20])
            std_V_error_Vienna = np.std(V_errors[10:20])

            with open(scores_results_path + 'patient_scores.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                if nb_labels == 7:
                    writer.writerow(['Mean of FeTA Vienna dataset', round(means_dice_Vienna, 3), round(means_precision_Vienna, 3), round(means_recall_Vienna, 3), round(np.mean(ECF_dice_scores[10:20]), 3), round(np.mean(GM_dice_scores[10:20]), 3), round(np.mean(WM_dice_scores[10:20]), 3), round(np.mean(V_dice_scores[10:20]), 3), round(np.mean(C_dice_scores[10:20]), 3), round(np.mean(DGM_dice_scores[10:20]), 3), round(np.mean(B_dice_scores[10:20]), 3), round(means_V_volume_Vienna, 3), round(means_V_error_Vienna, 3)])
                    writer.writerow(['Std Dev of FeTA Vienna dataset', round(std_dice_Vienna, 3), round(std_precision_Vienna, 3), round(std_recall_Vienna, 3), round(np.std(ECF_dice_scores[10:20]), 3), round(np.std(GM_dice_scores[10:20]), 3), round(np.std(WM_dice_scores[10:20]), 3), round(np.std(V_dice_scores[10:20]), 3), round(np.std(C_dice_scores[10:20]), 3), round(np.std(DGM_dice_scores[10:20]), 3), round(np.std(B_dice_scores[10:20]), 3), round(std_V_volume_Vienna, 3), round(std_V_error_Vienna, 3)])
                if nb_labels == 1:
                    writer.writerow(['Mean of FeTA Vienna dataset', round(means_dice_Vienna, 3), round(means_precision_Vienna, 3), round(means_recall_Vienna, 3), round(np.mean(V_dice_scores[10:20]), 3), round(means_V_volume_Vienna, 3), round(means_V_error_Vienna, 3)])
                    writer.writerow(['Std Dev of FeTA Vienna dataset', round(std_dice_Vienna, 3), round(std_precision_Vienna, 3), round(std_recall_Vienna, 3), round(np.std(V_dice_scores[10:20]), 3), round(std_V_volume_Vienna, 3), round(std_V_error_Vienna, 3)])

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


    # Calculate the average scores
    average_dice = np.mean(dice_scores)
    average_precision = np.mean(precision_scores)
    average_recall = np.mean(recall_scores)
    average_V_volume = np.mean(V_volumes)
    average_V_error = np.mean(V_errors)

    # Calculate the standard deviation of the scores
    std_dice = np.std(dice_scores)
    std_precision = np.std(precision_scores)
    std_recall = np.std(recall_scores)
    std_V_volume = np.std(V_volumes)
    std_V_error = np.std(V_errors)

    # Write the average scores in a csv file in a new row
    with open(scores_results_path + 'patient_scores.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if nb_labels == 7:
            writer.writerow(['Total Mean', round(average_dice, 3), round(average_precision, 3), round(average_recall, 3), round(np.mean(ECF_dice_scores), 3), round(np.mean(GM_dice_scores), 3), round(np.mean(WM_dice_scores), 3), round(np.mean(V_dice_scores), 3), round(np.mean(C_dice_scores), 3), round(np.mean(DGM_dice_scores), 3), round(np.mean(B_dice_scores), 3), round(average_V_volume, 3), round(average_V_error, 3)])
            writer.writerow(['Total Standard Deviation', round(std_dice, 3), round(std_precision, 3), round(std_recall, 3), round(np.std(ECF_dice_scores), 3), round(np.std(GM_dice_scores), 3), round(np.std(WM_dice_scores), 3), round(np.std(V_dice_scores), 3), round(np.std(C_dice_scores), 3), round(np.std(DGM_dice_scores), 3), round(np.std(B_dice_scores), 3), round(std_V_volume, 3), round(std_V_error, 3)])
        if nb_labels == 1:
            writer.writerow(['Total Mean', round(average_dice, 3), round(average_precision, 3), round(average_recall, 3), round(np.mean(V_dice_scores), 3), round(average_V_volume, 3), round(average_V_error, 3)])
            writer.writerow(['Total Standard Deviation', round(std_dice, 3), round(std_precision, 3), round(std_recall, 3), round(np.std(V_dice_scores), 3), round(std_V_volume, 3), round(std_V_error, 3)])

    print("Average Dice score:", average_dice)
    print("Average Precision score:", average_precision)
    print("Average Recall score:", average_recall)
    print("Average Ventricles volume:", average_V_volume)
    print("Average Ventricles error:", average_V_error)