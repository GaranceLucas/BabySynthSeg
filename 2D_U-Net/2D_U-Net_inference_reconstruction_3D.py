########################################################################################################################
#                                                    IMPORTS                                                           #
########################################################################################################################

import os
import nibabel as nib
import numpy as np
import csv


########################################################################################################################
#                                                      PATHS                                                           #
########################################################################################################################

input_mri_path = ''   # Path to the input MRI files
segmentation_path = ''   # Path to the ground truth segmentations
segmentation_pred_path = ''   # Path to the predicted segmentations
reconstruction_3D_path = ''   # Path to save the reconstructed 3D segmentations
scores_results_path = ''   # Path to save the scores


########################################################################################################################
#                                                    PARAMETERS                                                        #
########################################################################################################################

nb_labels = 1
# nb_labels = 7


########################################################################################################################
#                                                    FUNCTIONS                                                         #
########################################################################################################################

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
#                                               3D RECONSTRUCTION                                                      #
########################################################################################################################

# Check if the weights directory exists
if not os.path.exists(reconstruction_3D_path):
    os.makedirs(reconstruction_3D_path)


segmentation_files = [f for f in os.listdir(segmentation_path) if f.endswith('.nii.gz')]
segmentation_pred = [f for f in os.listdir(segmentation_pred_path) if f.endswith('.nii.gz')]
input_mri_files = [f for f in os.listdir(input_mri_path) if f.endswith('.nii.gz')]

print("Number of segmentation files:", len(segmentation_files))

# Construct a dictionary with the patient number as key and the predicted segmentation files as value
patients_pred = {}
i=0
for file in segmentation_pred:
    i+=1
    num_patient = file[file.find('_') + 1:file.rfind('_')]   # To modify depending on the file names
    slice_number = file[-10:-7]   # To modify depending on the file names
    # load the segmentation prediction
    seg = ((nib.as_closest_canonical(nib.load(segmentation_pred_path + file)).get_fdata(), nib.load(segmentation_pred_path + file).affine), (num_patient, slice_number))
    # if the patient is not in the dictionary's keys, add it
    if num_patient not in patients_pred.keys():
        patients_pred[num_patient] = []
    # add the segmentation prediction to the corresponding patient
    patients_pred[num_patient].append(seg)


# Sort the segmentations by slice number for each patient
for patient in patients_pred.keys():
    # Sort by slice_number 
    patients_pred[patient].sort(key=lambda x: x[1][0][4:])   # To modify depending on the file names

# print all the keys of patients_pred
print(len(patients_pred.keys()))


# For each patient, reconstruct the 3D segmentation
for patient in patients_pred.keys():
    patient_num = patient[:-3]   # To modify depending on the file names
    # load the ground truth segmentation
    seg_gt_file = (nib.as_closest_canonical(nib.load(segmentation_path + 'segmentation_' + patient_num + '.nii.gz')).get_fdata(), nib.load(segmentation_path + 'segmentation_' + patient_num + '.nii.gz').affine)
    seg_gt = seg_gt_file[0]

    # Create a copy of the ground truth segmentation
    seg_gt_copy = np.zeros(seg_gt.shape)

    # For each predicted segmentation file of the patient, add the prediction to the corresponding slice of the ground truth segmentation
    for slice in patients_pred[patient]:
        slice_number = int(slice[1][1])
        seg_pred = slice[0][0]
        seg_pred = np.squeeze(seg_pred)
        seg_gt_copy[:, :, slice_number] = seg_pred

    # Retrieve the input MRI file
    mri_file = input_mri_path + 'mri_' + patient[:-3] + '.nii.gz'
    mri = (nib.as_closest_canonical(nib.load(mri_file)).get_fdata(), nib.load(mri_file).affine)

    # Save the reconstructed 3D segmentation
    seg_gt_copy_to_save = nib.Nifti1Image(seg_gt_copy, mri[1])  # use the affine matrix of the input MRI
    nib.save(seg_gt_copy_to_save, reconstruction_3D_path + 'reconstructed_segmentation_' + patient + '.nii.gz')

# Calculate the number of patients in the test set
nb_patients_test = len(segmentation_files)

# Sort the segmentations by patient number
segmentation_files.sort(key=lambda x: x[x.find('_') + 1:x.rfind('_')])


########################################################################################################################
#                                               CALCULATE SCORES                                                       #
########################################################################################################################

# Calculate the Dice score, precision and recall for each structures

# Construct a list of tuples with the ground truth and reconstructed segmentations
segmentations_pairs = []
for file in segmentation_files:
    num_patient = file[file.find('_') + 1:file.rfind('_')]   # To modify depending on the file names
    seg_gt = ((nib.as_closest_canonical(nib.load(segmentation_path + file)).get_fdata(), nib.load(segmentation_path + file).affine), num_patient)
    seg_reconstructed = ((nib.as_closest_canonical(nib.load(reconstruction_3D_path + 'reconstructed_segmentation_' + num_patient + '.nii.gz')).get_fdata(), nib.load(reconstruction_3D_path + 'reconstructed_segmentation_' + num_patient + '.nii.gz').affine), num_patient)
    segmentations_pairs.append((seg_gt, seg_reconstructed))

# Sort the segmentations by patient number
segmentations_pairs.sort(key=lambda x: int(x[0][1]))

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

# Create a csv file to store the scores
with open(scores_results_path + 'patient_scores.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    if nb_labels == 7:
        writer.writerow(['Patient', 'Dice_score', 'Precision_score', 'Recall_score', 'ECF_Dice_score', 'GM_Dice_score', 'WM_Dice_score', 'V_Dice_score', 'C_Dice_score', 'DGM_Dice_score', 'B_Dice_score', 'Ventricles_volume', 'Ventricles_error'])
    if nb_labels == 1:
        writer.writerow(['Patient', 'Dice_score', 'Precision_score', 'Recall_score', 'V_Dice_score', 'Ventricles_volume', 'Ventricles_error'])
writer.writerow(['Patient', 'Dice_score', 'Precision_score', 'Recall_score', 'V_Dice_score'])


scores_patient = []
dice_patient = []
precision_patient = []
recall_patient = []
ventricles_volume_patient = []
ventricles_error_patient = []

numero = 0

for pair in segmentations_pairs:
    seg_gt = pair[0][0][0]
    seg_reconstructed = pair[1][0][0]
    num_patient = pair[0][1]

    # Compute scores
    if nb_labels == 1:
        c = 1  # class_ = 1 for the Ventricles
        v_label = 1
    if nb_labels == 7:
        v_label = 4
        c = None

    scores, dice = dice_score_multiclass(seg_reconstructed, seg_gt, nb_labels, class_=c)  
    precision = precision_score_multiclass(seg_reconstructed, seg_gt, nb_labels, class_=c)
    recall = recall_score_multiclass(seg_reconstructed, seg_gt, nb_labels, class_=c)
    ventricles_volume, ventricles_error = volume_ventricles(seg_reconstructed, seg_gt, v_label)
    scores_patient.append(scores)
    dice_patient.append(dice)
    precision_patient.append(precision)
    recall_patient.append(recall)
    ventricles_volume_patient.append(ventricles_volume)
    ventricles_error_patient.append(ventricles_error)

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

    with open(scores_results_path + 'patient_scores.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if nb_labels == 7:
            writer.writerow([num_patient, round(dice_scores[numero], 3), round(precision_scores[numero], 3), round(recall_scores[numero], 3), round(ECF_dice_scores[numero], 3), round(GM_dice_scores[numero], 3), round(WM_dice_scores[numero], 3), round(V_dice_scores[numero], 3), round(C_dice_scores[numero], 3), round(DGM_dice_scores[numero], 3), round(B_dice_scores[numero], 3), round(V_volumes[numero], 3), round(V_errors[numero], 3)])
        if nb_labels == 1:
            writer.writerow([num_patient, round(dice_scores[numero], 3), round(precision_scores[numero], 3), round(recall_scores[numero], 3), round(V_dice_scores[numero], 3), round(V_volumes[numero], 3), round(V_errors[numero], 3)])

    numero += 1

# Calculate the average scores
average_dice = np.mean(dice_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)

print("Average Dice score:", average_dice)
print("Average Precision score:", average_precision)
print("Average Recall score:", average_recall)