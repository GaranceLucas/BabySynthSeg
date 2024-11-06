# Imports
import os
import numpy as np
import nibabel as nib
import csv


########################################################################################################################
# Parameters

# Calculate the Dice score, the precision score, and the recall score for each class
# nb_labels = 7
nb_labels = 1

if nb_labels == 1:
        c = 1 # class_ = 1 for the Ventricles with SynthSeg
        v_label = 1
if nb_labels == 7:
        c = None
        v_label = 4


########################################################################################################################

# Paths
path_predicted_segmentation = '/Users/garance/Desktop/M2_DAC/STAGE/Test_results/SynthSeg/SynthSeg_FeTA_onlyV_1pre_100_5000steps_patch128_lr10-04/SynthSeg_segmentations_FeTA2024_test/pred_segmentations/'
path_manual_segmentation = '/Users/garance/Desktop/FeTA2024_test/3D/segmentation/'
scores_results_path = '/Users/garance/Desktop/M2_DAC/STAGE/Test_results/SynthSeg/SynthSeg_FeTA_onlyV_1pre_100_5000steps_patch128_lr10-04/scores/'

# Load the segmentation files
pred_segmentation_files = [f for f in os.listdir(path_predicted_segmentation) if f.endswith('.nii.gz')]
manual_segmentation_files = [f for f in os.listdir(path_manual_segmentation) if f.endswith('.nii.gz')]
pred_segmentation = [((nib.as_closest_canonical(nib.load(path_predicted_segmentation + seg)).get_fdata(), nib.load(path_predicted_segmentation + seg).affine), seg) for seg in pred_segmentation_files]
manual_segmentation = [((nib.as_closest_canonical(nib.load(path_manual_segmentation + seg)).get_fdata(), nib.load(path_manual_segmentation + seg).affine), seg) for seg in manual_segmentation_files]

# Construct a list of pairs pred,gt segmentations
list_pairs = []
for s in pred_segmentation:
     s_name = s[4:7]
     for m in manual_segmentation:
          m_name = m[4:7]
          if s_name == m_name:
              list_pairs.append((s, m))
              break
               
print("Number of pairs: ", len(list_pairs))

# Sort list_pairs by segmentation name
list_pairs = sorted(list_pairs, key=lambda x: int(x[0][1][4:7]))

nb_patients_test = len(pred_segmentation)


########################################################################################################################
# Functions

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

        print("Dice score for class", c, ":", dice)

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

# Create a csv file to store the scores
with open(scores_results_path + 'patient_scores.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    if nb_labels == 7:
        writer.writerow(
            ['Patient', 'Dice_score', 'Precision_score', 'Recall_score', 'ECF_Dice_score', 'GM_Dice_score',
                'WM_Dice_score', 'V_Dice_score', 'C_Dice_score', 'DGM_Dice_score', 'B_Dice_score', 'Ventricles_volume', 'Ventricles_error'])
    if nb_labels == 1:
            writer.writerow(['Patient', 'Dice_score', 'Precision_score', 'Recall_score', 'V_Dice_score', 'Ventricles_volume', 'Ventricles_error'])

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


num_patient = 0
for i in range(len(list_pairs)):
    seg_pred = list_pairs[i][0][0][0]
    seg_gt = list_pairs[i][1][0][0]
    seg_num = list_pairs[i][0][1][4:7]

    print("Patient ", seg_num)

    scores, dice = dice_score_multiclass(seg_pred, seg_gt, nb_labels, class_=c)
    precision = precision_score_multiclass(seg_pred, seg_gt, nb_labels, class_=c)
    recall = recall_score_multiclass(seg_pred, seg_gt, nb_labels, class_=c)
    ventricles_volume, ventricles_error = volume_ventricles(seg_pred, seg_gt, v_label)

    if nb_labels == 7:
        ECF_dice_scores[int(num_patient)-1]+=scores[0]
        GM_dice_scores[int(num_patient)-1]+=scores[1]
        WM_dice_scores[int(num_patient)-1]+=scores[2]
        V_dice_scores[int(num_patient)-1]+=scores[3]
        C_dice_scores[int(num_patient)-1]+=scores[4]
        DGM_dice_scores[int(num_patient)-1]+=scores[5]
        B_dice_scores[int(num_patient)-1]+=scores[6]
    if nb_labels == 1:
        V_dice_scores[int(num_patient)-1]+=scores[0]

    dice_scores[int(num_patient)-1]+= dice
    precision_scores[int(num_patient)-1]+=precision
    recall_scores[int(num_patient)-1]+=recall
    V_volumes[int(num_patient)-1]+=ventricles_volume
    V_errors[int(num_patient)-1]+=ventricles_error

    num_patient += 1

    # After processing the 10th subject, write the mean and std of the first 10 subjects (first test dataset)
    if num_patient == 10:
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
    if num_patient == 20:
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


print("#############################################")
for i in range(len(list_pairs)):
    patient_number = list_pairs[i][0][1][4:7]
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
