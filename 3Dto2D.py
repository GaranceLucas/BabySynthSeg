########################################################################################################################
#                                                  IMPORTS                                                             #
########################################################################################################################

import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random


########################################################################################################################
#                                                   DATA                                                               #
########################################################################################################################

# Paths
data_path = ''   # Path to the data 
segmentation_path = ''   # Path to the segmentations
results_path = ''   # Path to save the 2D slices

# Find all the files in the data and segmentation folders which end with .nii.gz
data_files = [file for file in os.listdir(data_path) if file.endswith('.nii.gz')]

# Load 3D MRI image and corresponding segmentation
data = []

for file in data_files:
    data.append((nib.as_closest_canonical(nib.load(data_path + file)).get_fdata(), nib.load(data_path + file).affine))


########################################################################################################################
#                                                 FUNCTIONS & CLASSES                                                  #
########################################################################################################################

# def generate_random_selection(min, max, n):
#     selection = set()
#     available_numbers = set(range(min, max))
    
#     while len(selection) < n:
#         num = random.choice(list(available_numbers))
#         selection.add(num)
#         available_numbers.difference_update(range(num - 2, num + 3))  # 3 slices gap
#         if len(available_numbers) < 3:
#             break
    
#     selection = list(selection)
#     selection.sort()
    
#     return selection


def convertion3Dto2D(mri, segmentation=None):
    print('------------ Number of slices in mri :', mri[0].shape[2])

    # Iterate through each axial slice in the 3D volume
    for slice in range(mri[0].shape[2]):
        
        # If the slice is in the list of slices to save
        # if slice in slices_to_save:
        if segmentation is not None:
            if segmentation[0].shape[2] == mri[0].shape[2]:

                # Extract 2D slices from both MRI and segmentation
                img_slice = mri[0][:, :, slice]
                if segmentation is not None:
                    seg_slice = segmentation[0][:, :, slice]

                # Save the results
                # Create the folder if it does not exist
                if not os.path.exists(results_path):
                    os.makedirs(results_path)
            
                # Save the cropped data and segmentation
                nib.save(nib.Nifti1Image(img_slice, affine=np.eye(4)), results_path + 'mri_' + file_number
                        + '_2D' + '_slice' + str(slice) + '.nii.gz')

                if segmentation is not None:
                    nib.save(nib.Nifti1Image(seg_slice, affine=np.eye(4)), results_path + 'segmentation_' + file_number
                            + '_2D' + '_slice' + str(slice) + '.nii.gz')
                

########################################################################################################################
#                                                   MAIN                                                               #
########################################################################################################################

# Loop over all the data
for i in tqdm(range(len(data))):

    # Retrieve the name of the mri file
    file_number = data_files[i][:-11]   # To modify depending on the file names
    # print('File number:', file_number)

    # If the mri has a corresponding segmentation in the segmentation folder
    segmentation_file = segmentation_path + file_number + '_segmentation'  + '.nii.gz'  # To modify depending on the file names
    if os.path.exists(segmentation_file):
        segmentation = nib.as_closest_canonical(nib.load(segmentation_file)).get_fdata(), nib.load(segmentation_file).affine
        print('Segmentation found for file:', file_number)
    else :
        segmentation = None




    # Convert the 3D data into 2D
    convertion3Dto2D(data[i], segmentation)


    print('----------- Conversion 3D to 2D done for', file_number)



