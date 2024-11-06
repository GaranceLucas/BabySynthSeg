########################################################################################################################
#                                                  IMPORTS                                                             #
########################################################################################################################
import os
import nibabel as nib
import numpy as np

########################################################################################################################
#                                                   DATA                                                               #
########################################################################################################################

# Paths
data_path = ''   # Path to the data
segmentation_path = ''   # Path to the segmentations
results_path = ''   # Path to save the results

# Find all the files in the data and segmentation folders which end with .nii.gz
data_files = [f for f in os.listdir(data_path) if f.endswith('.nii.gz')]  
segmentation_files = [f for f in os.listdir(segmentation_path) if f.endswith('.nii.gz')]

# Load the data
data = []
segmentation = []

for file in data_files:
    # append the data and the file name
    data.append(((nib.as_closest_canonical(nib.load(data_path + file)).get_fdata(), nib.load(data_path + file).affine), file))

for file in segmentation_files:
    segmentation.append(((nib.as_closest_canonical(nib.load(segmentation_path + file)).get_fdata(), nib.load(segmentation_path + file).affine), file))

for i in range(len(data)):
    mri_name = data[i][1][8:]
    seg_name = [seg for seg in segmentation_files if seg[17:] == mri_name][0]
    seg = (nib.as_closest_canonical(nib.load(segmentation_path + seg_name)).get_fdata(), nib.load(segmentation_path + seg_name).affine)

    # retrive the minimum pixel value in the MRI
    min_value = np.min(data[i][0][0])

    # retrive the pixel location where the segmentation is null
    null_pixel = np.where(seg[0] == 0)
    data[i][0][0][null_pixel] = min_value

    # save the modified mri
    mri = nib.Nifti1Image(data[i][0][0], affine=np.eye(4))
    nib.save(mri, results_path + data[i][1])

    # save the segmentation with its affine matrix = np.eye(4)
    segmentation_to_save = nib.Nifti1Image(seg[0], affine=np.eye(4))
    nib.save(segmentation_to_save, results_path + seg_name)
