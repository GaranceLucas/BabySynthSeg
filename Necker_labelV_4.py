########################################################################################################################
################################################## IMPORTS #############################################################
########################################################################################################################
import os
import nibabel as nib
import numpy as np

########################################################################################################################
#################################################### DATA ##############################################################
########################################################################################################################

# Paths
seg_path = '/Users/garance/Desktop/M2_DAC/STAGE/data/Necker/test_vierge/segmentation/'
results_path = '/Users/garance/Desktop/M2_DAC/STAGE/data/Necker/test_vierge_propre/'

seg_files = [file for file in os.listdir(seg_path) if file.endswith('.nii.gz')]
segmentations = [((nib.as_closest_canonical(nib.load(seg_path + file)).get_fdata(), nib.load(seg_path + file).affine), file)
        for file in seg_files]

print("Number of segmentations:", len(segmentations))

# For each segmentation, check if there is a label 1
for seg in segmentations:
    if (np.sum(seg[0][0] == 1) > 0):
        # All the voxels with label 4 are set to 1
        seg[0][0][seg[0][0] == 1] = 4
        # Save the new segmentation
        nib.save(nib.Nifti1Image(seg[0][0], seg[0][1]), results_path + seg[1]) 
        print("Segmentation of file", seg[1], "has been modified")
    else:
        # Save the segmentation
        nib.save(nib.Nifti1Image(seg[0][0], seg[0][1]), results_path + seg[1])