
# Tract based white matter hyperintensity patterns in Systemic Lupus Erythematosus.

README file for the manuscript "Tract based white matter hyperintensity patterns in Systemic Lupus Erythematosus". In this work a method was created to characterize WMH in a two-site Systemic Lupus Erythematosus (SLE). The core step consists of cluster analysis (hierarchical clustering) to identify clusters which can be associated to specific WM tracts. 

# Installation

LST-LGA was used as a non-compiled packaged and copied into the SPM12 toolbox folder.


# Preprocessing
<img src="images/pp_image.png" width="200" height="200" />

An overview of all preprocessing steps can be seen in the image above. Those steps are separated in four different parts. 


## part 1
First, all subjects with a 3D FLAIR image (Leiden cohort only) were reoriented to the standard orientation (MNI-tempate image space) and coregistered to the T1-space. This steps was necessary due to the fact that for those subjects the LST-LGA algorithm failed the coregistration and therefore also the lesion segmentation. An example of the commands can be seen here:
```bash
fslreorient2std FLAIR.nii.gz FLAIR_reor.nii.gz
flirt -in FLAIR_reor.nii.gz -ref T1.nii.gz -out FLAIR_reor_flirt.nii.gz
```

## part 2
Part 2 extracts the lesion probability maps from all subjects and used the original FLAIR (and reoriented and coregistered 3D FLAIRS) as well as T1 images as input. The resulted lesion probability maps as well as the FLAIR images are coregistered to T1-space. All parameters wer used as default, including kappa= 0.3.

## part 3
The T1-images were brain extracted and the brain masks were applied to the lesion probability maps. Latter, were transfered to MNI-space using the transformation files from the MNI-registration of the T1 image (see preprocessing_pipeline_3.py).

## part 4
The number and volume of the WMH for each subject were extracted for each subject in T1- and MNI-space. Additionally, the lesion probability maps in MNI-space were masked with the JHU WM tract atlas which resulted in a WMH burden for each of the 20 WM tracts for each subject (see preprocessing_pipeline_4.py).


# Quality Assessment
For manually controll a quality assessment was applied evaluating the following preprocessing steps for each subject. Every step was saved as an PNG-image in lightbox view (see preprocessing_QA.py):
* brain extraction
* FLAIR coregistration
* lesion probability map in T1-space
* lesion probability map in MNI-space
* T1 registration to MNI-space


# Analysis
Before cluster analysis, the WMH burden of each WM tract was l2-normalized. Additionally, the cluster performance evaluation, the volumentric as well as the statistical analysis was applied (cluster_analysis.py).


# Coresponding author: 
Theodor Rumetshofer, Department of Clinical Science Lund / Diagnostic Radiology, Lund University/SUS/Lund, 22185 Lund, Sweden, theodor.rumetshofer@med.lu.se, OCRID: 0000-0002-0778-0703

(Theodor Rumetshofer and Francesca Inglese contributed equally to this work)


# Software versions
The following software packages were used:
* Matlab v2016b
* SPM12
* LST-LGA 3.0
* Python Anaconda v3.6.7
* Nipype 1.5.1
* FSL 5.0.10
* ANTs 1.9.2
* scikit-learn 0.20.3
* Scipy 1.2.1
* statsmodels 0.10.1


# Copyright
Copyright © Theodor Rumetshofer, Department of Clinical Science/Diagnostic Radiology, Lund University, Sweden 
