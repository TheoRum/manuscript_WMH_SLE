#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

 | Title:
     Quality assessment of the preprocessing pipeline
     
 | Date:
     2021-02-01
     
 | Author(s):
     Theodor Rumetshofer

 | Description:    
     This script generates images for each subject for manual quality assessment.

 | List of functions:
     No user defined functions are used in the program.

 | List of "non standard" modules:
     None

 | Procedure:
     1) T1 brain extraction
     2) FLAIR coregistration to T1 (done by LST-LGA)
     3) lesion map in T1-space
     4) lesion map in MNI-space
     5) T1 contours over MNI-space to check the transformation  

 | Usage:
     ./preprocessing_QA.py

"""


from __future__ import (print_function, division, unicode_literals, absolute_import)                
import os
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
from matplotlib import pyplot as plt
from skimage import filters
from glob import glob as gl


## PATH
PATH = '/Users/data/_analysis/' # mac
img_path = PATH+'SLE/'

# LST - LGA
experiment_dir = img_path+'/data'
output_dir = experiment_dir+'/_LGA/'
thresh_lesion = 0.5


#############################################################################

#_________________________
# define folder structure \__________________________________________________

# lesion map T1-space
df_les_t1 = output_dir+'_outputdir/bet/*/*/*/ples_lga_*_masked.nii.gz' # LGA
df_les_t1 = experiment_dir+'/*/ples_lga_*_masked.nii.gz' # LGA
df_les_t1 = gl(df_les_t1)
df_les_t1 = np.sort(df_les_t1)
subject_list = [item.split('/ples_')[0].split('_')[-1] for item in df_les_t1]
df_les_t1 = pd.DataFrame(index=subject_list, columns=['les_t1'], data=df_les_t1)

# T1
df_t1 = experiment_dir+'/*/T1.nii.gz'
df_t1 = gl(df_t1)
df_t1 = np.sort(df_t1)
subject_list = [item.split('/T1.')[0].split('/')[-1] for item in df_t1]
df_t1 = pd.DataFrame(index=subject_list, columns=['t1'], data=df_t1)

# Brain mask
df_t1_brain_mask = experiment_dir+'/*/T1_brain_mask.nii.gz'
df_t1_brain_mask = gl(df_t1_brain_mask)
df_t1_brain_mask = np.sort(df_t1_brain_mask)
subject_list = [item.split('/T1_brain_')[0].split('/')[-1] for item in df_t1_brain_mask]
df_t1_brain_mask = pd.DataFrame(index=subject_list, columns=['brain_mask'], data=df_t1_brain_mask)

# T1 brain 
df_t1_brain = experiment_dir+'/*/T1_brain.nii.gz'
df_t1_brain = gl(df_t1_brain)
df_t1_brain = np.sort(df_t1_brain)
subject_list = [item.split('/T1_brain')[0].split('/')[-1] for item in df_t1_brain]
df_t1_brain = pd.DataFrame(index=subject_list, columns=['t1_brain'], data=df_t1_brain)

# FLAIR coregistered to T1
df_rmflair = experiment_dir+'/*/rFLAIR.nii'
df_rmflair = gl(df_rmflair)
df_rmflair = np.sort(df_rmflair)
subject_list = [item.split('/rFLAIR')[0].split('/')[-1] for item in df_rmflair]
df_rmflair = pd.DataFrame(index=subject_list, columns=['rmflair'], data=df_rmflair)

# T1 tranformed (mni-space)
df_t1_reg = experiment_dir+'/*/transform_Warped.nii.gz'
df_t1_reg = gl(df_t1_reg)
df_t1_reg = np.sort(df_t1_reg)
subject_list = [item.split('/transform_Warped')[0].split('/')[-1] for item in df_t1_reg]
df_t1_reg = pd.DataFrame(index=subject_list, columns=['t1_reg'], data=df_t1_reg)

# lesion map MNI-space
df_les_mni = output_dir+'_outputdir/antsapply/*/*/*/ples_lga_*_masked_trans.nii.gz' # LGA
df_les_mni = experiment_dir+'/*/ples_lga_*_masked_trans.nii.gz' # LGA
df_les_mni = gl(df_les_mni)
df_les_mni = np.sort(df_les_mni)
subject_list = [item.split('/ples_')[0].split('_subject_id_')[-1] for item in df_les_mni]
df_les_mni = pd.DataFrame(index=subject_list, columns=['les_mni'], data=df_les_mni)

# MNI standard
mni_standard = img_path+'MNI_template/MNI152_T1_1mm_brain.nii.gz'


# put together
df_all = pd.concat([df_les_t1, df_t1, df_t1_brain_mask, df_t1_brain,  
                    df_rmflair, df_t1_reg, df_les_mni], axis=1, join='inner')


df_all.insert(0, value=range(0,df_all.shape[0]), column='num_subjects')


#___________________________________________________
# QA - create the images and save them as png-files \__________________________

# define the output directory 
folder_save = '_qa/'
path_save = output_dir+folder_save 
if not os.path.exists(path_save):
    new_folder = os.makedirs(path_save)   

# run the loop over each subject
print('>>> QA is running')
for item in df_all.index:

    df = df_all.loc[item]

    print('  |subject:', df.name)


    #_____________
    # A) T1-space \_______________________________________________________________
    
    ### ==> brain extraction
    title_ = df.name+'_a-bet-t1'
    title_save = path_save+title_
    
    a = nib.load(df['t1']).get_fdata()
    b = nib.load(df['brain_mask']).get_fdata()
    
    my_cmap = plt.cm.spring
    my_cmap.set_under(color='white', alpha='0')
    
    fig, axes = plt.subplots(nrows=4, ncols=6, sharex='none', sharey='none', figsize=(8,5.5))
    plt.subplots_adjust(hspace=0, wspace=0, top = 0.90, bottom = 0, right = 1, left = 0)
    fig.suptitle(title_, fontsize=15, color='w')
    axes = axes.flatten()
    slices = np.linspace(10, a.shape[-1]-10, num=len(axes), dtype=int)
    
    for axis, z in zip(axes,slices):
        # flip
        axis.imshow(np.flip(sp.ndimage.rotate(a[:,:,z],90), axis=1), cmap='gray')
        axis.imshow(np.flip(sp.ndimage.rotate(b[:,:,z],90), axis=1), cmap=my_cmap, vmin=0, vmax=1, alpha=0.2)
    
        axis.set_axis_off()

    fig.text(-0.02, 0.5, 'LEFT', va='center', rotation='vertical', color='w')
    plt.savefig(title_save, dpi=150, facecolor='k', bbox_inches = 'tight', pad_inches = 0)
     
    
    ### ==> FLAIR over T1   
    title_ = df.name+'_b-flair-t1-coreg'
    title_save = path_save+title_
    
    a = nib.load(df['t1']).get_fdata()
    b = nib.load(df['rmflair']).get_fdata()
    
    b[np.isnan(b)] = 0
    b = b.astype(float)
    
    fig, axes = plt.subplots(nrows=4, ncols=6, sharex='none', sharey='none', figsize=(8,5.5))
    plt.subplots_adjust(hspace=0, wspace=0, top = 0.90, bottom = 0, right = 1, left = 0)
    fig.suptitle(title_, fontsize=15, color='w')
    axes = axes.flatten()  
    
    slices = np.linspace(10, a.shape[-1]-10, num=len(axes), dtype=int)
    
    for axis, z in zip(axes,slices):
        # flip
        axis.imshow(np.flip(sp.ndimage.rotate(a[:,:,z],90), axis=1), cmap='gray')
        d = filters.sobel(np.flip(sp.ndimage.rotate(b[:,:,z],90), axis=1))
        axis.imshow(d, alpha=0.7)       
        
        axis.set_axis_off()

    fig.text(-0.02, 0.5, 'LEFT', va='center', rotation='vertical', color='w')
    plt.savefig(title_save, dpi=150, facecolor='k', bbox_inches = 'tight', pad_inches = 0)
              
        
    ### ==> lesion map over T1
    title_ = df.name+'_c-LesT1-indivspace'
    title_save = path_save+title_
    
    a = nib.load(df['t1_brain']).get_fdata()
    b = nib.load(df['les_t1']).get_fdata()
    
    # remove nan
    b[np.isnan(b)] = 0
    
    c = b.copy()
    c[c>=thresh_lesion] = 1
    c[c<thresh_lesion] = 0
    
    my_cmap = plt.cm.spring
    my_cmap.set_under(color='white', alpha='0')
    
    fig, axes = plt.subplots(nrows=4, ncols=6, sharex='none', sharey='none', figsize=(8,5.5))
    plt.subplots_adjust(hspace=0, wspace=0, top = 0.90, bottom = 0, right = 1, left = 0)
    fig.suptitle(title_, fontsize=15, color='w')
    axes = axes.flatten()
    slices = np.linspace(10, a.shape[-1]-10, num=len(axes), dtype=int)
    
    for axis, z in zip(axes,slices):
        # flip
        axis.imshow(np.flip(sp.ndimage.rotate(a[:,:,z],90),axis=1) ,cmap='gray')
        
        if np.unique(c[:,:,z])[-1]>0:
            # flip
            axis.imshow(np.flip(sp.ndimage.rotate(c[:,:,z],90), axis=1), cmap=my_cmap, vmin=0.5)
    
        axis.set_axis_off()
    
    fig.text(-0.02, 0.5, 'LEFT', va='center', rotation='vertical', color='w')
    plt.savefig(title_save, dpi=150, facecolor='k', bbox_inches = 'tight', pad_inches = 0)
    
   
    #______________
    # B) MNI-space \______________________________________________________________
       
    ### ==> Lesion map over T1_mni
    title_ = df.name+'_d-LesT1-mnispace'  
    title_save = path_save+title_
    
    a = nib.load(df['t1_reg']).get_fdata()
    b = nib.load(df['les_mni']).get_fdata()
    # remove nan
    b[np.isnan(b)] = 0
    
    c = b.copy()
    c[c>=thresh_lesion] = 1
    c[c<thresh_lesion] = 0
     
    my_cmap = plt.cm.spring
    my_cmap.set_under(color='white', alpha='0')
    
    fig, axes = plt.subplots(nrows=4, ncols=6, sharex='none', sharey='none', figsize=(8,5.5))
    plt.subplots_adjust(hspace=0, wspace=0, top = 0.90, bottom = 0, right = 1, left = 0)
    fig.suptitle(title_, fontsize=15, color='w')
    axes = axes.flatten()
    
    slices = np.linspace(20, 150, num=len(axes), dtype=int)
    
    for axis, z in zip(axes,slices):

        axis.imshow(np.flip(sp.ndimage.rotate(a[:,:,z],90),axis=1) ,cmap='gray')
        
        if np.unique(c[:,:,z])[-1]>0:
            axis.imshow(np.flip(sp.ndimage.rotate(c[:,:,z],90), axis=1), cmap=my_cmap, vmin=0.5)
      
        
        axis.set_axis_off()
    
    
    fig.text(-0.02, 0.5, 'LEFT', va='center', rotation='vertical', color='w')
    plt.savefig(title_save, dpi=150, facecolor='k', bbox_inches = 'tight', pad_inches = 0)
    
    
    
    ### ==> Contours of T1 over MNI 
    title_ = df.name+'_e-t1-mni-reg'
    title_save = path_save+title_
    
    a = nib.load(mni_standard).get_fdata()
    b = nib.load(df['t1_reg']).get_fdata()
    
    fig, axes = plt.subplots(nrows=4, ncols=6, sharex='none', sharey='none', figsize=(8,5.5))
    plt.subplots_adjust(hspace=0, wspace=0, top = 0.90, bottom = 0, right = 1, left = 0)
    fig.suptitle(title_, fontsize=15, color='w')
    axes = axes.flatten()  
    
    slices = np.linspace(20, 150, num=len(axes), dtype=int)
    
    for axis, z in zip(axes,slices):
        
        # flip
        axis.imshow(np.flip(sp.ndimage.rotate(a[:,:,z],90), axis=1), cmap='gray')
        d = feature.canny(np.flip(sp.ndimage.rotate(b[:,:,z],90), axis=1), sigma=4)
        axis.imshow(d, alpha=0.5, cmap='summer')
        
        axis.set_axis_off()

    fig.text(-0.02, 0.5, 'LEFT', va='center', rotation='vertical', color='w')
    plt.savefig(title_save, dpi=150, facecolor='k', bbox_inches = 'tight', pad_inches = 0)    



print('......THE END')
