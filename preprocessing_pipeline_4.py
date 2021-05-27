#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

 | Title:
     part 4 of the preprocessing
     
 | Date:
     2021-02-20
     
 | Author(s):
     Theodor Rumetshofer

 | Description:    
     As stated in the manuscript, this part of the preprocessing is extracting 
     the WMH volume and number in T1- and MNI-space as well as the WMH burden
     on each WM tract of the JHU atlas.
     Additionally stat. analysis was done between the sex (male vs female) and 
     FLAIR images (2D vs 3D)

 | List of functions:
     get_lesion_load() - extract WMH volume and number on T1- and MNI-space
     les_masking_jhu() - extract WMH burden on the JHU WM tracts
     lesion_frequency_maps() - make a plot of the average lesion maps in MNI-space

 | List of "non standard" modules:
     None

 | Procedure:
     1) load meta data for both cohorts
     2) extract the WMH volume and number in T1-space
     3) extract the WMH volume and number in MNI-space
     4) extract the WMH burden on the 20 JHU WM tracts
     5) test differences between sex and FLAIR
     6) plot a mean image of the clinical subgroups

 | Usage:
     ./preprocessing_pipeline_4.py

"""

from ast import literal_eval
import pandas as pd
import numpy as np
import nibabel as nib
from glob import glob as gl
from scipy.ndimage.measurements import label
import os
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import seaborn as sns
import scipy as sp
import matplotlib.cm as mplcm


#############################################################################


## lesion load in t1 and mni-space
def get_lesion_load(df_les_map, lesion_thresh):

    df_lesion_info = pd.DataFrame()
    
    for kk in range(df_les_map.shape[0]):
        
        # read data
        in_file = df_les_map.iloc[kk][0]
        les_data = nib.load(in_file).get_fdata()
    
        # remove nan
        les_data[np.isnan(les_data)] = 0
    
        # thresh the map
        les_data_thresh = les_data.copy()
        les_data_thresh[les_data_thresh<lesion_thresh] = 0
        
        # calc the resolution
        res = nib.load(in_file).header['pixdim'][1:4].prod()
        
        
        ## CALC THE LESION LOAD 
        # calc the lesion number
#        s = generate_binary_structure(3,2) # get the number of lesions (connected components)
        l, lesion_no = label(les_data_thresh,structure=None)
        
        # get the size of the lesions
        lesion_sizes = []
        for ii in np.unique(l)[1:]:
            lesion_sizes.append(np.sum(l==ii))
        
        lesion_sizes = np.sort(lesion_sizes)
        lesion_sizes = lesion_sizes*res # ml
        lesion_sizes = [round(item/1000,4) for item in lesion_sizes] 
        
        # get overall size of the lesions
        lesion_vol = np.sum(lesion_sizes)
        
        
        ## CALC THE LESION LOAD THRESHED
        idx = np.array(lesion_sizes) > 0.015
        lesion_no_thresh = sum(idx)
        lesion_vol_thresh = np.sum(np.array(lesion_sizes)[idx])

        
        out_dict =  {'lesion_no':lesion_no, 'lesion_vol_ml':lesion_vol, 
                     'lesion_no_thresh':lesion_no_thresh, 'lesion_vol_thresh_ml':lesion_vol_thresh,
                     'lesion_sizes_ml':[lesion_sizes[::-1]]
                     }
        
        tmp = pd.DataFrame(out_dict, index = [df_les_map.iloc[kk].name])
        
        df_lesion_info = df_lesion_info.append(tmp)
    
    return df_lesion_info




# individual node - mask lesion prob brain with JHU prob atlas in mni space and extract the tract prob  
def les_masking_jhu(les_mni_list, jhu_atlas_prob, jhu_atlas_labels, lesion_thresh):

    # load the JHU label atlas
    jhu_atlas_labels = nib.load(jhu_atlas_labels).get_fdata()
    
    # load the JHU probability atlas
    jhu_atlas_prob = nib.load(jhu_atlas_prob).get_fdata()
    jhu_atlas_prob = jhu_atlas_prob.astype('float')
    jhu_atlas_prob = jhu_atlas_prob/100    
    
    df_tract = pd.DataFrame()
    
    for kk in range(les_mni_list.shape[0]):
        
        # read data
        subj = les_mni_list.iloc[kk]
        
        # load lesion map in MNI space
        les_prob = nib.load(subj.les_map_mni).get_fdata()
        
        # remove nan
        les_prob[np.isnan(les_prob)] = 0
        
        ### (1) STANDARD PROCDURE USING RAW LESION MAPS
        ## calc for the jhu labels atlas
        # thresh the lesion map
        les_prob_thresh = (les_prob>lesion_thresh)*1
        les_prob_thresh_ind = np.argwhere(les_prob_thresh.flatten())
        # create empty lists
        les_jhu_labels_vx = np.zeros(sum(np.unique(jhu_atlas_labels)>0)) # 48 rois
        
        for jj in range(1,len(les_jhu_labels_vx)+1):
            jhu_atlas_labels_ind = np.argwhere(jhu_atlas_labels.flatten()==jj)
            overlap_label = np.intersect1d(jhu_atlas_labels_ind, les_prob_thresh_ind)
            les_jhu_labels_vx[jj-1] = len(overlap_label)
        
        les_jhu_labels_vx = [round(item) for item in les_jhu_labels_vx]

        ## calculations for the jhu prob atlas
        # create empty lists
        les_jhu_prob_sum = np.zeros(jhu_atlas_prob.shape[-1]) # 20 rois    
        for ii in range(jhu_atlas_prob.shape[-1]):            
            ## calc the "prob" pure
            les_jhu_prob_sum[ii] = np.sum(les_prob*jhu_atlas_prob[:,:,:,ii])
            
        les_jhu_prob_sum = [round(item, 2) for item in les_jhu_prob_sum]
      
        
        
        ### (2) THRESHING THE LESION MAPS BEFORE MASKING
        # thresh the map
        les_prob_thresh = les_prob.copy()
        les_prob_thresh[les_prob_thresh<lesion_thresh] = 0

        # calc the number of lesions        
        l, lesion_no = label(les_prob_thresh,structure=None)
            
        # get the size of the lesions
        lesion_sizes = []
        for ii in np.unique(l)[1:]:
            lesion_sizes.append(np.sum(l==ii))
        lesion_sizes = [round(item/1000,4) for item in lesion_sizes] 
        
        # calc the threshed
        idx = np.array(lesion_sizes) < 0.015
        
        les_prob_new = les_prob_thresh.copy()
        for jj, to_small, size_les in zip(np.unique(l)[1:],idx, lesion_sizes):
            if to_small == True:
                les_prob_new[l==jj] = 0
             
            
        ## calc for the jhu labels atlas
        # thresh the lesion map
        les_prob_thresh = (les_prob_new>lesion_thresh)*1
        les_prob_thresh_ind = np.argwhere(les_prob_thresh.flatten())
        # create empty lists
        les_jhu_labels_vx_threshed = np.zeros(sum(np.unique(jhu_atlas_labels)>0)) # 48 rois
        
        for jj in range(1,len(les_jhu_labels_vx_threshed)+1):
            jhu_atlas_labels_ind = np.argwhere(jhu_atlas_labels.flatten()==jj)
            overlap_label = np.intersect1d(jhu_atlas_labels_ind, les_prob_thresh_ind)
            les_jhu_labels_vx_threshed[jj-1] = len(overlap_label)
        
        les_jhu_labels_vx_threshed = [round(item) for item in les_jhu_labels_vx_threshed]

        ## calculations for the jhu prob atlas
        # create empty lists
        les_jhu_prob_sum_threshed = np.zeros(jhu_atlas_prob.shape[-1]) # 20 rois    
        for ii in range(jhu_atlas_prob.shape[-1]):            
            ## calc the "prob" pure
            les_jhu_prob_sum_threshed[ii] = np.sum(les_prob_new*jhu_atlas_prob[:,:,:,ii])
            
        les_jhu_prob_sum_threshed = [round(item, 2) for item in les_jhu_prob_sum_threshed]
        
        
        ## calc the overlapping vx of the threshed lesion map and the jhu prob atlas (0.25)
        # thresh the lesion map
        les_prob_thresh = (les_prob>lesion_thresh)*1
        les_prob_thresh_ind = np.argwhere(les_prob_thresh.flatten())
        # thresh the atlas 
        jhu_atlas_prob_thresh = (jhu_atlas_prob>0.25)*1
        # create empty lists
        les_jhu_prob_vx_threshed = np.zeros(jhu_atlas_prob_thresh.shape[-1]) # 20 rois 
        
        for jj in range(len(les_jhu_prob_vx_threshed)):
            jhu_atlas_prob_thresh_ind = np.argwhere(jhu_atlas_prob_thresh[:,:,:,jj].flatten())
            overlap_label = np.intersect1d(jhu_atlas_prob_thresh_ind, les_prob_thresh_ind)
            les_jhu_prob_vx_threshed[jj] = len(overlap_label)  
        
        
    
        ### (4) SAVE TO DF
        out_dict = {'subject_id':[subj.name],
                    'les_jhu_prob_sum':[les_jhu_prob_sum],
                    'les_jhu_prob_thresh_sum':[les_jhu_prob_sum_threshed],
                    'les_jhu_prob_thresh_vx':[les_jhu_prob_vx_threshed],
                    'les_jhu_labels_vx':[les_jhu_labels_vx],
                    'les_jhu_labels_thresh_vx':[les_jhu_labels_vx_threshed]
                    }     
   
        df_out_dict = pd.DataFrame(out_dict)
    
        df_tract = pd.concat([df_tract, df_out_dict]) 
    
    df_tract.index = df_tract.subject_id.values
    df_tract = df_tract.drop('subject_id', axis=1)
    
    return df_tract     
 


## lesion frequency map 
def lesion_frequency_maps(df, path_save, les_type, les_map, thresh_ples):
    
    # thresh the lesion maps before and sum it up
    
    test = []
    for kk in range(0,df.shape[0]):
        
        df_les = df.iloc[kk]
        affine = nib.load(df_les[les_map]).affine
        
        les = nib.load(df_les[les_map]).get_fdata()
        
        test.append(les)
    
    test = np.stack(test)        
    
    test_bin = test>=thresh_ples
    test1 = test_bin*1
    
    
    ALL = test1.sum(axis=0)
    
    hc_ind = (df.label=="HC").values
    hc = test1[hc_ind,:,:,:].sum(axis=0)
    
    sle_ind = (df.label!="HC").values
    sle = test1[sle_ind,:,:,:].sum(axis=0)
    
    nonnpsle_ind = (df.label=="nonNPSLE").values
    nonnpsle = test1[nonnpsle_ind,:,:,:].sum(axis=0)
    
    npsle_ind = (df.label=="NPSLE").values
    npsle = test1[npsle_ind,:,:,:].sum(axis=0)
    
    npsle_inflammatory_ind = (df.label_phenotype=='inflammatory').values
    npsle_inflammatory = test1[npsle_inflammatory_ind,:,:,:].sum(axis=0)
    
    npsle_ischemic_ind = (df.label_phenotype=='ischemic').values
    npsle_ischemic = test1[npsle_ischemic_ind,:,:,:].sum(axis=0)
    
    
    ALL_nifti = nib.Nifti1Image(ALL, affine=affine)
    hc_nifti = nib.Nifti1Image(hc, affine=affine)
    sle_nifti = nib.Nifti1Image(sle, affine=affine)
    nonnpsle_nifti = nib.Nifti1Image(nonnpsle, affine=affine)
    npsle_nifti = nib.Nifti1Image(npsle, affine=affine)
    npsle_inflammatory_nifti = nib.Nifti1Image(npsle_inflammatory, affine=affine)
    npsle_ischemic_nifti = nib.Nifti1Image(npsle_ischemic, affine=affine)
    
    nib.save(ALL_nifti,path_save+les_type+'all_subj'+str(test.shape[0])+'.nii.gz')
    nib.save(hc_nifti, path_save+les_type+'hc_subj'+str(np.sum(hc_ind))+'.nii.gz')
    nib.save(sle_nifti, path_save+les_type+'sle_subj'+str(np.sum(sle_ind))+'.nii.gz')
    nib.save(nonnpsle_nifti, path_save+les_type+'nonnpsle_subj'+str(np.sum(nonnpsle_ind))+'.nii.gz')
    nib.save(npsle_nifti, path_save+les_type+'npsle_subj'+str(np.sum(npsle_ind))+'.nii.gz')
    nib.save(npsle_inflammatory_nifti, path_save+les_type+'npsle_inflammatory_subj'+str(np.sum(npsle_inflammatory_ind))+'.nii.gz')
    nib.save(npsle_ischemic_nifti, path_save+les_type+'npsle_ischemic_subj'+str(np.sum(npsle_ischemic_ind))+'.nii.gz')
    
    
    ## plot the lesion map    
    mni = PATH+'SLE/MNI_template/MNI152_T1_1mm_brain.nii.gz'
    mni = nib.load(mni).get_fdata()
    
    coord = 79
    
    min_val = 0
    

    # with all 6 labels
    max_val= 50
#    max_val = max(hc.max(), sle.max(), nonnpsle.max(), npsle.max(), npsle_inflammatory.max(), npsle_ischemic.max())
    data_stream = np.stack([hc, sle, nonnpsle, npsle, npsle_inflammatory, npsle_ischemic])
    title_stream = ('HC ('+str(np.sum(hc_ind))+')','SLE ('+str(np.sum(sle_ind))+')',
                    'nonNPSLE ('+str(np.sum(nonnpsle_ind))+')','NPSLE ('+str(np.sum(npsle_ind))+')',
                    'inflammatory('+str(np.sum(npsle_inflammatory_ind))+')',
                    'ischemic('+str(np.sum(npsle_ischemic_ind))+')')
    
    fig, axes = plt.subplots(nrows=int(data_stream.shape[0]/2), ncols=2, figsize=(5,9), sharex=True, sharey=True, edgecolor='k', facecolor='k')
    plt.subplots_adjust(left=0.01, top=0.85, bottom=0.0, right=0.99)
    
    for kk in range(np.shape(data_stream)[0]):
        
        d = data_stream[kk,:,:,:]
        d = d[:,:,coord]
        d = np.flip(np.rot90(d), axis=1)
        d = np.ma.masked_where(d == 0, d)    
    
        ax = axes.flatten()[kk]
        ax.imshow(np.flip(sp.ndimage.rotate(mni[:,:,coord],90),axis=1), cmap='gray')
        im = ax.imshow(d, cmap=mplcm.jet, vmin=min_val, vmax=max_val)
        ax.set_title(title_stream[kk], fontsize=15, pad=10, color='w')
        ax.axis("off")
    
    # text left
    fig.text(-0.08, 0.45, 'LEFT', va='center', rotation='vertical', color='w', fontsize=10)
    # text z-coord
    fig.text(0.35, 0.01, 'z-coord='+str(coord), va='center', rotation=None, color='w', fontsize=10)
    
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.25, 0.025, 0.35])
    cb = fig.colorbar(im, cax=cbar_ax,  format='%i')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    cbar_ax.set_ylabel('lesion frequency', size=10, color='white')
    
    plt.savefig(path_save+'fig_'+les_type+'lesfreqmaps_labels.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches = 'tight')
    plt.show()       


#############################################################################

## PATH
PATH = '/Users/theo/_analysis/' # mac
img_path = PATH+'SLE/Leiden/'

data_path = img_path+'data/'

#!! define the lesion threshold to count them for MNI-space (should be same as in the pipeline)
lesion_thresh_lga = 0.5 # LGA

# LGA parameter
kappa = 0.3

jhu_atlas_prob = PATH+'SLE/JHU_atlas/JHU-ICBM-tracts-prob-1mm.nii.gz'
jhu_atlas_labels = PATH+'SLE/JHU_atlas/JHU-ICBM-labels-1mm.nii.gz'


folder_save = 'sle_leiden_preproc'
path_save = PATH+'coding/'+folder_save+'/' 
if not os.path.exists(path_save):
    new_folder = os.makedirs(path_save)   

print('...create project folder')



#________________
# load meta data \_____________________________________________________________

print('...load meta data')

### META DATA
df_meta = pd.read_excel(img_path+'leiden_meta_modified.xlsx', index_col=0)
df_meta.index = [str(item).zfill(3) for item in df_meta.index.values]
df_meta = df_meta.drop(['_SLE_since', '_sledai2k_score', '_sdi_score', '_visdat'], axis=1)
df_meta = df_meta.sort_index()  

### CLINICAL DATA
df_clinic = pd.read_excel(img_path+'leiden_clinical_modified_200713.xlsx', index_col=0)
df_clinic.index = [str(item).zfill(3) for item in df_clinic.index.values]
df_clinic = df_clinic.drop(['_pat_age', '_sex', '_label', '_label_phenotype', '_label_all'], axis=1)
df_clinic = df_clinic.sort_index()

### SAVE THE META DATA
df_meta_all = pd.concat([df_meta, df_clinic], sort=False, axis=1)



#_________
# LST-LGA \____________________________________________________________________

print('...load LST-LGA lesio data in T1- and MNI-space')

### Lesion load in T1-space  
les_lga_t1 = data_path+'*/ples_lga_*_masked.nii.gz'
les_lga_t1_list = gl(les_lga_t1)    
les_lga_t1_list = np.sort(les_lga_t1_list)    
subject_list = [item.split('/ples_lga')[0].split('/')[-1] for item in les_lga_t1_list]

# create dataframe
df_LGA_lesion_t1_list = pd.DataFrame(index=subject_list, data=les_lga_t1_list, columns=['LGA_les_map_t1'])    

# lesion load
df_LGA_lesion_t1 = get_lesion_load(df_LGA_lesion_t1_list, lesion_thresh_lga)

# rename
df_LGA_lesion_t1 = df_LGA_lesion_t1.rename(columns={'lesion_no':'wmh_LGA_no_t1',
                                                    'lesion_vol_ml':'wmh_LGA_vol_ml_t1',
                                                    'lesion_no_thresh':'wmh_LGA_no_thresh_t1',
                                                    'lesion_vol_thresh_ml':'wmh_LGA_vol_thresh_ml_t1',
                                                    'lesion_sizes_ml':'wmh_LGA_lesions_vol_t1'
                                                    })


### Lesion load in MNI-space
les_lga_mni = data_path+'*/ples_lga_*_masked_trans.nii.gz'
les_lga_mni_list = gl(les_lga_mni)    
les_lga_mni_list = np.sort(les_lga_mni_list)    
subject_list = [item.split('/ples_lga')[0].split('/')[-1] for item in les_lga_mni_list]

# create dataframe
df_LGA_lesion_mni_list = pd.DataFrame(index=subject_list, data=les_lga_mni_list, columns=['LGA_les_map_mni'])

# load the number of lesions in mni-space
df_LGA_lesion_mni = get_lesion_load(df_LGA_lesion_mni_list, lesion_thresh_lga)

# rename
df_LGA_lesion_mni = df_LGA_lesion_mni.rename(columns={'lesion_no':'wmh_LGA_no_mni',
                                                    'lesion_vol_ml':'wmh_LGA_vol_ml_mni',
                                                    'lesion_no_thresh':'wmh_LGA_no_thresh_mni',
                                                    'lesion_vol_thresh_ml':'wmh_LGA_vol_thresh_ml_mni',
                                                    'lesion_sizes_ml':'wmh_LGA_lesions_vol_mni'
                                                    })

### Lesion tract load in MNI-space
print('...mask the lesion maps with the JHU atlas tracts')

df_les_mni_list = pd.DataFrame(index=subject_list, data=les_lga_mni_list, columns=['les_map_mni'])

# get lesion tract masking
df_lesion_tract = les_masking_jhu(df_les_mni_list, jhu_atlas_prob, jhu_atlas_labels, lesion_thresh_lga)
df_lesion_tract = df_lesion_tract[['les_jhu_prob_sum', 'les_jhu_prob_thresh_sum','les_jhu_prob_thresh_vx',
                                   'les_jhu_labels_vx', 'les_jhu_labels_thresh_vx']]

# load the JHU tract names and save the values of the vlaues in a new dataframe
names_labels = pd.read_csv(PATH+'SLE/JHU_atlas/JHU_labels.csv', index_col=0)
names_tracts = pd.read_csv(PATH+'SLE/JHU_atlas/JHU_tracts.csv', index_col=0)
df_LGA_tract = pd.DataFrame()
for col in df_lesion_tract:    
    temp = df_lesion_tract[col].apply(pd.Series)    
    if 'les_jhu_labels_vx' in col:
        temp.columns = ['LGA_lblVx_'+item for item in names_labels.jhu_labels.values]            
    elif 'les_jhu_labels_thresh_vx' in col:
        temp.columns = ['LGA_lblVxThresh_'+item for item in names_labels.jhu_labels.values]        
    elif 'les_jhu_prob_sum' in col:
        temp.columns = ['LGA_probSum_'+item for item in names_tracts.jhu_tracts.values] 
    elif 'les_jhu_prob_thresh_sum' in col:
        temp.columns = ['LGA_probSumThresh_'+item for item in names_tracts.jhu_tracts.values]         
    elif 'les_jhu_prob_thresh_vx' in col:
        temp.columns = ['LGA_probVxThresh_'+item for item in names_tracts.jhu_tracts.values]         
    df_LGA_tract = pd.concat([df_LGA_tract,temp],axis=1)  





#______________
# put together \______________________________________________________________

print('...put data together and save as csv-files')

# all
df_all = pd.concat([df_meta_all,
                df_LGA_lesion_t1, df_LGA_lesion_mni, df_LGA_tract, df_LGA_lesion_mni_list, df_LGA_lesion_t1_list
                ], axis=1)    

# set new numbering
df_all.insert(0, value=range(0,df_all.shape[0]), column='#')
df_all = df_all.drop('no', axis=1)
# save
df_all.to_csv(path_save+'leiden_wmh_pp.csv')     


# remove excluded and outliers
df = df_all.copy()
df = df.query('excluded_LGA=="0"')
df = df.query('outliers_lesion=="0"') # no in the leiden cohort





##############################################################################
# ANALYSIS


#__________________
# FLAIR: 3D vs 2D  \___________________________________________________________

# nonNPSLE only
# NOT T1-vs-MNI
# TAKING ONLY THE THRESHED DATA

# LGA
vol_var = 'wmh_LGA_vol_thresh_ml_t1'
num_var = 'wmh_LGA_no_thresh_t1'

df.groupby(['FLAIR','label']).count()

df_plot = df.query('label=="nonNPSLE"')

## statistics on LPA only
vol2D = df_plot[df_plot.FLAIR=="2D"][vol_var]
vol3D = df_plot[df_plot.FLAIR=="3D"][vol_var]
num2D = df_plot[df_plot.FLAIR=="2D"][num_var]
num3D = df_plot[df_plot.FLAIR=="3D"][num_var]

## take only those that are not zero!
#vol2D = vol2D[vol2D != 0]
#vol3D = vol3D[vol3D != 0]
#num2D = num2D[num2D != 0]
#num3D = num3D[num3D != 0]

vol_all = pd.concat([vol2D,vol3D])
num_all = pd.concat([num2D,num3D])

### RAW DATA
print('RAW DATA:')
# test for normality
ndtest_vol = sp.stats.normaltest(vol_all)
ndtest_num = sp.stats.normaltest(num_all)

if ndtest_vol[1] < 0.05:
    print('H0 rejected. volume is NOT normal distributed')
else: print ('H0 not rejected. volume is normal distributed')
if ndtest_num[1] < 0.05:
    print('H0 rejected. number of lesions are NOT normal distributed')
else: print ('H0 not rejected. number of lesions are normal distributed')


kw_vol = sp.stats.kruskal(vol2D,vol3D)
kw_num = sp.stats.kruskal(num2D,num3D)

print('Kruskal-Wallis test p-values:')
print(' | Volume = ',kw_vol[1])
print(' | Number of lesions = ',kw_num[1])
print()

### LOG DATA to use t-statistics
print('LOG DATA:')
# test for normality
ndtest_vol = sp.stats.normaltest(np.log(vol_all))
ndtest_num = sp.stats.normaltest(np.log(num_all))

if ndtest_vol[1] < 0.05:
    print('H0 rejected. volume is NOT normal distributed')
else: print ('H0 not rejected. volume is normal distributed')
if ndtest_num[1] < 0.05:
    print('H0 rejected. number of lesions are NOT norma l distributed')
else: print ('H0 not rejected. number of lesions are normal distributed')


kw_vol = sp.stats.ttest_ind(vol2D,vol3D)
kw_num = sp.stats.ttest_ind(num2D,num3D)

print('t-test p-values:')
print(' | Volume = ',kw_vol[1])
print(' | Number of lesions = ',kw_num[1])




#_________________
# MALE vs FEMALE  \____________________________________________________________

# TAKING ONLY THE THRESHED DATA and T1
df.groupby(['sex','label']).count()

# LGA
#vol_var = 'wmh_LGA_vol_thresh_ml_t1'
#vol_var = 'rwmh_LGA_vol_thresh_ml_t1' # relative wmh to icv
#num_var = 'wmh_LGA_no_thresh_t1'

## statistics on 
volM = df[df.sex=="male"][vol_var]
volF = df[df.sex=="female"][vol_var]
numM = df[df.sex=="male"][num_var]
numF = df[df.sex=="female"][num_var]

# take only those that are not zero!
volM = volM[volM != 0]
volF = volF[volF != 0]
numM = numM[numM != 0]
numF = numF[numF != 0]

vol_all = pd.concat([volM,volF])
num_all = pd.concat([numM,numF])

### RAW DATA
print('RAW DATA:')
# test for normality
ndtest_vol = sp.stats.normaltest(vol_all)
ndtest_num = sp.stats.normaltest(num_all)

if ndtest_vol[1] < 0.05:
    print('H0 rejected. volume is NOT normal distributed')
else: print ('H0 not rejected. volume is normal distributed')
if ndtest_num[1] < 0.05:
    print('H0 rejected. number of lesions are NOT normal distributed')
else: print ('H0 not rejected. number of lesions are normal distributed')


kw_vol = sp.stats.kruskal(volM,volF)
kw_num = sp.stats.kruskal(numM,numF)

print('Kruskal-Wallis test p-values:')
print(' | Volume = ',kw_vol[1])
print(' | Number of lesions = ',kw_num[1])
print()


### LOG DATA to use t-statistics
print('LOG DATA:')
# test for normality
ndtest_vol = sp.stats.normaltest(np.log(vol_all))
ndtest_num = sp.stats.normaltest(np.log(num_all))

if ndtest_vol[1] < 0.05:
    print('H0 rejected. volume is NOT normal distributed')
else: print ('H0 not rejected. volume is normal distributed')
if ndtest_num[1] < 0.05:
    print('H0 rejected. number of lesions are NOT normal distributed')
else: print ('H0 not rejected. number of lesions are normal distributed')


kw_vol = sp.stats.ttest_ind(np.log(volM),np.log(volF))
kw_num = sp.stats.ttest_ind(np.log(numM),np.log(numF))

print('t-test p-values:')
print(' | Volume = ',kw_vol[1])
print(' | Number of lesions = ',kw_num[1])
print()


sns.barplot(data=df, y=vol_var, x='sex', ci=95, estimator=np.median)
plt.show()

sns.barplot(data=df, y=num_var, x='sex', ci=95, estimator=np.median)
plt.show()



#________________
# frequency maps \_____________________________________________________________

# create the lesion frequency map of the clinical labels and plot them

# LGA
thresh_ples = lesion_thresh_lga
les_type = 'LGA_'
les_map = les_type + 'les_map_mni'
lesion_frequency_maps(df, path_save, les_type, les_map, thresh_ples)





print('>>>>>>>>>>>>>> The END')
