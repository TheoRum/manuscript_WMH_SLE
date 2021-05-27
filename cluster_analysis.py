#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

 | Title:
     cluster analysis on the WMH burden on the different JHU WM tracts
     
 | Date:
     2021-03-03
     
 | Author(s):
     Theodor Rumetshofer

 | Description:    
     Performs cluster analysis on the WMH burden on the different JHU WM tracts
     extracted by the preprocessing pipelines part 3. Additionally, the clusters
     are evaluated and statistical analysis is performed on the extracted clusters

 | List of functions:
     plot_heatmap() - plot the heatmap from the cluster analysis
     lesion_frequency_maps() - make a plot of the average lesion maps in MNI-space
     get_lesion_load_range() - load lesion from T1- and MNI-space

 | List of "non standard" modules:
     None

 | Procedure:
     1) load the meta data as well as the lesion burden on the JHU WM tracts
        extracted from preprocessing_pipeline_3.py
     2) l2-normalization
     3) cluster analysis
     4) cluster evaluation performance
     5) statistic of the meta data between the clusters
     6) save the values in a table

 | Usage:
     ./cluster_analysis.py

"""

from __future__ import (print_function, division, unicode_literals, absolute_import)
import time
import os
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy as sch
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score
from sklearn.preprocessing import normalize
from scipy.ndimage.measurements import label
from glob import glob as gl
import matplotlib.cm as mplcm
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.formula.api import ols


#############################################################################

def plot_heatmap(X_plot, color_lbl, cluster_labels, title, v_max=None):
    
    cm=sns.clustermap(X_plot.T, method=None, metric=None, cmap=None
                  , row_cluster=False
                  , col_cluster=False, col_colors = color_lbl
                  , yticklabels=1, xticklabels=0
                  , standard_scale=None, z_score=None, robust=True
                  , figsize=(X_plot.T.shape[1]/5,X_plot.T.shape[0]/2)
                  )
    
    ax = cm.ax_heatmap
    ax.set_ylabel('JHU WM tracts', fontsize=40)
    ax.set_xlabel('subjects', fontsize=40)
    ax.set_title(title, y=1.3, fontsize=30)
    
    # set labelsize smaller
    cm_ax = plt.gcf().axes[-2]
    cm_ax.tick_params(axis='y', labelsize=25)

    for tick in cm.ax_col_colors.get_yticklabels():
        tick.set_fontsize(25)
    
    # plot lines for the different clusters
    col_per_cluster = [list(cluster_labels).count(item) for item in np.unique(cluster_labels)]
    ver_lines = list(np.cumsum(col_per_cluster))
    ver_lines.insert(0,0)
    for pp in range(1,len(ver_lines)-1):
        cm.ax_heatmap.vlines(ver_lines[pp],0,X_plot.T.shape[1], colors='y')  
        cm.ax_col_colors.vlines(ver_lines[pp],0,X_plot.T.shape[1], colors='y')  
    
    # size
    cm.cax.set_visible(False)
    cm.ax_heatmap.set_position([.1, .1, 0.9, .8])
    if (X_plot.T.shape[0]) < 15:
        size_col_color = 0.4
    else: size_col_color = 0.2
    cm.ax_col_colors.set_position([.1 ,0.91, 0.9, size_col_color],)
    plt.rcParams["axes.grid"] = False
    
    
    plt.close()
    cm.savefig(path_save_new+title+'.png',dpi=200, bbox_inches='tight')
  


## lesion frequency map 
def lesion_frequency_maps(df, path_save_new, les_map, les_thresh):

    # label
    labelby_ = 'cl_'+str(best_cluster)
    # drop 
    df_lfm = df[[les_map,labelby_]].dropna(axis=0)  
    # name for saving
    # thresh the lesion maps before with thresh and sum it up
    test = []
    for kk in range(0,df_lfm.shape[0]):
    
        df_les = df_lfm.iloc[kk]
        affine = nib.load(df_les[les_map]).affine
        les = nib.load(df_les[les_map]).get_fdata()
        test.append(les)
    
    test = np.stack(test)        
    test_bin = test>=les_thresh
    test1 = test_bin*1
    
    data_stream = []
    max_values = []
    title_stream = []
    
    for cl in np.unique(df_lfm[labelby_].values):
    
        ind = (df_lfm[labelby_]==cl).values
    
        d = test1[ind,:,:,:].sum(axis=0) # total occurence
        d = 1/sum(ind) * d
        d_nifti = nib.Nifti1Image(d, affine=affine)
        nib.save(d_nifti, path_save_new+'cl'+str(int(cl))+'_subj'+str(np.sum(ind))+'.nii.gz')
    
        data_stream.append(d)
        max_values.append(d.max())
        title_stream.append('Cluster '+str(int(cl)))
    
    
    data_stream = np.stack(data_stream)
    data_stream = data_stream[0:6,:,:,:]
#    max_val = max(max_values)
    max_val = 0.5
#    title_stream[0] = ti1tle_stream[0].replace('0','Healthy controls')
    title_stream[0] = 'Healthy controls'
    title_stream[-1] = 'SLE (no WMH)'
    ## plot the lesion map
    coord = 76
    min_val = 0
    
    mni = img_path+'MNI_template/MNI152_T1_1mm_brain.nii.gz'
    mni = nib.load(mni).get_fdata()
    
    t = nib.load('/Users/theo/_analysis/SLE/JHU_atlas/JHU-ICBM-tracts-prob-1mm.nii.gz')
    t = t.get_data()
    t_atrl = t[:,:,coord,0]
    t_atrr = t[:,:,coord,1]
    t_fmj = t[:,:,coord,8]
    t_fmn = t[:,:,coord,9]
    t_atrl = np.flip(np.rot90(t_atrl), axis=1)
    t_atrl = np.ma.masked_where(t_atrl == 0, t_atrl)
    t_atrr = np.flip(np.rot90(t_atrr), axis=1)
    t_atrr = np.ma.masked_where(t_atrr == 0, t_atrr)
    t_fmj = np.flip(np.rot90(t_fmj), axis=1)
    t_fmj = np.ma.masked_where(t_fmj == 0, t_fmj)
    t_fmn = np.flip(np.rot90(t_fmn), axis=1)
    t_fmn = np.ma.masked_where(t_fmn == 0, t_fmn)
    
    cmap_ = mplcm.jet
    fig, axes = plt.subplots(nrows=1, ncols=int(data_stream.shape[0]), figsize=(14,3), sharex=True, sharey=True, edgecolor='k', facecolor='k')
    plt.subplots_adjust(left=0.01, top=0.85, bottom=0.0, right=0.99)
    
    for kk in range(np.shape(data_stream)[0]):
    
        d = data_stream[kk,:,:,:]
        d = d[:,:,coord]
        d = np.flip(np.rot90(d), axis=1)
        d = np.ma.masked_where(d == 0, d)    
    
        ax = axes.flatten()[kk]
        ax.imshow(np.flip(sp.ndimage.rotate(mni[:,:,coord],90),axis=1), cmap='gray', alpha=1)
        
        if kk == 1:
            im = ax.imshow(t_fmj,cmap='copper', alpha=0.6)
        elif kk == 2:
            im = ax.imshow(t_atrr,cmap='copper', alpha=0.6)            
        elif kk == 3:
            im = ax.imshow(t_fmn,cmap='copper', alpha=0.6) 
        elif kk == 4:
            im = ax.imshow(t_atrl,cmap='copper', alpha=0.6) 
            
        im = ax.imshow(d, cmap=cmap_, vmin=min_val, vmax=max_val)
        ax.set_title(title_stream[kk], fontsize=15, pad=10, color='w')
        ax.axis("off")
    
    # text left
    fig.text(-0.01, 0.45, 'left', va='center', rotation='vertical', color='w', fontsize=12)
    # text z-coord
    fig.text(0.45, -0.01, 'z-coord='+str(coord), va='center', rotation=None, color='w', fontsize=12)
    
        
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.6])
    cb = fig.colorbar(im, cax=cbar_ax,  format='%.1f')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    cbar_ax.set_ylabel('lesion probability', size=12, color='white')
    
    plt.savefig(path_save_new+'fig_lesfreqmaps_'+labelby_+'.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches = 'tight')
    plt.show()



## lesion load in t1 and mni-space
def get_lesion_load_range(df_les_map):

    df_lesion_info = pd.DataFrame()
    
    for kk in range(df_les_map.shape[0]):
        
        # read data
        subj_name = df_les_map.iloc[kk].name
        in_file = df_les_map.iloc[kk][0]
        les_data = nib.load(in_file).get_fdata()
    
        # remove nan
        les_data[np.isnan(les_data)] = 0
    
        # calc the resolution
        res = nib.load(in_file).header['pixdim'][1:4].prod()
    
        for lesion_thresh in np.arange(0.1,1,0.1):
            
            # thresh the map
            les_data_thresh = les_data.copy()
            
            les_data_thresh[les_data_thresh<lesion_thresh] = 0
        
        
            # CALC THE LESION LOAD 
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
#            lesion_vol = np.sum(lesion_sizes)
            
            
            ## CALC THE LESION LOAD THRESHED
            idx = np.array(lesion_sizes) > 0.015
            lesion_no_thresh = sum(idx)
            lesion_vol_thresh = np.sum(np.array(lesion_sizes)[idx])

            out_dict =  {'lesion_thresh':lesion_thresh
                         , 'lesion_no_thresh':lesion_no_thresh
                         , 'lesion_vol_thresh_ml':lesion_vol_thresh
                         }
            
#            tmp = pd.DataFrame(out_dict, index = [df_les_map.keys()[kk]])
            tmp = pd.DataFrame(out_dict, index = [subj_name])
        
            df_lesion_info = df_lesion_info.append(tmp)
    
    return df_lesion_info


    
#############################################################################


## PATH
PATH = '/Users/data/_analysis/' # mac
img_path = PATH+'SLE/'

leiden_data = img_path+'/sle_LEIDEN_WMH_preproc/'
lund_data = img_path+'/sle_3T_WMH_preproc/'


lesion_thresh_lga = 0.5 # LGA
lesion_thresh_lpa = 0.2 # LPA
lesion_thresh_salem = 0.5 # SALEM
lesion_thresh_manual = 0.5 # MANUAL

jhu_atlas_prob = PATH+'SLE/JHU_atlas/JHU-ICBM-tracts-prob-1mm.nii.gz'
jhu_atlas_labels = PATH+'SLE/JHU_atlas/JHU-ICBM-labels-1mm.nii.gz'


folder_save = 'sle_WMH_clustering'
path_save = img_path+folder_save+'/' 
if not os.path.exists(path_save):
    new_folder = os.makedirs(path_save)   


#___________
# load data \__________________________________________________________________


# leiden
df_leiden = pd.read_csv(leiden_data+'leiden_wmh_pp.csv',index_col=0)
# insert study_id
df_leiden.insert(0, column='study_id', value=[str(item).zfill(3) for item in df_leiden.index.values])
# rename "label_all" to "label2"
df_leiden = df_leiden.rename(columns={'label_all':'label2'})
# rename index
df_leiden.index = ['leiden_'+str(item).zfill(3) for item in df_leiden.index.values]
# insert cohort
df_leiden.insert(0, column = 'cohort', value=[item.split('_')[0] for item in df_leiden.index.values])

# rename the LAB data to ACR from Leiden
df_leiden = df_leiden.rename(columns={'LAB_anti_nuclantibod':'ACR_ana_level',
                                        'LAB_anti_dsdna':'ACR_anti_dsdna'
                                        ,'LAB_anti_sm':'ACR_anti_sm'})

# take one excluded (LGA or LPA), rename and remove the other
# LGA
df_leiden = df_leiden.rename(columns={'excluded_LGA':'excluded'}) 
df_leiden = df_leiden.drop('excluded_LPA', axis=1)               
        


# lund  
df_lund = pd.read_csv(lund_data+'wmh_pp.csv',index_col=0)
# insert sex column
df_lund.insert(1, column='sex', value='female')
# rename index
df_lund.index = ['lund_'+str(item).split('_')[2] for item in df_lund.index.values]
# insert cohort
df_lund.insert(0, column = 'cohort', value=[item.split('_')[0] for item in df_lund.index.values])


#______________
# put together \______________________________________________________________

df_all = pd.concat([df_lund, df_leiden], sort=False)
df_all = df_all.drop(['#'], axis=1)

age_onset = df_all.pat_age - df_all.disease_duration
#age_onset = age_onset.fillna(0).values               
df_all['age_onset'] = age_onset.values

# take only values of interest (voi)
list_voi = ['num_subjects','cohort','study_id','label','label2',
            'sex','pat_age','disease_duration','age_onset',
            'sdi_score','sledai2k_score','FLAIR',
            'excluded','outliers_lesion','comments', 'VE_WMH', 'ACRno_criteria', 
            
            # LAB/ ACR data
            'ACR_ana_level','ACR_anti_dsdna','ACR_anti_sm',
            
            'wmh_LPA_vol_thresh_ml_t1','wmh_LPA_vol_thresh_ml_mni',
            'wmh_LPA_no_thresh_t1','wmh_LPA_no_thresh_mni',
            'LPA_les_map_t1','LPA_les_map_mni',
            
            'wmh_LGA_vol_thresh_ml_t1','wmh_LGA_vol_thresh_ml_mni',
            'wmh_LGA_no_thresh_t1','wmh_LGA_no_thresh_mni',     
            'LGA_les_map_t1','LGA_les_map_mni',
            
            'wmh_SALEM_vol_thresh_ml_t1','wmh_SALEM_vol_thresh_ml_mni',
            'wmh_SALEM_no_thresh_t1','wmh_SALEM_no_thresh_mni',     
            'SALEM_les_map_t1','SALEM_les_map_mni',              

            'wmh_MANUAL_vol_thresh_ml_t1','wmh_MANUALvol_thresh_ml_mni',
            'wmh_MANUAL_no_thresh_t1','wmh_MANUAL_no_thresh_mni',     
            'MANUAL_les_map_t1','MANUAL_les_map_mni'       
            
            ,'ICV_liter', 'rwmh_LPA_vol_thresh_ml_t1','rwmh_LGA_vol_thresh_ml_t1','rwmh_SALEM_vol_thresh_ml_t1',
            
            
            'FSS_score', 'MADRS_score', 'HADS_score',
            'smoking_perday','pain_VAS','fatigue_VAS','QoL_eq5d'  
            
            ]

df_voi = df_all.filter(items=list_voi)
df_np = df_all.filter(regex=r'(NP_cns_|NP_pns_)')
df_cns = df_all.filter(regex='_ss')


LGA_probSum = df_all.filter(regex='LGA_probSum')
LGA_probVx = df_all.filter(regex='LGA_probVx')
LGA_lblVx = df_all.filter(regex='LGA_lblVx')


df = pd.concat([df_voi, df_np, df_cns,
                LGA_probSum, LGA_probVx, LGA_lblVx,
                ], sort=False, axis=1)


# remove outliers from each dataset
df = df.query('excluded=="0"')
df = df.query('outliers_lesion=="0"')

df.insert(0, column = '#', value=range(0,df.shape[0]))

df.to_csv(path_save+'wmh_clustering_multisite.csv')

# overview
df.groupby(['label','cohort']).label.count()




#____________
# CLUSTERING \________________________________________________________________


# LGA
features_used = 'LGA_probSum_'


# define the name by the clinical labels found in the data
datetimestamp = time.strftime('%Y%m%d%H%M%S')             
# define the new folder
save_path_folder = features_used+datetimestamp
path_save_new = path_save+save_path_folder+'/'
if not os.path.exists(path_save_new):
    new_folder = os.makedirs(path_save_new)     


# ALL
X_all = df.copy()
X_data = df[df.filter(regex=features_used).sum(axis=1)>0]
X_raw = X_data.filter(regex=features_used)

## remove subjects with zero lesion on the tracts
idx_notzero = X_data.filter(regex=features_used).sum(axis=1)>0

# 
cluster_range = range(1,11)
method_ = 'ward'
metric_ = 'euclidean'
X_lbl = X_all


# labels
color_lbl = pd.DataFrame(columns=['cohort LEIDEN','3D FLAIR','male sex','HC','nonNPSLE', 'NPSLE'], index=X_lbl.index, data='white')

color_lbl['cohort LEIDEN'][X_lbl[X_lbl.cohort=='leiden'].index] = 'saddlebrown'
color_lbl['3D FLAIR'][X_lbl[X_lbl.FLAIR=='3D'].index] = 'pink'
color_lbl['HC'][X_lbl[X_lbl.label=='HC'].index] = 'g'
color_lbl['nonNPSLE'][X_lbl[X_lbl.label=='nonNPSLE'].index] = 'b'
color_lbl['NPSLE'][X_lbl[X_lbl.label=='NPSLE'].index] = 'r'



# plot the data raw and sort by
new_order_row = [list(r) for r in zip(X_data.cohort, X_data.label.str.lower(), X_data.index)]
new_order_row.sort(key=lambda k: (k[0], k[1]), reverse=False)
new_order = [i[-1] for i in new_order_row]

cluster_labels = np.zeros(len(color_lbl))
plot_heatmap(X_raw.reindex(new_order), color_lbl, cluster_labels, '0_raw_data')




#### L-NORM
norm_type = 'l2'

# norm on subjects
# using demean (also negative values)
l = X_raw.values.T - np.mean(X_raw.values, axis=1)
X_norm_all = normalize(l.T, norm=norm_type, axis=1)

X_norm_all = pd.DataFrame(X_norm_all, columns=X_raw.columns, index=X_raw.index)
cluster_labels = np.zeros(len(X_norm_all))
plot_heatmap(X_norm_all.reindex(new_order), color_lbl, cluster_labels, '1_'+norm_type+'_subjects')



#### RAW
title_plot = features_used+'raw_'
X_cl = X_norm_all[idx_notzero]
X_cl = X_norm_all
X_cl = X_norm_all[X_data.label!="HC"]
X_cl_all = X_norm_all.copy()




#### CLUSTERING
title_plot = '3_'+features_used+'l2_'

cm=sns.clustermap(X_cl.T, method=method_, metric=metric_, cmap=None
              , row_cluster=False
              , col_cluster=True, col_colors = color_lbl
              , yticklabels=1, xticklabels=1
              , standard_scale=None, z_score=None, robust=True
              , figsize=(X_cl.T.shape[1]/5,X_cl.T.shape[0]/2)
              )
#plt.tight_layout()
ax = cm.ax_heatmap
ax.set_ylabel('features', fontsize=20)
ax.set_xlabel('subjects', fontsize=20)
ax.set_title(title_plot, y=1.65, fontsize=20)

# set labelsize smaller
cm_ax = plt.gcf().axes[-2]
cm_ax.tick_params(axis='x', labelsize=12)
cm_ax.tick_params(axis='y', labelsize=20)

for tick in cm.ax_col_colors.get_yticklabels():
    tick.set_fontsize(20)

# size
cm.cax.set_visible(False)
#hm = cm.ax_heatmap.get_position()
cm.ax_heatmap.set_position([.1, .1, 0.9, .8])
cm.ax_col_colors.set_position([.1, .91, 0.9, .2]) 
cm.ax_col_dendrogram.set_position([.1, 1.12, .9, .3]) # show dendrogramm
plt.rcParams["axes.grid"] = False
cm.savefig(path_save_new+title_plot+'cluster-0.png',dpi=200, bbox_inches='tight')
plt.close()


#### PLOT DENDROGRAM
# use dendrogram from clustermap
Z = cm.dendrogram_col.linkage
# calc dendrogram new OBS: is the same when no Normalization is used in the clustermap
Z = sch.linkage(X_cl, metric=metric_, method = method_)
l = sch.fcluster(Z, 5, criterion='maxclust')

fig, axis = plt.subplots(figsize=(8,3))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('subjects')
plt.ylabel('distance (Ward)')
# set colors
sch.set_link_color_palette(None) # default! see https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.set_link_color_palette.html

dendrogram(Z, labels=X_cl.index, ax=axis, color_threshold=3.8, above_threshold_color='grey', no_labels=True)
plt.axhline(y=3.8, c='red', lw=2, linestyle='dashed')
plt.savefig(path_save_new+title_plot+'cluster-0_dendrogram.png',dpi=300, bbox_inches='tight')

### CLUSTERING over range
df_cluster = pd.DataFrame(index=X_cl.index)

## Elbow criterion
sse = []
nocl_silh = []
nocl_caha = []
nocl_dabo  = []

for nc in cluster_range:  
    
    # using scikit-learn
    clustering = AgglomerativeClustering(n_clusters=nc, affinity=metric_, linkage = method_)
    clustering.fit(X_cl)
    l = clustering.labels_   
    
    df_cluster['cl_'+str(nc)] = l

    title_plot_nc = title_plot+'cluster-'+str(nc)

    # sort the subjects within a cluster by their clinical label
    X_meta_sort = X_data.loc[list(X_cl.index.values)]
    newOrder_row = [list(r) for r in zip(l, X_meta_sort.cohort, X_meta_sort.FLAIR, X_meta_sort.label2.str.lower(), X_meta_sort.index, X_meta_sort.label)]
    newOrder_row.sort(key=lambda k: (k[0], k[1], k[5], k[3]), reverse=False)    
    order_row = [i[-2] for i in newOrder_row]

    X_plot = X_cl.reindex(order_row)
    cluster_labels = l
    plot_heatmap(X_plot, color_lbl, cluster_labels, title_plot_nc)
    

    ## CALC CLUSTER EVALUATION PERFORMANCE
    # elbow 
    ss=0
    cnt=0
    for k in np.unique(l):
        ss+=np.sum((X_cl.values[l==k]-np.mean(X_cl.values[l==k],axis=0))**2)
        cnt+=1.
    sse.append(ss)

    # silhouette score (starts with 2 clusters)
    if nc == 1:
        nocl_silh.append(np.nan)
    else: nocl_silh.append(metrics.silhouette_score(X_cl, l, metric='euclidean'))
   
    # Calinski-Harabaz score (starts with 2 clusters)
    if nc == 1:
        nocl_caha.append(np.nan)
    else: nocl_caha.append(metrics.calinski_harabaz_score(X_cl, l)) 

    # Davies-Bouldin Index (starts with 2 clusters)
    if nc == 1:
        nocl_dabo.append(np.nan)
    else: nocl_dabo.append(metrics.davies_bouldin_score(X_cl, l))    
    
    
#________________________________
# cluster evaluation performance \_____________________________________________  
        
## Elbow
elbow_thresh = 0.070
x=np.arange(cluster_range.start,cluster_range.stop).tolist()
ssen=sse/sse[0]
dssen=ssen[:-1]-ssen[1:]
nocl_elbow=x[np.where(dssen<elbow_thresh)[0][0]]
print('   |==> elbow point: n = '+str(nocl_elbow))


## PLOT ALL 4 scores
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,4))
fig.suptitle('Clustering performance evaluation', fontsize=15)
plt.subplots_adjust(hspace=0.3,wspace=0.2,top=0.85)

ax = axes.flatten()

ax[0].plot(x, nocl_silh)
ax[0].set_title('Silhouette Coefficient')
ax[0].set_xlabel('number of clusters')
ax[0].set_ylabel('coefficient')
ax[0].vlines(5,0.28, 0.4, colors='r', linestyles='dashed', linewidth=2) 
#ax[0].vlines(10,0.28, 0.4, colors='r', linestyles='dashed', linewidth=2) 


ax[1].plot(x, nocl_caha)
ax[1].set_title('Calinski-Harabaz Index')
ax[1].set_xlabel('number of clusters')
ax[1].set_ylabel('index')
ax[1].vlines(5, 59, 75, colors='r', linestyles='dashed', linewidth=2) 

fig.savefig(path_save_new+title_plot+'cluster_evaluation.png',bbox_inches='tight')
plt.show()


## cALC THE BEST POINT:
possible_clusters = [nocl_elbow, nocl_silh.index(np.nanmax(nocl_silh))+1, 
                nocl_caha.index(np.nanmax(nocl_caha))+1, nocl_dabo.index(np.nanmin(nocl_dabo))+1]
best_cluster = max(set(possible_clusters), key = possible_clusters.count)
#best_cluster = 3
print(' | Best clusters from the evaluation: ', possible_clusters)
print(' | Check out the best cluster: ', best_cluster)




#_____________________________________
# save clusters in the main dataframe \________________________________________

cl_save = np.empty((X_all.shape[0], df_cluster.shape[1]))
cl_save[:] = np.nan
ind_hc = X_all.query('label=="HC"')
ind_sle_zeroMWH = X_all[X_all.filter(regex=features_used).sum(axis=1)==0].query('label!="HC"')
#cl_save[ind_hc.index] = 0
cl_save = pd.DataFrame(cl_save, columns=df_cluster.keys().values, index=X_all.index)
cl_save.set_value(index=ind_hc.index, col=df_cluster.keys().values, value=0) # set HC
cl_save.set_value(index=ind_sle_zeroMWH.index, col=df_cluster.keys().values, value=99) # set subjects wihtout lesions
cl_save.set_value(index=df_cluster.index, col=df_cluster.keys().values, value=df_cluster.values)

df2 = pd.concat([X_all, cl_save], axis=1)

df2.to_csv(path_save_new+'df2.csv')


 
#___________________
# NORMED clustering \__________________________________________________________


if 'LPA' in features_used:
    wmh_vol = 'wmh_LPA_vol_thresh_ml_t1'
    wmh_no = 'wmh_LPA_no_thresh_t1'
elif 'LGA' in features_used:
    wmh_vol = 'wmh_LGA_vol_thresh_ml_t1'
    wmh_no = 'wmh_LGA_no_thresh_t1'
#    wmh_vol_mni = 'wmh_LGA_vol_thresh_ml_mni'
elif 'SALEM' in features_used:
    wmh_vol = 'wmh_SALEM_vol_thresh_ml_t1'
    wmh_no = 'wmh_SALEM_no_thresh_t1'    
    
    
print(wmh_vol)
print(wmh_no)    
    


v_max_raw = 50
v_max_norm = 1

## define lesion sum
wmh_sumJHU = df2.filter(regex=features_used).sum(axis=1).values
wmh_sumJHU = pd.DataFrame(data=wmh_sumJHU, index=df2.index, columns=['wmh_sumJHU'])

# define data
df2 = pd.concat([df2,wmh_sumJHU], axis=1, sort=False)
X_plot_all = df2[df2['cl_'+str(best_cluster)] != 99]
X_plot_all = X_plot_all.query('wmh_sumJHU>0')
X_plot_all = X_plot_all[['cohort','label','label2','FLAIR','sex','cl_'+str(best_cluster), 'wmh_sumJHU']]



new_order_row = [list(r) for r in zip(X_plot_all.cohort, X_plot_all.label.str.lower(), X_plot_all['wmh_sumJHU'], X_plot_all.index)]
new_order_row.sort(key=lambda k: (k[0], k[1], k[2]), reverse=False)
new_order = [i[-1] for i in new_order_row]

# plot the raw data
title = title_plot+'n0_raw'
cluster_labels = np.zeros(len(X_plot_all['cl_'+str(best_cluster)]))
X_plot = X_raw.reindex(new_order)
X_plot.columns = [item.split('_',maxsplit=2)[-1].replace('_', ' ') for item in X_plot.keys()]
plot_heatmap(X_plot, color_lbl, cluster_labels, title, v_max=v_max_raw)
 
# plot the normed data
title = title_plot+'n1_norm'
cluster_labels = np.zeros(len(X_plot_all['cl_'+str(best_cluster)]))
X_plot = X_cl_all.reindex(new_order)
X_plot.columns = [item.split('_',maxsplit=2)[-1].replace('_', ' ')  for item in X_plot.keys()]
plot_heatmap(X_plot, color_lbl, cluster_labels, title, v_max=v_max_norm)

## plot the data normed clustered 
new_order_row = [list(r) for r in zip(X_plot_all['cl_'+str(best_cluster)], X_plot_all.cohort, X_plot_all.label.str.lower(), X_plot_all.label2, X_plot_all.index)]
new_order_row.sort(key=lambda k: (k[0], k[1], k[2], k[3]), reverse=False)
new_order = [i[-1] for i in new_order_row]
     
title = title_plot+'n2_clnorm_norm'
cluster_labels = X_plot_all['cl_'+str(best_cluster)].values
X_plot = X_cl_all.reindex(new_order)
X_plot.columns = [item.split('_',maxsplit=2)[-1].replace('_', ' ')  for item in X_plot.keys()]
plot_heatmap(X_plot, color_lbl, cluster_labels, title, v_max=v_max_norm)


title = title_plot+'n3_clnorm_raw'
cluster_labels = X_plot_all['cl_'+str(best_cluster)].values
X_plot = X_raw.reindex(new_order)
X_plot.columns = [item.split('_',maxsplit=2)[-1].replace('_', ' ')  for item in X_plot.keys()]
plot_heatmap(X_plot, color_lbl, cluster_labels, title, v_max=v_max_raw)


## sort the data by the mean raw value
new_order_row = [list(r) for r in zip(X_plot_all['cl_'+str(best_cluster)], X_plot_all['wmh_sumJHU'], X_plot_all.index)]
new_order_row.sort(key=lambda k: (k[0], k[1], k[2]), reverse=False)
new_order = [i[-1] for i in new_order_row]

# plot the data raw and sorted  
title = title_plot+'n4_clnorm_sortbywmh_raw'
cluster_labels = X_plot_all['cl_'+str(best_cluster)].values
X_plot = X_raw.reindex(new_order)
X_plot.columns = [item.split('_',maxsplit=2)[-1].replace('_', ' ')  for item in X_plot.keys()]
plot_heatmap(X_plot, color_lbl, cluster_labels, title, v_max=v_max_raw)

## plot the data normed and sorted   
title = title_plot+'n5_clnorm_sortbywmh_norm'
cluster_labels = X_plot_all['cl_'+str(best_cluster)].values
X_plot = X_cl_all.reindex(new_order)
X_plot.columns = [item.split('_',maxsplit=2)[-1].replace('_', ' ')  for item in X_plot.keys()]
plot_heatmap(X_plot, color_lbl, cluster_labels, title, v_max=v_max_norm)


# plot the lesion load in the different clusters
lesion_load = pd.DataFrame(new_order_row)
lesion_load.index = lesion_load[2].values
lesion_load = lesion_load.drop(labels=2, axis=1)
lesion_load.columns = ['cluster','raw']

lesion_load['raw_sort1'] = range(0,lesion_load.shape[0])

aa = []
for jj in np.unique(lesion_load.cluster):
    a = list(range(0,sum(lesion_load.cluster==jj)))
    aa = aa+a

lesion_load['raw_sort2'] = aa

for kk in np.unique(lesion_load.cluster):

    vals = lesion_load.query('cluster==@kk').raw.values
#    vals = vals/np.max(vals)

    fig = plt.figure(figsize=(len(vals)/6,2))
    ax = fig.add_subplot(111)
    ax.plot(vals)
    ax.axhline(y=np.mean(vals),color='r', linestyle='--')

#    plt.axis('off')
#    ax.yaxis.set_visible(False)
#    ax.xaxis.set_visible(False)
    
    plt.ylim(0,np.ceil(lesion_load['raw'].max()))
    plt.xlim(0,len(vals)-0.5)
    
    if kk == 0:
        plt.title('HC')
        ax.axes.get_yaxis().set_visible(False)
        
    elif kk==lesion_load.cluster.max():
        plt.title('cluster: '+str(int(kk)))
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        plt.ylabel('total lesion load')
    else: 
        plt.title('cluster: '+str(int(kk)))
#        ax.axes.get_yaxis().set_visible(False)
        
    plt.savefig(path_save_new+title_plot+'n6_load_cl'+str(int(kk)), bbox_inches='tight')
#    plt.close()



### add the new cluster sorting by mean raw value to the dataframe

df3 = pd.concat([df2,lesion_load.iloc(axis=1)[1:]], axis=1, sort=False)

# set the subjects without WMH to zero
ind_noWMH = df3[df3.raw.isna()].filter(regex='raw')
df3.set_value(index=ind_noWMH.index, col=ind_noWMH.keys().values, value=0)

df3.to_csv(path_save_new+'df3.csv')



#_________________________________
# save lesion map of the clusters \___________________________________________

# create the lesion frequency map of the clinical labels and plot them
#les_map = 'LPA_les_map_mni'
les_map = 'LGA_les_map_mni'

les_thresh = lesion_thresh_lga

lesion_frequency_maps(df3, path_save_new, les_map, les_thresh)




#=============
# ANALYSIS    \================================================================


#___________________________________________
# compare the NP events between the cohorts \__________________________________
#
#### NP
#new_order_row = [list(r) for r in zip(df3['cl_'+str(nocl_elbow)], df3.cohort, df3.label.str.lower(), df3.sex, df3.FLAIR, df3.label2.str.lower(), df3.index)]
#new_order_row.sort(key=lambda k: (k[0], k[1], k[2], k[5], k[6]), reverse=False)
##new_order_row.sort(key=lambda k: (k[0], k[1], k[4]), reverse=False)
#new_order = [i[-1] for i in new_order_row]


X_plot = df3.copy()


# sort by 
new_order_row = [list(r) for r in zip(X_plot['cl_'+str(best_cluster)], X_plot['wmh_sumJHU'], X_plot.filter(regex='NP_').sum(axis=1), X_plot.cohort, X_plot.label.str.lower(), X_plot.index)]
new_order_row.sort(key=lambda k: (k[0], k[2], k[3], k[4]), reverse=False) # sort by number of NP events within each cluster
new_order = [i[-1] for i in new_order_row]

cluster_labels = np.zeros(len(color_lbl))
cluster_labels = df3['cl_'+str(best_cluster)]

X_plot = X_plot.reindex(new_order)
X_plot = X_plot.filter(regex='NP_')


title = title_plot+'n5_clnorm_NP'
plot_heatmap(X_plot, color_lbl, cluster_labels, title, v_max=1)
 



##### PLOT only NPs from NPSLE
X_plot_NP = df3.query('label=="NPSLE"')

# sort by 
new_order_row = [list(r) for r in zip(X_plot_NP['cl_'+str(best_cluster)], X_plot_NP['wmh_sumJHU'], X_plot_NP.filter(regex='NP_').sum(axis=1), X_plot_NP.cohort, X_plot_NP.label.str.lower(), X_plot_NP.label2.str.lower(),X_plot_NP.index)]
new_order_row.sort(key=lambda k: (k[3], k[4], k[5], k[6]), reverse=False) # sort by cohort within each cluster
#new_order_row.sort(key=lambda k: (k[0], k[2], k[3], k[4]), reverse=False) # sort by number of NP events within each cluster
#new_order_row.sort(key=lambda k: (k[0], k[1], k[3], k[4]), reverse=False) # sort by WMH vol within each cluster
new_order = [i[-1] for i in new_order_row]

#cluster_labels = np.zeros(len(color_lbl))
cluster_labels = X_plot_NP.cohort
#cluster_labels = X_plot_NP['cl_'+str(best_cluster)]

X_plot_NP = X_plot_NP.reindex(new_order)
X_plot_NP = X_plot_NP.filter(regex='NP_')


title = title_plot+'n5_clnorm_NP_nponly'
plot_heatmap(X_plot_NP, color_lbl, cluster_labels, title, v_max=1)


def stat_analysis(df_in, item, classif, alpha):
    
    test = df_in[[item,classif]].dropna(axis=0)  

    # check if there are more than 10 entries in the feature
    if test.shape[0] > 10:   
        
        ## Mann-Whitney-U-test => MAINLY for differences between the cohorts
        # get the two groups
        a = np.unique(test[classif].values)
        d = []
        for group in a:
            d.append(test[test[classif]==group][item].values)
        
        if len(d) == 2:
            res_mwu = sp.stats.mannwhitneyu(d[0],d[1], alternative='two-sided')
            p_mwu = res_mwu[1]
        else: p_mwu = np.nan
        
        ## KRUSKAL-WALLIS TEST
        a = np.unique(test[classif].values)
        d = []
        for group in a:
            d.append(test[test[classif]==group][item].values)
        if len(d)==1:
            res_kw = [99,1]
        if len(d)==2:
            res_kw = sp.stats.kruskal(d[0],d[1], nan_policy='omit')
        elif len(d)==3:
            res_kw = sp.stats.kruskal(d[0],d[1],d[2], nan_policy='omit')
        elif len(d)==4:
            res_kw = sp.stats.kruskal(d[0],d[1],d[2],d[3], nan_policy='omit')
        elif len(d)==5:
            res_kw = sp.stats.kruskal(d[0],d[1],d[2],d[3],d[4], nan_policy='omit')
        elif len(d)==6:
            res_kw = sp.stats.kruskal(d[0],d[1],d[2],d[3],d[4],d[5], nan_policy='omit')
        elif len(d)==7:
            res_kw = sp.stats.kruskal(d[0],d[1],d[2],d[3],d[4],d[5],d[6], nan_policy='omit')         
        
        p_kw = res_kw[1]
        
        ## PAIRWISE COMPARISON
        a = np.unique(test[classif].values)
        pairs = [(a[p1], a[p2]) for p1 in range(len(a)) for p2 in range(p1+1,len(a))]
        
        t_kwpair = []
        p_kwpair = []
        p_mwupair = []
        for pair in pairs:
            test1 = test.loc[(test[classif] == pair[0]) | (test[classif] == pair[1])]
            a = test1[test1[classif]==pair[0]][item]
            b = test1[test1[classif]==pair[1]][item]        
            
            # Mann-Whitney-U-test
            res_mwupair = sp.stats.mannwhitneyu(a,b, alternative='two-sided')
            p_mwupair.append(res_mwupair[1])
            
            
        print()
        print('>>>>>', item, '<<<<<')
        print()            
        print(' | KRUSKAL-WALLIS (all groups):',p_kw)             
        print(' | MANN-WHITNEY-U (all groups):',p_mwu) 
        print(' | MANN-WHITNEY-U (pairwise):')
        for pair, val in zip(pairs,p_mwupair):
            if val < 0.001:
                sign_item = '***'
            elif val < 0.01:
                sign_item = '**'
            elif val < 0.05:
                sign_item = '*'
            else: sign_item = ''
            print(str(pair[0])+' vs '+str(pair[1]), val, sign_item)
        print() 
        
        
        plt.figure()
        ax = sns.boxplot(x=classif, y=item, data=test)
        ax = sns.stripplot(x=classif, y=item, data=test, jitter=True, color=".6")
        plt.title('no of subjects=%i' %test.shape[0])
        plt.show()
            
        return_p_kw = p_kw
        return_p_mwu = p_mwu
        return_ppairs_mwu = p_mwupair
       

    else: 
        return_p_kw = 1
        return_p_mwu = 1
        return_ppairs_mwu = 1      
        
    # return the values
    return return_p_kw, return_p_mwu, return_ppairs_mwu

    
##### plot all of the data


labelby_ = 'cl_5'

X_lookat = df3.copy() 


columns_ = [
        'pat_age', 'disease_duration','age_onset',
        'sdi_score','sledai2k_score',
        'ACRno_criteria',
        'wmh_LGA_vol_thresh_ml_t1', 'wmh_LGA_no_thresh_t1'
       ]

##### TEST FOR NORMALITY
nd_true = {}
nd_false = {}

for item in X_lookat.filter(items=columns_):
#for item in X_lookat.filter(regex='LGA_probSum_'):
    
    p = sp.stats.normaltest(X_lookat[item].values, nan_policy='omit')[1]
    if p > 0.05:
        nd_true.update({item:p})
    else: nd_false.update({item: p})

print('*** TEST FOR NORMALITY')
print()
print('| normally distributed:')
for key, value in nd_true.items():
    print(key, ' : ', value)
print()
print('| NOT normally distributed:')
for key, value in nd_false.items():
    print(key, ' : ', value)


alpha = .05

fs_roi = pd.DataFrame(index=X_lookat.filter(items=columns_).keys(), columns=['tval_KW','pval_KW','pval_pairs_KW','pval_MWU','pval_pairs_MWU'])    

for item in X_lookat.filter(items=columns_):            
    pval_kw, pval_mwu, pval_pairs_mwu = stat_analysis(X_lookat, item, labelby_, alpha)
    fs_roi.loc[item].pval_KW = pval_kw
    fs_roi.loc[item].pval_MWU = pval_mwu
    fs_roi.loc[item].pval_pairs_MWU = pval_pairs_mwu       





##### stat within a cluster, eg NPSLE vs nonNPSLE
#compare_by = 'label'
#compare_by = 'cohort'
labelby_new = 'label'
  

for item in X_lookat.filter(items=columns_):
            
    for label_ in np.unique(X_lookat[labelby_new]):
        
        X_lookat_new = X_lookat.query('label==@label_')
        label_ = 'cohort'
           
        print(item, label_,np.unique(X_lookat_new.label))
    
        pval, tval, pval_pairs = stat_analysis(X_lookat_new, item, label_, alpha)           

    
### TEST FOR THE VOLUMNE ON THE TRACTS (abstract Jeoren)

sle = df3.query('label!="HC"').filter(regex='LGA_probSum_')
lbl = df3.query('label!="HC"')['cl_5']
new = pd.concat([lbl, sle], axis=1)

for item in new.filter(regex='LGA_probSum_'):
    
    p_nd = sp.stats.normaltest(new[item])[1]
    
    np = new.query('label=="NPSLE"')[item]
    nonnp = new.query('label=="nonNPSLE"')[item]
    
    p_mwu = sp.stats.mannwhitneyu(np, nonnp, alternative='two-sided')
    
    print(item)
#    print(p_nd) 
    
    if p_mwu[1] < 0.05:
        print(p_mwu)
        print(' | significance')
    print()
    


##### save the volume of the tracts
## TABLE 2 for abstract version 1.1
X_table = df3.query('cl_5!=0 and cl_5!=99')
df_tbl = pd.DataFrame(data=None, index=X_table.filter(regex='LGA_probSum_').keys(), columns=np.unique(X_table.cl_5.values))
for cl in np.unique(X_table.cl_5.values):
    for item in X_table.filter(regex='LGA_probSum_'):
        dat = X_table.query('cl_5==@cl')[item]
        
        med_ = np.round(np.median(dat),2)
        ci10_ = np.round(np.nanpercentile(dat,10),1)
        ci90_ = np.round(np.nanpercentile(dat,90),1)
        
        print(cl, item, med_, ci10_, ci90_)
        
        df_tbl[cl][item] = str(med_)+' ('+str(ci10_)+'-'+str(ci90_)+')'
  
index_new = [item.split('probSum_')[1].replace('_',' ') for item in df_tbl.filter(regex='LGA_probSum_').index.values]     
df_tbl.index = index_new 
df_tbl.to_excel(path_save_new+'_'+features_used+'table2.xlsx')  



## TABLE 3 for abstract version 3
X_table = df3.query('cl_5!=0')
variables = ['sex','pat_age','disease_duration','age_onset','sdi_score','sledai2k_score',
             'ACRno_criteria','wmh_LGA_vol_thresh_ml_t1', 'wmh_LGA_no_thresh_t1']
df_tbl = pd.DataFrame(data=None, index=variables, columns=np.unique(X_table.cl_5))
for cl in np.unique(X_table.cl_5.values):
    for item in variables:
    
        dat = X_table.query('cl_5==@cl')[item]
        
        if item != 'sex':
            med_ = np.round(np.nanmedian(dat),2)
            ci10_ = np.round(np.nanpercentile(dat,10),1)
            ci90_ = np.round(np.nanpercentile(dat,90),1)    
            print(cl, item, med_, ci10_, ci90_)    
    
            df_tbl[cl][item] = str(med_)+' ('+str(ci10_)+'-'+str(ci90_)+')'
        
        else: # sex
            female_tot = np.sum(dat == 'female')
            female_perc = np.round(100/dat.shape[0]*female_tot,1)
            
            df_tbl[cl][item] = str(female_tot)+' ('+str(female_perc)+')'

df_tbl.to_excel(path_save_new+'_'+features_used+'table3.xlsx')  


## TABLE 4 for abstract version 1.1
X_lookat = df3.query('label!="HC"')
total_num = X_lookat.shape[0]
npsle_num = X_lookat.query('label=="NPSLE"').shape[0]
nonnpsle_num = X_lookat.query('label=="nonNPSLE"').shape[0]
df_tbl = pd.DataFrame(data=None, index=np.unique(X_lookat.cl_5.values), columns=['total','NPSLE','non-NSPSLE'])
for cl in np.unique(X_lookat.cl_5.values):
    
    dat = X_lookat.query('cl_5==@cl')
    n_tot = dat.shape[0]
    n_npsle = dat.query('label=="NPSLE"').shape[0]
    n_nonnp = dat.query('label=="nonNPSLE"').shape[0]

    perc_tot = np.round(100/total_num*n_tot,1)
    perc_npsle = np.round(100/npsle_num*n_npsle,1)
    perc_nonnp = np.round(100/nonnpsle_num*n_nonnp,1)

    df_tbl['total'][cl] = str(n_tot)+' ('+str(perc_tot)+'%)'
    df_tbl['NPSLE'][cl] = str(n_npsle)+' ('+str(perc_npsle)+'%)'
    df_tbl['non-NSPSLE'][cl] = str(n_nonnp)+' ('+str(perc_nonnp)+'%)'

df_tbl.to_excel(path_save_new+'_'+features_used+'table4.xlsx')  


## SUPPLEMENTAL TABLE 2 for abstract version 1.1
X_lookat = df3.query('label!="HC"')
df_tbl = pd.DataFrame(data=None, index=X_lookat.filter(regex='LGA_probSum_').keys(), columns=['NPSLE','non-NSPSLE','p-value NPSLE vs non-NPSLE'])

for item in X_lookat.filter(regex='LGA_probSum_'):
    
    npsle = X_lookat.query('label=="NPSLE"')[item]
    nonnpsle = X_lookat.query('label=="nonNPSLE"')[item]

    # NPSLE
    med_npsle = np.round(np.median(npsle),2)
    ci10_npsle = np.round(np.nanpercentile(npsle,10),1)
    ci90_npsle = np.round(np.nanpercentile(npsle,90),1)
 
    # nonNPSLE
    med_nonnpsle = np.round(np.median(nonnpsle),2)
    ci10_nonnpsle = np.round(np.nanpercentile(nonnpsle,10),1)
    ci90_nonnpsle = np.round(np.nanpercentile(nonnpsle,90),1)

    df_tbl[item]['NPSLE'] = str(med_npsle)+' ('+str(ci10_npsle)+'-'+str(ci90_npsle)+')'
    df_tbl[item]['nonNPSLE'] = str(med_nonnpsle)+' ('+str(ci10_nonnpsle)+'-'+str(ci90_nonnpsle)+')'
    
    # Mann-Whitney-U-test
    res_mwu = sp.stats.mannwhitneyu(npsle,nonnpsle, alternative='two-sided')


index_new = [item.split('probSum_')[1].replace('_',' ') for item in df_tbl.filter(regex='LGA_probSum_').index.values]     
df_tbl.index = index_new 
df_tbl.to_excel(path_save_new+'_'+features_used+'suppl_table2.xlsx') 




### SAVE RESULTS for each cluster
cols = ['no','female/male','Leiden/Lund','NPSLE/nonNPSLE', 'NPSLE Leiden (inf/isc)','NPSLE Lund (A/B/not)',
        '3D FLAIR (%)',
        'age median (CI)','disease duration years median (CI)','age of onset median (CI)',
        'SDI-score median (CI)','SLEDAI2K-score median (CI)','ACR criteria median (CI)',
        'WMH volume ml median (CI)','WMH number median (CI)',
        'anti-nuclear antibodies (%)','anti-ds-DNA antibodies (%)','anti Sm-nuclear antigen (%)'
        ]


df_table = pd.DataFrame(index=np.sort(X_lookat[labelby_].unique()), columns=cols)

for cl in df_table.index:
    
#    cl = int(cl)
    
    d = X_lookat[X_lookat['cl_'+str(best_cluster)] == cl]

    
    df_table.loc[cl].no = d.shape[0]
    female = d.query('sex=="female"').shape[0]
    male = d.query('sex=="male"').shape[0]
    df_table.loc[cl]['female/male'] = str(female)+'/'+str(male)

    leiden = d.query('cohort=="leiden"').shape[0]
    lund = d.query('cohort=="lund"').shape[0]
    df_table.loc[cl]['Leiden/Lund'] = str(leiden)+'/'+str(lund)

    NPSLE = d.query('label=="NPSLE"').shape[0]
    nonNPSLE = d.query('label=="nonNPSLE"').shape[0]    
    df_table.loc[cl]['NPSLE/nonNPSLE'] = str(NPSLE)+'/'+str(nonNPSLE)
    
    NPSLE_leiden = d.query('cohort=="leiden" & label=="NPSLE"').shape[0]
    NPSLE_lund = d.query('cohort=="lund" & label=="NPSLE"').shape[0]
    
    if NPSLE_leiden > 0:    
        NPSLE_inf = d.query('label2=="NPSLE_inflammatory"').shape[0]
        NPSLE_isc = d.query('label2=="NPSLE_ischemic"').shape[0]  
        df_table.loc[cl]['NPSLE Leiden (inf/isc)'] = str(NPSLE_leiden)+' ('+str(int(100*NPSLE_inf/NPSLE_leiden))+'/'+str(int(100*NPSLE_isc/NPSLE_leiden))+')'
    
    if NPSLE_lund > 0: 
        NPSLE_A = d.query('label2=="NPSLE_A"').shape[0]
        NPSLE_B = d.query('label2=="NPSLE_B"').shape[0] 
        NPSLE_not = d.query('label2=="NPSLE"').shape[0] 
        df_table.loc[cl]['NPSLE Lund (A/B/not)'] = str(NPSLE_lund)+' ('+str(int(100*NPSLE_A/NPSLE_lund))+'/'+str(int(100*NPSLE_B/NPSLE_lund))+'/'+str(int(100*NPSLE_not/NPSLE_lund))+')'      

  
    df_table.loc[cl]['3D FLAIR (%)'] = int(100*d.query('FLAIR=="3D"').shape[0]/d.shape[0])
    
    age_median = np.round(d.pat_age.median(), decimals=1)
    age_ci1 = np.round(np.nanpercentile(d['pat_age'], q=10), decimals=1)
    age_ci2 = np.round(np.nanpercentile(d['pat_age'], q=90), decimals=1)  
#    age_std = np.round(d.pat_age.std(), decimals=1)
    df_table.loc[cl]['age mean (std)'] = str(age_median)+' ('+str(age_ci1)+'-'+str(age_ci2)+')'
    
    wmhvol_median = np.round(d[wmh_vol].median(),decimals=2)
    wmhvol_ci1 = np.round(np.nanpercentile(d[wmh_vol], q=10), decimals=1)
    wmhvol_ci2 = np.round(np.nanpercentile(d[wmh_vol], q=90), decimals=1)       
    df_table.loc[cl]['WMH volume ml median (CI)'] = str(wmhvol_median)+' ('+str(wmhvol_ci1)+'-'+str(wmhvol_ci2)+')'

    wmhnum_median = int(d[wmh_no].median())
    wmhnum_ci1 = int(np.round(np.nanpercentile(d[wmh_no], q=10), decimals=0))
    wmhnum_ci2 = int(np.round(np.nanpercentile(d[wmh_no], q=90), decimals=0))   
    df_table.loc[cl]['WMH number median (CI)'] = str(wmhnum_median)+' ('+str(wmhnum_ci1)+'-'+str(wmhnum_ci2)+')'
    
    if cl > 0:

        duration_median = np.round(d.disease_duration.median(), decimals=1)
        duration_ci1 = int(np.round(np.nanpercentile(d.disease_duration, q=10), decimals=0))
        duration_ci2 = int(np.round(np.nanpercentile(d.disease_duration, q=90), decimals=0))
        df_table.loc[cl]['disease duration years median (CI)'] = str(duration_median)+' ('+str(duration_ci1)+'-'+str(duration_ci2)+')'
        
        ageonset_median = np.round(d.age_onset.median(), decimals=1)
        ageonset_ci1 = int(np.round(np.nanpercentile(d.age_onset, q=10), decimals=0))
        ageonset_ci2 = int(np.round(np.nanpercentile(d.age_onset, q=90), decimals=0))   
        df_table.loc[cl]['age of onset median (CI)'] = str(ageonset_median)+' ('+str(ageonset_ci1)+'-'+str(ageonset_ci2)+')'
        
    
        sdi_median = np.round(d.sdi_score.median(), decimals=1)
        sdi_ci1 = int(np.round(np.nanpercentile(d.sdi_score, q=10), decimals=0))
        sdi_ci2 = int(np.round(np.nanpercentile(d.sdi_score, q=90), decimals=0))          
        df_table.loc[cl]['SDI-score median (CI)'] = str(sdi_median)+' ('+str(sdi_ci1)+'-'+str(sdi_ci2)+')'  
        
        sledai_median = np.round(d.sledai2k_score.median(), decimals=1)
        sledai_ci1 = int(np.round(np.nanpercentile(d.sledai2k_score, q=10), decimals=0))
        sledai_ci2 = int(np.round(np.nanpercentile(d.sledai2k_score, q=90), decimals=0))          
        df_table.loc[cl]['SLEDAI2K-score median (CI)'] = str(sledai_median)+' ('+str(sledai_ci1)+'-'+str(sledai_ci2)+')'
    
        acr_median = np.round(d.ACRno_criteria.median(), decimals=1)   
        acr_ci1 = int(np.round(np.nanpercentile(d.ACRno_criteria, q=10), decimals=0))
        acr_ci2 = int(np.round(np.nanpercentile(d.ACRno_criteria, q=90), decimals=0))
        df_table.loc[cl]['ACR criteria median (CI)'] = str(acr_median)+' ('+str(acr_ci1)+'-'+str(acr_ci2)+')'    
        
        df_table.loc[cl]['anti-nuclear antibodies (%)'] = int(100*d.ACR_ana_level.sum()/d.shape[0])
        df_table.loc[cl]['anti-ds-DNA antibodies (%)'] = int(100*d.ACR_anti_dsdna.sum()/d.shape[0])      
        df_table.loc[cl]['anti Sm-nuclear antigen (%)'] = int(100*d.ACR_anti_sm.sum()/d.shape[0]) 
       
df_table.T.to_excel(path_save_new+'_'+features_used+'cluster_data.xlsx')  



################## Linear regression of the clusters with some meta data

from scipy.stats import pearsonr
from scipy.stats import spearmanr

columns_NEW = [
        'pat_age', 'disease_duration', 'age_onset',
        'sdi_score','sledai2k_score',
#        'ACRno_criteria'
       ]



for cl in X_lookat.query('label!="HC"').filter(regex='LGA_probSum'):
    X_reg = X_lookat.query('label!="HC"')
    for item in columns_NEW:

        x = np.log(x_reg[cl])
        x = x.replace([np.inf, -np.inf], np.nan)

        y = X_reg[item]

        if y.shape[0] != 0 and x.shape[0] != 0:
            r_val, p_val = sp.stats.spearmanr(x,y, nan_policy='omit')
        
            if p_val < 0.05 and r_val >0.5:
                print(cl, item, r_val, p_val)
                sns.jointplot(x=x, y=y,  kind = 'reg', stat_func=spearmanr) 
                plt.suptitle(cl, fontsize = 16)
                plt.ylabel(item)
                plt.xlabel('log-transformed WMH load')
                plt.savefig(path_save_new+'4_corr_wmhvol_vs_'+item+'_'+cl, bbox_inches='tight')  






print('......THE END')
