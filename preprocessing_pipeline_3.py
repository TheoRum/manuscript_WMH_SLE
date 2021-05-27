#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

 | Title:
     part 3 of the preprocessing
     
 | Date:
     2021-02-01
     
 | Author(s):
     Theodor Rumetshofer

 | Description:    
     This is doing the second part of the preprocessing for the Leiden and Lund 
     cohort. Subjects from the Leiden cohort with a 3D FLAIR were manually pre-
     processed using FLIRT from FSL (see README.md) using reorientation and co-
     registration to T1.
     In this pipeline all lesion maps from all subjects were transformed into 
     MNI-space by using the transformation matrix from the T1-transformation to
     MNI. 

 | List of functions:
     No user defined functions are used in the program.

 | List of "non standard" modules:
     None

 | Procedure:
     1) define input data
     2) mask the T1 images
     3) apply the mask on the lesion maps
     4) transform the T1 to MNI-space
     5) apply the transformation matrix from the T1 to the lesion map

 | Usage:
     ./preprocessing_pipeline_3.py

"""
 
from __future__ import (print_function, division, absolute_import)
import numpy as np
import nipype
import nipype.interfaces.ants as ants
import nipype.interfaces.fsl as fsl
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import IdentityInterface
import nipype.interfaces.io as nio 
import nipype as nipy
from glob import glob as gl
from os.path import join as opj


## PATHS
PATH = '/Users/data/_analysis/' # mac
img_path = PATH+'SLE/'


# number of threads
num_threads_node = 8


#############################################################################

#__________________
# define variables \_________________________________________________________

# location of experiment folder
experiment_dir = img_path+'/data'

# get names of subjects
lst_file_found= experiment_dir+'/*/ples_lga_*_lst.nii.gz' # LGA 
subject_lst_found = gl(lst_file_found)
subject_lst_found = np.sort(subject_lst_found)
subject_list = [item.split('/segm_lesion')[0].split('/')[-1] for item in subject_lst_found] 

# working directory and output folder
output_dir = '_outputdir'
working_dir = '_workingdir'
reference_dir = img_path

# define mni_template for transformation into MNI-space
mni_brain = 'MNI152_T1_1mm_brain.nii.gz'



#______________
# define Nodes \______________________________________________________________

# FSL - brain extraction
node_bet = Node(fsl.BET(frac=0.5
                        , mask=True
                        , output_type = 'NIFTI_GZ'
                        )
                , name='node_bet')


# FSL - apply mask
node_mask = Node(fsl.ApplyMask(), name='node_mask')

  
# ANTS - registration anat with MNI brain 1mm
node_reg = Node(ants.Registration(
                args='--float'
                , transforms = ['Rigid', 'Affine', 'SyN']
                , transform_parameters = [(0.1,),
                                          (0.1,),
                                          (0.1, 3.0, 0.0)]
                , dimension = 3
                , interpolation = 'BSpline'
                , collapse_output_transforms = True
                , initial_moving_transform_com = True
                , initialize_transforms_per_stage = False
                , output_warped_image=True
                , output_inverse_warped_image=True
                , sigma_units=['vox']*3
                , metric = ['Mattes', 'Mattes', 'CC']
                , metric_weight = [1.0, 1.0, 1.0]              
                
                , number_of_iterations = [[1000, 500, 250, 100],
                                          [1000, 500, 250, 100],
                                          [100, 70, 50, 20]]
                , smoothing_sigmas = [[3, 2, 1, 0],
                                      [3, 2, 1, 0],
                                      [3, 2, 1, 0]]
                , shrink_factors = [[8, 4, 2, 1],
                                    [8, 4, 2, 1],
                                    [8, 4, 2, 1]]
                , use_estimate_learning_rate_once = [True, True, True]
                , use_histogram_matching = [False, False, True]
                , radius_or_number_of_bins = [32, 32, 4]
                , sampling_percentage=[0.25, 0.25, 1]
                , sampling_strategy=['Regular','Regular','None']
                , convergence_threshold = [1e-6]
                , convergence_window_size=[10]
                , write_composite_transform=True
                , terminal_output='file'
                , num_threads = num_threads_node
                )
                , name='node_reg')


# ANTS - transform lesion prob map into mni
node_applytransform = Node(ants.ApplyTransforms(
                                  interpolation='NearestNeighbor'
                                  , invert_transform_flags=[False]
                                  , num_threads = num_threads_node
                                  ),
                  name='node_applytransform')

    

#_____________________________________
# infosource, input and output stream \_______________________________________

## infosource - a function free node to iterate over the list of subject names
infosource = Node(IdentityInterface(fields=['subject_id']),name="infosource")

infosource.iterables = [('subject_id', subject_list)] 
 
   
## input stream (SelectFiles)
data_pool = {'anat': '{subject_id}/T1.nii.gz'
             , 'ples_realign': '{subject_id}/ples_lga_*_lst.nii.gz'
             }

selectfiles = Node(nipy.SelectFiles(data_pool,
                                    base_directory=experiment_dir),
                        name="selectfiles")


## input stream for reference files
data_ref = {
        'mni_brain': 'MNI_template/'+mni_brain
            }

selectrefs = Node(nipy.SelectFiles(data_ref,
                                   base_directory=reference_dir
                                   ), name='selectrefs')


## output stream (datasink)
datasink = Node(nio.DataSink(base_directory=experiment_dir
                              , container=output_dir),
                    name="datasink")



#_________________________________
# define workflow & connect nodes \___________________________________________

#Create a workflow to connect all those nodes
analysisflow = Workflow(name='analysisflow', base_dir=opj(experiment_dir, working_dir))

# connect all nodes to a workflow and with SelectFiles and DataSink to the workflow
analysisflow.connect([
                        # get the source
                        (infosource, selectfiles, [('subject_id', 'subject_id')])
                        

                        # node bet
                        , (selectfiles, node_bet, [('anat', 'in_file')])
                        , (node_bet, datasink, [('out_file', 'bet.@brain')
                                                , ('mask_file', 'bet.@mask')
                                                ])
                        
                        # node masking
                        , (selectfiles, node_mask, [('ples_realign', 'in_file')])
                        , (node_bet, node_mask, [('bet_mask', 'mask_file')])
                        , (node_mask, datasink, [('out_file', 'bet.@prob_brain')])
                        

                        # node registration
                        , (selectrefs, node_reg, [('mni_brain', 'fixed_image')])
                        , (node_bet, node_reg, [('out_file', 'moving_image')])
                        , (node_reg, datasink, [('warped_image', 'antsreg.@warped_image')
                                                , ('composite_transform', 'antsreg.@transform')
                                                , ('inverse_composite_transform','antsreg.@inverse_transform')
                                                ])
                       
                        # node apply transform
                        , (selectrefs, node_applytransform, [('mni_brain', 'reference_image')])
                        , (node_mask, node_applytransform, [('out_file', 'input_image')])
                        , (node_reg, node_applytransform, [('composite_transform', 'transforms')])
                        , (node_applytransform, datasink, [('output_image', 'antsapply.@lesion')])                        
                        ])

    
#_________________________
# run workflow / pipeline \___________________________________________________
analysisflow.write_graph(dotfilename='workflow_graph', graph2use='orig', format='png', simple_form=True)

# MULTI PROC 
analysisflow.run('MultiProc') # deteced automatically










print('>>>>>>>>>>>>>> The END')
