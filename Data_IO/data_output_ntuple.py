# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, join
from os import walk
import os
import json
import collections
import math
import random
from shutil import copy
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

from joblib import Parallel, delayed
import multiprocessing

import Data_IO.tfrecord_io as tfrecord_io
import Data_IO.kitti_shared as kitti

def _apply_prediction(pclB, targetT, targetP, **kwargs):
    '''
    Transform pclB, Calculate new targetT based on targetP, Create new depth image
    Return:
        - New PCLB
        - New targetT
        - New depthImageB
    '''
    # remove trailing zeros
    pclA = kitti.remove_trailing_zeros(pclB)
    # get transformed pclB based on targetP
    tMatP = kitti._get_tmat_from_params(targetP) 
    pclBTransformed = kitti.transform_pcl(pclB, tMatP)
    # get new depth image of transformed pclB
    depthImageB, _ = kitti.get_depth_image_pano_pclView(pclBTransformed)
    pclBTransformed = kitti._zero_pad(pclBTransformed, kwargs.get('pclCols')-pclBTransformed.shape[1])
    # get residual Target
    #tMatResB2A = kitti.get_residual_tMat_Bp2B2A(targetP, targetT) # first is A, second is B
    targetResP2T = targetT - targetP
    return pclBTransformed, targetResP2T, depthImageB

def output_clsf(batchImages, batchPcl, bTargetVals, bTargetT, bTargetP, bRngs, batchTFrecFileIDs, **kwargs):
    """
    TODO: SIMILAR TO DATA INPUT -> WE NEED A QUEUE RUNNER TO WRITE THIS OFF TO BE FASTER

    Everything evaluated
    Warp second image based on predicted HAB and write to the new address
    Args:
        batchImages:        img
        batchPcl:           3 x n 
        bTargetVals:        b * 6 * nt
        bTargetT:           b * 6 * 32 * nT targets
        bTargetP:           b * 6 * 32 * nT predictions
        bRngs:              b * 33 * nT ranges
        batchTFrecFileIDs:  fileID 
        **kwargs:           model parameters
    Returns:
    Raises:
      ValueError: If no dataDir
    """
    num_cores = multiprocessing.cpu_count() - 2
    #Parallel(n_jobs=num_cores)(delayed(output_loop_clsf)(batchImages, batchPcl, bTargetT, bTargetP, bRngs, batchTFrecFileIDs, i, **kwargs) for i in range(kwargs.get('activeBatchSize')))
    for i in range(kwargs.get('activeBatchSize')):
        output_loop_clsf(batchImages, batchPcl, bTargetVals, bTargetT, bTargetP, bRngs, batchTFrecFileIDs, i, **kwargs)
    return

def output_loop_clsf(batchImages, batchPcl, bTargetVals, bTargetT, bTargetP, bRngs, batchTFrecFileIDs, i, **kwargs):
    """
    TODO: SIMILAR TO DATA INPUT -> WE NEED A QUEUE RUNNER TO WRITE THIS OFF TO BE FASTER

    Everything evaluated
    Warp second image based on predicted HAB and write to the new address
    Args:
        batchImages:        img
        batchPcl:           3 x n 
        bTargetVals:        b * 6 * nt
        bTargetT:           b * 6 * 32 * nT targets
        bTargetP:           b * 6 * 32 * nT predictions
        bRngs:              b * 33 * nT ranges
        batchTFrecFileIDs:  fileID 
        i:                  ID of the batch
        **kwargs:           model parameters
    Returns:
        N/A
    Raises:
        ValueError: If no dataDir
    """
    numTuples = kwargs.get('numTuple')
    # Two tasks:
    #   1 - Use the predicted bTargetP[b, 6, 32, nt] and brngs[b, 6, 33, nt] to get the parameters
    #   2 - Update the rngs based on the predicted values for each image and save them
    # find argmax for the bTargeP and use it to get the corresponding params
    if kwargs.get('lastTuple'):
        # for training on last tuple
        predParam = kitti.get_params_from_binarylogits(bTargetP[i], bRngs[i,:,:,numTuples-2:numTuples-1])
        # get updated ranges
        newRanges = kitti.get_updated_ranges(bTargetP[i], bRngs[i,:,:,numTuples-2:numTuples-1], 'squared') # softmax | squared
        # get updated bit target
        newBitTarget = kitti.get_multi_bit_target(bTargetVals[i,:,numTuples-2:numTuples-1], newRanges, bTargetP.shape[2])
    else:
        # for training on all tuples
        predParam = kitti.get_params_from_binarylogits(bTargetP[i], bRngs[i])
        # get updated ranges
        newRanges = kitti.get_updated_ranges(bTargetP[i], bRngs[i], 'squared') # softmax | squared
        # get updated bit target
        newBitTarget = kitti.get_multi_bit_target(bTargetVals[i], newRanges, bTargetP.shape[2]) # make sure this function is compatible with nT > 2

    # Apply the prediction from extracted parameters 
    
    #print("oldRNG === ", bRngs[i,0,:,3])
    #print("newRNG === ", newRanges[0,:,0])
    print("difRNG === ", newRanges[0,:,0]-bRngs[i,0,:,3])
    print('tartParams ========= ', bTargetT.shape)
    print('tarPParams ========= ', bTargetP.shape)
    print('predParams ========= ', predParam.shape)
    print('old Ranges ========= ', bRngs.shape)
    print('new Ranges ========= ', newRanges.shape)
    print('target val ========= ', bTargetVals[i,0,numTuples-2])
    print('target val =5=28= ', newRanges[0,5], newRanges[0,28])
    print('target rng =-1=0=+1= ', newRanges[0,np.argmax(newBitTarget)-1], newRanges[0,np.argmax(newBitTarget)], newRanges[0,np.argmax(newBitTarget)+1])
    #print('pred   val ========= ', predParam[1])

    #### Shift the ranges so morphed image will have correct target
    #print('new Ranges ========= ', newRanges[1])
    for i in range(newRanges.shape[0]):
        newRanges[i] -= predParam[i]
    #print('new-pred   ========= ', newRanges[1])

    #print(bTargetP[i,0,:,0])
    import matplotlib.pyplot as plt
    plt.subplot(311)
    plt.plot(bTargetT[i,0,:,3])
    normP = bTargetP[i,0,:,0]/np.linalg.norm(bTargetP[i,0,:,0])
    plt.plot(normP)
    plt.title('Target Prediction')
    #plt.show()
    
    plt.subplot(312)
    plt.plot(bRngs[i,0,:,3])
    #plt.plot(newRanges[0,2:31,0]-newRanges[0,1:30,0])
    plt.plot(newRanges[0,:,0])
    plt.title('Ranges')
    #plt.show()

    plt.subplot(313)
    plt.plot(newBitTarget[0])
    #plt.plot(newRanges[0,1:31,0]-newRanges[0,0:30,0])
    plt.title('Ranges diff')
    plt.show()

    if kwargs.get('lastTuple'):
        pclBTransformed, targetRes, depthBTransformed = _apply_prediction(batchPcl[i,:,:,numTuples-1], bTargetVals[i,:,numTuples-2], predParam, **kwargs)
        outBatchPcl = batchPcl.copy()
        outBatchImages = batchImages.copy()
        bTargetVals = bTargetT.copy()
        outBatchPcl[i,:,:,numTuples-1] = pclBTransformed
        outBatchImages[i,:,:,numTuples-1] = depthBTransformed
        outTargetT[i,:,numTuples-2] = targetRes
    #else:
        # Do WE UPDATE ALL???

    ################## TO BE FIXED APPLYING THE PREDICTION BASED ON PREDPARAM
    # split for depth dimension
    #pclBTransformed, targetRes, depthBTransformed = _apply_prediction(batchPcl[i,:,:,numTuples-1], bTargetVals[i,:,numTuples-2], predParam, **kwargs)
    #outBatchPcl = batchPcl.copy()
    #outBatchImages = batchImages.copy()
    #bTargetVals = bTargetT.copy()
    #outBatchPcl[i,:,:,numTuples-1] = pclBTransformed
    #outBatchImages[i,:,:,numTuples-1] = depthBTransformed
    #outTargetT[i,:,numTuples-2] = targetRes
    # Write each Tensorflow record
    filename = str(batchTFrecFileIDs[i][0]+100) + "_" + str(batchTFrecFileIDs[i][1]+100000) + "_" + str(batchTFrecFileIDs[i][2]+100000)
    tfrecord_io.tfrecord_writer_ntuple(batchTFrecFileIDs[i],
                                       outBatchPcl[i],
                                       outBatchImages[i],
                                       outTargetT[i],
                                       kwargs.get('warpedOutputFolder')+'/',
                                       numTuples,
                                       filename)

    # Write the predicted transformation to a folder 
    if kwargs.get('phase') == 'train':
        folderTmat = kwargs.get('tMatTrainDir')
    else:
        folderTmat = kwargs.get('tMatTestDir')
    write_predictions(batchTFrecFileIDs[i], bTargetP[i], folderTmat)
    return

def write_json_file(filename, datafile):
    filename = 'Model_Settings/../'+filename
    datafile = collections.OrderedDict(sorted(datafile.items()))
    with open(filename, 'w') as outFile:
        json.dump(datafile, outFile, indent = 0)

def _set_folders(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

def write_predictions(tfrecID, targetP, folderOut):
    """
    Write prediction outputs to generate path map
    """
    _set_folders(folderOut)
    dataJson = {'seq' : tfrecID[0].tolist(),
                'idx' : tfrecID[1].tolist(),
                'idxNext' : tfrecID[2].tolist(),
                'tmat' : targetP.tolist()}
    write_json_file(folderOut + '/' + str(tfrecID[0]) + '_' + str(tfrecID[1]) + '_' + str(tfrecID[2]) +'.json', dataJson)
    return
