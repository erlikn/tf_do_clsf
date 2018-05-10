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

def output_rgs(batchImages, batchPcl, bTargetT, bTargetP, bPrevP, batchTFrecFileIDs, **kwargs):
    """
    """
    num_cores = multiprocessing.cpu_count() - 2
    #Parallel(n_jobs=num_cores)(delayed(_output_loop_rgs_morph)(batchImages, batchPcl, bTargetT, bTargetP, bPrevP, batchTFrecFileIDs, i, **kwargs) for i in range(kwargs.get('activeBatchSize')))
    for i in range(kwargs.get('activeBatchSize')):
        _output_loop_rgs_morph(batchImages, batchPcl, bTargetT, bTargetP, bPrevP, batchTFrecFileIDs, i, **kwargs)
    return

def output_rgs_colorOnly(batchImages, bTargetT, bTargetP, bPrevP, batchTFrecFileIDs, **kwargs):
    """
    """
    num_cores = multiprocessing.cpu_count() - 2
    #Parallel(n_jobs=num_cores)(delayed(_output_loop_rgs_morph)(batchImages, bTargetT, bTargetP, bPrevP, batchTFrecFileIDs, i, **kwargs) for i in range(kwargs.get('activeBatchSize')))
    for i in range(kwargs.get('activeBatchSize')):
        _output_loop_rgs_morph(batchImages, bTargetT, bTargetP, bPrevP, batchTFrecFileIDs, i, **kwargs)
    return

def _output_loop_rgs_morph(batchImages, batchPcl, bTargetT, bTargetP, bPrevP, batchTFrecFileIDs, i, **kwargs):
    """
    """
    numTuples = kwargs.get('numTuple')
    outBatchPcl = batchPcl.copy()
    outBatchImages = batchImages.copy()
    outTargetT = bTargetT.copy()
    outPrevP = bPrevP.copy()
    # split for depth dimension
    if bPrevP is 0:
        pclBTransformed, targetRes, depthBTransformed = _apply_prediction_pcl_depImg(batchPcl[i,:,:,numTuples-1], bTargetT[i,:,numTuples-2], targetP[i], **kwargs)
        outBatchPcl[i,:,:,numTuples-1] = pclBTransformed
    
    else:
        pPrev, targetRes, depthBTransformed = _apply_prediction_depImg(batchPcl[i,:,:,numTuples-1], bTargetT[i,:,numTuples-2], bTargetP[i], bPrevP[i], False, **kwargs)
        outPrevP[i,:,numTuples-2] = pPrev.reshape([12])

    outBatchImages[i,:,:,numTuples-1] = depthBTransformed
    outTargetT[i,:,numTuples-2] = targetRes
    # Write each Tensorflow record
    filename = str(batchTFrecFileIDs[i][0]+100) + "_" + str(batchTFrecFileIDs[i][1]+100000) + "_" + str(batchTFrecFileIDs[i][2]+100000)
    if bPrevP is 0: 
        tfrecord_io.tfrecord_writer_ntuple(batchTFrecFileIDs[i],
                                           outBatchPcl[i],
                                           outBatchImages[i],
                                           outTargetT[i],
                                           kwargs.get('warpedOutputFolder')+'/',
                                           numTuples,
                                           filename)
    #else:
    #    tfrecord_io.tfrecord_writer_ntuple(batchTFrecFileIDs[i],
    #                                       outBatchPcl[i],
    #                                       outBatchImages[i],
    #                                       outTargetT[i],
    #                                       outPrevP[i],
    #                                       kwargs.get('warpedOutputFolder')+'/',
    #                                       numTuples,
    #                                       filename)

    if kwargs.get('phase') == 'train':
        folderTmat = kwargs.get('tMatTrainDir')
    else:
        folderTmat = kwargs.get('tMatTestDir')
    ####################### print('only will work for first round p0 * p1 should be done in apply pred and getten back')
    _write_predictions(batchTFrecFileIDs[i], bTargetP[i], folderTmat)
    return

def _output_loop_rgs_morph_colorOnly(batchImages, bTargetT, bTargetP, bPrevP, batchTFrecFileIDs, i, **kwargs):
    """
    """
    numTuples = kwargs.get('numTuple')
    outBatchImages = batchImages.copy()
    outTargetT = bTargetT.copy()
    outPrevP = bPrevP.copy()
    # split for depth dimension
    if bPrevP is 0: # no previous prediction mode DOESNOT MAKE SENSE --- APPLY PREDICTION ALSO DOESN'T MAKE SENSE
        targetRes, imgColorB = _apply_prediction_colorOnly(batchImages[i,:,:,numTuples-1], bTargetT[i,:,numTuples-2], targetP[i], **kwargs)
        outBatchImages[i,:,:,numTuples-1] = imgColorB

    else:
        pPrev, targetRes, depthBTransformed = _apply_prediction_colorOnly(batchImages[i,:,:,numTuples-1], bTargetT[i,:,numTuples-2], bTargetP[i], bPrevP[i], False, **kwargs)
        outBatchImages[i,:,:,numTuples-1] = imgColorB
        outPrevP[i,:,numTuples-2] = pPrev.reshape([12])

    outBatchImages[i,:,:,numTuples-1] = depthBTransformed
    outTargetT[i,:,numTuples-2] = targetRes
    # Write each Tensorflow record
    filename = str(batchTFrecFileIDs[i][0]+100) + "_" + str(batchTFrecFileIDs[i][1]+100000) + "_" + str(batchTFrecFileIDs[i][2]+100000)
    #if bPrevP is 0: 
    #    tfrecord_io.tfrec_writer_nt_colorOnly(batchTFrecFileIDs[i],
    #                                           outBatchImages[i],
    #                                           outTargetT[i],
    #                                           kwargs.get('warpedOutputFolder')+'/',
    #                                           numTuples,
    #                                           filename)
    #else:
    #### bPrevP is always available
    tfrecord_io.tfrec_writer_nt_colorOnly(batchTFrecFileIDs[i],
                                           outBatchImages[i],
                                           outTargetT[i],
                                           outPrevP[i],
                                           kwargs.get('warpedOutputFolder')+'/',
                                           numTuples,
                                           filename)

    if kwargs.get('phase') == 'train':
        folderTmat = kwargs.get('tMatTrainDir')
    else:
        folderTmat = kwargs.get('tMatTestDir')
    ####################### print('only will work for first round p0 * p1 should be done in apply pred and getten back')
    _write_predictions(batchTFrecFileIDs[i], bTargetP[i], folderTmat)
    return


def _write_json_file(filename, datafile):
    filename = 'Model_Settings/../'+filename
    datafile = collections.OrderedDict(sorted(datafile.items()))
    with open(filename, 'w') as outFile:
        json.dump(datafile, outFile, indent = 0)

def _set_folders(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

def _write_predictions(tfrecID, targetP, folderOut):
    """
    Write prediction outputs to generate path map
    """
    _set_folders(folderOut)
    dataJson = {'seq' : tfrecID[0].tolist(),
                'idx' : tfrecID[1].tolist(),
                'idxNext' : tfrecID[2].tolist(),
                'tmat' : targetP.tolist()}
    _write_json_file(folderOut + '/' + str(tfrecID[0]) + '_' + str(tfrecID[1]) + '_' + str(tfrecID[2]) +'.json', dataJson)
    return
def _apply_prediction_pcl_depImg(pclSource, targetT, targetP, params6=True, **kwargs):
    '''
    Transform pclSource, Calculate new targetT based on targetP, Create new depth image, and pcl file
    Return:
        - New pclSource
        - New targetT
        - New depthImageB
    '''
    # remove trailing zeros
    pclSource = kitti.remove_trailing_zeros(pclSource)
    if params6:
        tMatP = kitti._get_tmat_from_params(targetP)
        tMatT = kitti._get_tmat_from_params(targetT)
    else:
        tMatP = targetP.reshape([3,4])
        tMatT = targetT.reshape([3,4])
    # get transformed pclSource based on targetP
    pclSourceTransformed = kitti.transform_pcl(pclSource, tMatP)    
    # get new depth image of transformed pclSource
    depthImageB, _ = kitti.get_depth_image_pano_pclView(pclSourceTransformed)
    pclSourceTransformed = kitti._zero_pad(pclSourceTransformed, kwargs.get('pclCols')-pclSourceTransformed.shape[1])
    # get residual Target
    tMatResP2T = kitti.get_residual_tMat_p2t(tMatT, tMatP) # first is source2target, second is source2predicted
    if params6:
        targetResP2T = kitti._get_tmat_from_params(tMatResP2T)
    else:
        targetResP2T = tMatResP2T.reshape([12])
    return pclSourceTransformed, targetResP2T, depthImageB

def _apply_prediction_depImg(pclSource, targetT, targetP, prevP, params6=True, **kwargs):
    '''
    Transform pclSource, Calculate new targetT based on targetP, Create new depth image
    DOESN"T TOUCH THE PCL, BUT USES prevP TO DO TRANSFORMATION FOR CORRECT DEPTH IMAGE
    
    TargetP
        is transformation from space pi to space t.
    prevP
        is transformation from space s to space pi. s to p0 is identity.
    targetT
        is transformation from space s to space t.
    Return:
        - New pclSource
        - New targetT
        - New depthImageB
    '''
    # remove trailing zeros
    pclSource = kitti.remove_trailing_zeros(pclSource)
    if params6:
        tMatP = kitti._get_tmat_from_params(targetP)
        tMatT = kitti._get_tmat_from_params(targetT)
        tMatPrevP = kitti._get_tmat_from_params(prevP)
    else:
        tMatP = targetP.reshape([3,4])
        tMatT = targetT.reshape([3,4])
        tMatPrevP = prevP.reshape([3,4])
    # get the transformation from space s to pi
    tMatS2P = np.matmul(kitti._add_row4_tmat(tMatPrevP), kitti._add_row4_tmat(tMatP))
    tMatS2P = kitti._remove_row4_tmat(tMatS2P)
    # get transformed pclSource based on targetP
    pclSourceTransformed = kitti.transform_pcl(pclSource, tMatS2P)
    # get new depth image of transformed pclSource
    depthImageB, _ = kitti.get_depth_image_pano_pclView(pclSourceTransformed)
    # get residual Target
    tMatResP2T = kitti.get_residual_tMat_p2t(tMatT, tMatS2P) # first is source2target, second is source2predicted
    if params6:
        targetResP2T = kitti._get_tmat_from_params(tMatResP2T)
    else:
        targetResP2T = tMatResP2T.reshape([12])
    return tMatS2P, targetResP2T, depthImageB

def _apply_prediction_colorOnly(imgColor, targetT, targetP, params6=True, **kwargs):
    '''
    Transform imgColor, Calculate new targetT based on targetP, Create new depth image, and pcl file
    Return:
        - New imgColor
        - New targetT
        - New depthImageB
    '''
    # target is B2A (next2prev)
    if params6:
        tMatP = kitti._get_tmat_from_params(targetP)
        tMatT = kitti._get_tmat_from_params(targetT)
    else:
        tMatP = targetP.reshape([3,4])
        tMatT = targetT.reshape([3,4])
    # get transformed color image based on iterative target
    imgColorB = kitti.transform_image(imgColorB, tMatP)

    # get residual Target OR [NEW TARGET]
    tMatResP2T = kitti.get_residual_tMat_p2t(tMatT, tMatP) # first is source2target, second is source2predicted
    
    if params6:
        targetResP2T = kitti._get_tmat_from_params(tMatResP2T)
    else:
        targetResP2T = tMatResP2T.reshape([12])
    return targetResP2T, imgColorB

def _apply_prediction_colorOnly(imgColor, targetT, targetP, prevP, params6=True, **kwargs):
    '''
    Transform imgColor, Calculate new targetT based on targetP, Create new depth image, and pcl file
    Return:
        - New imgColor
        - New targetT
        - New depthImageB
    '''
    # target is B2A (next2prev)
    if params6:
        tMatP = kitti._get_tmat_from_params(targetP)
        tMatT = kitti._get_tmat_from_params(targetT)
        tMatPrevP = kitti._get_tmat_from_params(prevP)
    else:
        tMatP = targetP.reshape([3,4])
        tMatT = targetT.reshape([3,4])
        tMatPrevP = prevP.reshape([3,4])
    # get the transformation from space s to pi
    tMatS2P = np.matmul(kitti._add_row4_tmat(tMatPrevP), kitti._add_row4_tmat(tMatP))
    tMatS2P = kitti._remove_row4_tmat(tMatS2P)
    # get transformed color image based on iterative target
    imgColorB = kitti.transform_image(imgColorB, tMatS2P)

    # get residual Target OR [NEW TARGET]
    tMatResP2T = kitti.get_residual_tMat_p2t(tMatT, tMatP) # first is source2target, second is source2predicted
    
    if params6:
        targetResP2T = kitti._get_tmat_from_params(tMatResP2T)
    else:
        targetResP2T = tMatResP2T.reshape([12])
    return tMatS2P, targetResP2T, imgColorB