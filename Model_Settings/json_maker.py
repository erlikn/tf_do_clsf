import json
import collections
import numpy as np
import os

def write_json_file(filename, datafile):
    filename = 'Model_Settings/'+filename
    datafile = collections.OrderedDict(sorted(datafile.items()))
    with open(filename, 'w') as outFile:
        json.dump(datafile, outFile, indent=0)

def _set_folders(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

####################################################################################
####################################################################################
####################################################################################

# REGRESSION --- Twin Common Parameters
baseTrainDataDir = '../Data/kitti/train_reg'
baseTestDataDir = '../Data/kitti/test_reg'
trainRegLogDirBase = '../Data/kitti/logs/tfdh_itr_reg_logs/train_logs/'
testRegLogDirBase = '../Data/kitti/logs/tfdh_itr_reg_logs/test_logs/'

# CLASSIFICATION --- Twin Correlation Matching Common Parameters
baseTrainDataDir = '../Data/kitti/train_clsf'
baseTestDataDir = '../Data/kitti/test_clsf'
trainClsfLogDirBase = '../Data/kitti/logs/tfdh_itr_clsf_logs/train_logs/'
testClsfLogDirBase = '../Data/kitti/logs/tfdh_itr_clsf_logs/test_logs/'



warpedTrainDirBase = '../Data/kitti/train_itr/'
warpedTestDirBase = '../Data/kitti/test_itr/'

####################################################################################
####################################################################################
####################################################################################
reCompileJSON = True
####################################################################################
####################################################################################
####################################################################################

def write_iterative(runName, itrNum):
    dataLocal = {
        # Data Parameters
        'numTrainDatasetExamples' : 20400,
        'numTestDatasetExamples' : 2790,
        'trainDataDir' : '../Data/kitti/train_tfrecords',
        'testDataDir' : '../Data/kitti/test_tfrecords',
        'warpedTrainDataDir' : warpedTrainDirBase+'',
        'warpedTestDataDir' : warpedTestDirBase+'',
        'trainLogDir' : trainClsfLogDirBase+'',
        'testLogDir' : testClsfLogDirBase+'',
        'tMatTrainDir' : trainClsfLogDirBase+'/target',
        'tMatTestDir' : testClsfLogDirBase+'/target',
        'writeWarped' : False,
        'pretrainedModelCheckpointPath' : '',
        # Image Parameters
        'imageDepthRows' : 128,
        'imageDepthCols' : 512,
        'imageDepthChannels' : 2, # All PCL files should have same cols
        # PCL Parameters
        'pclRows' : 3,
        'pclCols' : 62074,
        # tMat Parameters
        'tMatRows' : 3,
        'tMatCols' : 4,
        # Model Parameters
        'modelName' : '',
        'modelShape' : [64, 64, 64, 64, 128, 128, 128, 128, 1024],
        'numParallelModules' : 2,
        'batchNorm' : True,
        'weightNorm' : False,
        'optimizer' : 'MomentumOptimizer', # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        'momentum' : 0.9,
        'initialLearningRate' : 0.005,
        'learningRateDecayFactor' : 0.1,
        'numEpochsPerDecay' : 10000.0,
        'epsilon' : 0.1,
        'dropOutKeepRate' : 0.5,
        'clipNorm' : 1.0,
        'lossFunction' : 'L2',
        # Train Parameters
        'trainBatchSize' : 16,
        'testBatchSize' : 16,
        'outputSize' : 6, # 6 Params
        'trainMaxSteps' : 30000,
        'testMaxSteps' : 1,
        'usefp16' : False,
        'logDevicePlacement' : False,
        'classification' : {'Model' : False, 'binSize' : 0},
        'lastTuple' : False
        }
    dataLocal['testMaxSteps'] = int(np.ceil(dataLocal['numTestDatasetExamples']/dataLocal['testBatchSize']))

    dataLocal['writeWarpedImages'] = True
    # Iterative model only changes the wayoutput is written, 
    # so any model can be used by ease

    reCompileITR = True
    NOreCompileITR = False

    if runName == '170706_ITR_B':
        itr_170706_ITR_B_inception(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, dataLocal)
    ####
    elif runName == '171003_ITR_B': # using 170706_ITR_B but with loss for all n-1 tuples
        dataLocal['classificationModel'] = {'Model' : True, 'binSize' : 32}
        itr_171003_ITR_B_clsf(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, dataLocal)
    ####
    elif runName == '180110_ITR_B': # using 171003_ITR_B but with softmax loss for all 2 tuples
        dataLocal['classificationModel'] = {'Model' : True, 'binSize' : 32}
        itr_180110_ITR_B_clsf(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, dataLocal)
    ####
    elif runName == '180111_ITR_B': # using 171003_ITR_B but with softmax loss for all 5 tuples
        dataLocal['classificationModel'] = {'Model' : True, 'binSize' : 32}
        itr_180111_ITR_B_clsf(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, dataLocal)
    ####
    elif runName == '180111_ITR_B_4_clsf': # using 171003_ITR_B but with softmax loss for all 5 tuples
        dataLocal['classificationModel'] = {'Model' : True, 'binSize' : 32}
        itr_180111_ITR_B_clsf_long(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, dataLocal)
    ####
    elif runName == '180111_ITR_B_4_clsf_lastTup': # using 171003_ITR_B but with softmax loss for all last tuple
        dataLocal['classificationModel'] = {'Model' : True, 'binSize' : 32}
        itr_180111_ITR_B_clsf_long(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, dataLocal)
    ####
    elif runName == '180326_ITR_B_4_clsf_lastTup': # using 171003_ITR_B but with gaussian location softmax loss for all last tuple
        dataLocal['classificationModel'] = {'Model' : True, 'binSize' : 32}
        itr_180111_ITR_B_clsf_long_glsmcel2(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, dataLocal)
    ####
    elif runName == '180329_ITR_B_4_clsf_lastTup': # using 171003_ITR_B but with gaussian location softmax loss for all last tuple
        dataLocal['classificationModel'] = {'Model' : True, 'binSize' : 256}
        itr_180111_ITR_B_clsf_long_glsmcel2(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, dataLocal)
    ####
    elif runName == '180402_ITR_B_4_reg_lastTup': # using 171003_ITR_B but with transformation loss
        dataLocal['morph'] = {'model': 'depth'} # depth or both
        itr_180402_ITR_B_reg_trnsfLoss(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, dataLocal)
    ####
    elif runName == '180406_ITR_B_4_reg_lastTup': # using 171003_ITR_B but with transformation loss
        dataLocal['morph'] = {'model': 'depth'} # depth or both
        itr_180402_ITR_B_reg_trnsfLoss(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, dataLocal)
    ####
    else:
        print("--error: Model name not found!")
        return False
    return True
    ##############
    ##############
    ##############

def itr_170706_ITR_B_inception(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, data):
    if reCompileITR:
        runPrefix = runName+'_'
        data['modelName'] = 'twin_cnn_4p4l2f_inception'
        data['numParallelModules'] = 2
        data['imageDepthChannels'] = 2
        data['numTuple'] = 2
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['trainBatchSize'] = 32
        data['testBatchSize'] = 32
        data['numTrainDatasetExamples'] = 20400
        data['numTestDatasetExamples'] = 2790
        
        data['logicalOutputSize'] = 6

        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "Weighted_Params_L2_loss_nTuple_last"
        
        runName = runPrefix+str(itrNum)
        ### Auto Iteration Number
        if itrNum == 1:
            data['trainDataDir'] = baseTrainDataDir
            data['testDataDir'] = baseTestDataDir
        ### Auto Iteration Number 2,3,4
        if itrNum > 1:
            data['trainDataDir'] = warpedTrainDirBase + runPrefix+str(itrNum-1) # from previous iteration
            data['testDataDir'] = warpedTestDirBase + runPrefix+str(itrNum-1) # from previous iteration
        ####
        data['trainLogDir'] = trainClsfLogDirBase + runName
        data['testLogDir'] = testClsfLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def itr_171003_ITR_B_clsf(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, data):
    if reCompileITR:
        runPrefix = runName+'_'
        data['modelName'] = 'twin_cnn_4p4l2f_inception'
        data['numParallelModules'] = 2
        data['imageDepthChannels'] = 2
        data['numTuple'] = 2
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['trainBatchSize'] = 2#8#16
        data['testBatchSize'] = 2#8#16
        data['numTrainDatasetExamples'] = 20400
        data['numTestDatasetExamples'] = 2790
        data['logicalOutputSize'] = 6
        data['networkOutputSize'] = data['logicalOutputSize']*data['classificationModel']['binSize']
        data['lossFunction'] = "_params_classification_l2_loss_nTuple"
        
        
        runName = runPrefix+str(itrNum)
        ### Auto Iteration Number
        if itrNum == 1:
            data['trainDataDir'] = baseTrainDataDir
            data['testDataDir'] = baseTestDataDir
        ### Auto Iteration Number 2,3,4
        if itrNum > 1:
            data['trainDataDir'] = warpedTrainDirBase + runPrefix+str(itrNum-1) # from previous iteration
            data['testDataDir'] = warpedTestDirBase + runPrefix+str(itrNum-1) # from previous iteration
        ####
        data['trainLogDir'] = trainClsfLogDirBase + runName
        data['testLogDir'] = testClsfLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
    
def itr_180110_ITR_B_clsf(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, data):
    if reCompileITR:
        runPrefix = runName+'_'
        data['modelName'] = 'twin_cnn_4p4l2f_inception'
        data['numParallelModules'] = 2
        data['imageDepthChannels'] = 2
        data['numTuple'] = 2
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['trainBatchSize'] = 2#8#16
        data['testBatchSize'] = 2#8#16
        data['numTrainDatasetExamples'] = 20400
        data['numTestDatasetExamples'] = 2790
        
        data['logicalOutputSize'] = 6
        data['networkOutputSize'] = data['logicalOutputSize']*data['classificationModel']['binSize']*(data['numTuple']-1)
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss_nTuple"
        
        ## runs
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)

        runName = runPrefix+str(itrNum)
        ### Auto Iteration Number
        if itrNum == 1:
            data['trainDataDir'] = baseTrainDataDir
            data['testDataDir'] = baseTestDataDir
        ### Auto Iteration Number 2,3,4
        if itrNum > 1:
            data['trainDataDir'] = warpedTrainDirBase + runPrefix+str(itrNum-1) # from previous iteration
            data['testDataDir'] = warpedTestDirBase + runPrefix+str(itrNum-1) # from previous iteration
        ####
        data['trainLogDir'] = trainClsfLogDirBase + runName
        data['testLogDir'] = testClsfLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def itr_180111_ITR_B_clsf(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, data):
    if reCompileITR:
        runPrefix = runName+'_'
        data['modelName'] = 'twin_cnn_4p4l2f_inception'
        data['numParallelModules'] = 5
        data['imageDepthChannels'] = 5
        data['numTuple'] = 5
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['trainBatchSize'] = 2#8#16
        data['testBatchSize'] = 2#8#16
        data['numTrainDatasetExamples'] = 20400
        data['numTestDatasetExamples'] = 2790
        
        data['logicalOutputSize'] = 6
        data['networkOutputSize'] = data['logicalOutputSize']*data['classificationModel']['binSize']*(data['numTuple']-1)
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss_nTuple"
        
        ## runs
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)

        runName = runPrefix+str(itrNum)
        ### Auto Iteration Number
        if itrNum == 1:
            data['trainDataDir'] = '../Data/kitti/train_itr_5tpl/'+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def itr_180111_ITR_B_clsf_long(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, data):
    if reCompileITR:
        runPrefix = runName+'_'
        data['modelName'] = 'twin_cnn_4p4l2f_inception'
        data['numParallelModules'] = 5
        data['imageDepthChannels'] = 5
        data['numTuple'] = 5
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['trainBatchSize'] = 6#8#16
        data['testBatchSize'] = 6#8#16
        data['numTrainDatasetExamples'] = 20400
        data['numTestDatasetExamples'] = 2790
        data['logicalOutputSize'] = 6
        data['lastTuple'] = True
        # For all tuples
        if data['lastTuple']:
            data['networkOutputSize'] = data['logicalOutputSize']*data['classificationModel']['binSize']
        else:
            data['networkOutputSize'] = data['logicalOutputSize']*data['classificationModel']['binSize']*(data['numTuple']-1)
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss_nTuple"
        
        ## runs
        data['trainMaxSteps'] = 100000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)

        runName = runPrefix+str(itrNum)
        ### Auto Iteration Number
        if itrNum == 1:
            data['trainDataDir'] = '../Data/kitti/train_itr_5tpl/'+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def itr_180111_ITR_B_clsf_long_glsmcel2(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, data):
    if reCompileITR:
        runPrefix = runName+'_'
        data['modelName'] = 'twin_cnn_4p4l2f_inception'
        data['numParallelModules'] = 5
        data['imageDepthChannels'] = 5
        data['numTuple'] = 5
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['trainBatchSize'] = 6#8#16
        data['testBatchSize'] = 6#8#16
        data['numTrainDatasetExamples'] = 20400
        data['numTestDatasetExamples'] = 2790
        data['logicalOutputSize'] = 6
        data['lastTuple'] = True
        # For all tuples
        if data['lastTuple']:
            data['networkOutputSize'] = data['logicalOutputSize']*data['classificationModel']['binSize']
        else:
            data['networkOutputSize'] = data['logicalOutputSize']*data['classificationModel']['binSize']*(data['numTuple']-1)
        data['lossFunction'] = "_params_classification_gaussian_softmaxCrossentropy_loss_nTuple"
        
        ## runs
        data['trainMaxSteps'] = 100000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)

        runName = runPrefix+str(itrNum)
        ### Auto Iteration Number
        if itrNum == 1:
            data['trainDataDir'] = '../Data/kitti/train_clsf_'+str(data['numTuple'])+'_tpl_'+str(data['classificationModel']['binSize'])+'_bin'
            data['testDataDir'] = '../Data/kitti/test_clsf_'+str(data['numTuple'])+'_tpl_'+str(data['classificationModel']['binSize'])+'_bin'
        ### Auto Iteration Number 2,3,4
        if itrNum > 1:
            data['trainDataDir'] = '../Data/kitti/train_itr_'+str(data['numTuple'])+'_tpl/' + runPrefix+str(itrNum-1) # from previous iteration
            data['testDataDir'] = '../Data/kitti/test_itr_'+str(data['numTuple'])+'_tpl/' + runPrefix+str(itrNum-1) # from previous iteration
        ####
        data['trainLogDir'] = trainClsfLogDirBase + runName
        data['testLogDir'] = testClsfLogDirBase + runName
        data['warpedTrainDataDir'] = '../Data/kitti/train_itr_'+str(data['numTuple'])+'_tpl/' + runName
        data['warpedTestDataDir'] = '../Data/kitti/test_itr_'+str(data['numTuple'])+'_tpl/'+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def itr_180402_ITR_B_reg_trnsfLoss(reCompileITR, trainClsfLogDirBase, testClsfLogDirBase, runName, itrNum, data):
    if reCompileITR:
        runPrefix = runName+'_'
        data['modelName'] = 'twin_cnn_4p4l2f_inception'
        data['numParallelModules'] = 2
        data['imageDepthChannels'] = 2
        data['numTuple'] = 2
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['trainBatchSize'] = 8#16
        data['testBatchSize'] = 8#16
        data['numTrainDatasetExamples'] = 20400
        data['numTestDatasetExamples'] = 2790
        data['logicalOutputSize'] = 12 # transformation matrix
        data['lastTuple'] = True

        # For all tuples
        if data['lastTuple']:
            data['networkOutputSize'] = data['logicalOutputSize']
        else:
            data['networkOutputSize'] = data['logicalOutputSize']*(data['numTuple']-1)
        data['lossFunction'] = "_transformation_loss_nTuple_last"
        
        ## runs
        data['trainMaxSteps'] = 75000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)

        runName = runPrefix+str(itrNum)
        ### Auto Iteration Number
        if itrNum == 1:
            data['trainDataDir'] = '../Data/kitti/train_reg_'+str(data['numTuple'])+'_tpl_'+str(data['logicalOutputSize'])+'_prm'
            data['testDataDir'] = '../Data/kitti/test_reg_'+str(data['numTuple'])+'_tpl_'+str(data['logicalOutputSize'])+'_prm'
        ### Auto Iteration Number 2,3,4
        if itrNum > 1:
            data['trainDataDir'] = '../Data/kitti/train_itr_'+str(data['numTuple'])+'_tpl/' + runPrefix+str(itrNum-1) # from previous iteration
            data['testDataDir'] = '../Data/kitti/test_itr_'+str(data['numTuple'])+'_tpl/' + runPrefix+str(itrNum-1) # from previous iteration
        ####
        data['trainLogDir'] = trainRegLogDirBase + runName
        data['testLogDir'] = testRegLogDirBase + runName
        data['warpedTrainDataDir'] = '../Data/kitti/train_itr_'+str(data['numTuple'])+'_tpl/' + runName
        data['warpedTestDataDir'] = '../Data/kitti/test_itr_'+str(data['numTuple'])+'_tpl/'+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
####################################################################################
####################################################################################
####################################################################################
####################################################################################

def recompile_json_files(runName, itrNum):
    successItr = write_iterative(runName, itrNum)
    if successItr:
        print("JSON files updated")
    return successItr
