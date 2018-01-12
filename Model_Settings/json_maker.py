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


baseTrainDataDir = '../Data/kitti/train_tfrecords_clsf'
baseTestDataDir = '../Data/kitti/test_tfrecords_clsf'

# Twin Common Parameters
trainLogDirBase = '../Data/kitti/logs/tfdh_twin_py_logs/train_logs/'
testLogDirBase = '../Data/kitti/logs/tfdh_twin_py_logs/test_logs/'
warpedTrainDirBase = '../Data/kitti/train_tfrecords_iterative/'
warpedTestDirBase = '../Data/kitti/test_tfrecords_iterative/'

data = {
    # Data Parameters
    'numTrainDatasetExamples' : 20400,
    'numTestDatasetExamples' : 2790,
    'trainDataDir' : '../Data/kitti/train_tfrecords',
    'testDataDir' : '../Data/kitti/test_tfrecords',
    'warpedTrainDataDir' : warpedTrainDirBase+'',
    'warpedTestDataDir' : warpedTestDirBase+'',
    'trainLogDir' : trainLogDirBase+'',
    'testLogDir' : testLogDirBase+'',
    'tMatTrainDir' : trainLogDirBase+'/target',
    'tMatTestDir' : testLogDirBase+'/target',
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
    'classification': {'Model' : False, 'binSize' : 0}
    }
data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))

####################################################################################
####################################################################################
####################################################################################
reCompileJSON = True
####################################################################################
####################################################################################
####################################################################################

def write_iterative(runName, itrNum, dataLocal):
    # Twin Correlation Matching Common Parameters
    trainLogDirBase = '../Data/kitti/logs/tfdh_iter_clsf_logs/train_logs/'
    testLogDirBase = '../Data/kitti/logs/tfdh_iter_clsf_logs/test_logs/'

    dataLocal['writeWarpedImages'] = True

    # Iterative model only changes the wayoutput is written, 
    # so any model can be used by ease

    reCompileITR = True
    NOreCompileITR = False

    if runName == '170706_ITR_B':
        itr_170706_ITR_B_inception(reCompileITR, trainLogDirBase, testLogDirBase, runName, itrNum, dataLocal)
    elif runName == '171003_ITR_B': # using 170706_ITR_B but with loss for all n-1 tuples
        data['classificationModel'] = {'Model' : True, 'binSize' : 32}
        itr_171003_ITR_B_clsf(reCompileITR, trainLogDirBase, testLogDirBase, runName, itrNum, dataLocal)
    elif runName == '180110_ITR_B': # using 171003_ITR_B but with softmax loss for all 2 tuples
        data['classificationModel'] = {'Model' : True, 'binSize' : 32}
        itr_180110_ITR_B_clsf(reCompileITR, trainLogDirBase, testLogDirBase, runName, itrNum, dataLocal)
    elif runName == '180111_ITR_B': # using 171003_ITR_B but with softmax loss for all 5 tuples
        data['classificationModel'] = {'Model' : True, 'binSize' : 32}
        itr_180110_ITR_B_clsf(reCompileITR, trainLogDirBase, testLogDirBase, runName, itrNum, dataLocal)
    else:
        print("--error: Model name not found!")
        return False
    return True
    ##############
    ##############
    ##############

def itr_170706_ITR_B_inception(reCompileITR, trainLogDirBase, testLogDirBase, runName, itrNum, data):
    if reCompileITR:
        runPrefix = runName+'_'
        data['modelName'] = 'twin_cnn_4p4l2f_inception'
        data['numParallelModules'] = 2
        data['imageDepthChannels'] = 2
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['trainBatchSize'] = 32
        data['testBatchSize'] = 32
        data['numTrainDatasetExamples'] = 20400
        data['numTestDatasetExamples'] = 2790
        data['logicalOutputSize'] = 6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "Weighted_Params_L2_loss_nTuple_last"
        data['numTuple'] = 2
        
        runName = runPrefix+str(itrNum)
        ### Auto Iteration Number
        if itrNum == 1:
            data['trainDataDir'] = baseTrainDataDir
            data['testDataDir'] = baseTestDataDir
        ### Auto Iteration Number 2,3,4
        if itrNum > 1:
            data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
            data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        ####
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
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

def itr_171003_ITR_B_clsf(reCompileITR, trainLogDirBase, testLogDirBase, runName, itrNum, data):
    if reCompileITR:
        runPrefix = runName+'_'
        data['modelName'] = 'twin_cnn_4p4l2f_inception'
        data['numParallelModules'] = 2
        data['imageDepthChannels'] = 2
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['trainBatchSize'] = 2#8#16
        data['testBatchSize'] = 2#8#16
        data['numTrainDatasetExamples'] = 20400
        data['numTestDatasetExamples'] = 2790
        data['logicalOutputSize'] = 6
        data['networkOutputSize'] = data['logicalOutputSize']*data['classificationModel']['binSize']
        data['lossFunction'] = "_params_classification_l2_loss_nTuple"
        data['numTuple'] = 2
        
        runName = runPrefix+str(itrNum)
        ### Auto Iteration Number
        if itrNum == 1:
            data['trainDataDir'] = baseTrainDataDir
            data['testDataDir'] = baseTestDataDir
        ### Auto Iteration Number 2,3,4
        if itrNum > 1:
            data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
            data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        ####
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
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
    
def itr_180110_ITR_B_clsf(reCompileITR, trainLogDirBase, testLogDirBase, runName, itrNum, data):
    if reCompileITR:
        runPrefix = runName+'_'
        data['modelName'] = 'twin_cnn_4p4l2f_inception'
        data['numParallelModules'] = 2
        data['imageDepthChannels'] = 2
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['trainBatchSize'] = 2#8#16
        data['testBatchSize'] = 2#8#16
        data['numTrainDatasetExamples'] = 20400
        data['numTestDatasetExamples'] = 2790
        data['logicalOutputSize'] = 6
        data['networkOutputSize'] = data['logicalOutputSize']*data['classificationModel']['binSize']
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss_nTuple"
        data['numTuple'] = 2
        
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
            data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
            data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        ####
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
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

def itr_180111_ITR_B_clsf(reCompileITR, trainLogDirBase, testLogDirBase, runName, itrNum, data):
    if reCompileITR:
        runPrefix = runName+'_'
        data['modelName'] = 'twin_cnn_4p4l2f_inception'
        data['numParallelModules'] = 2
        data['imageDepthChannels'] = 2
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['trainBatchSize'] = 2#8#16
        data['testBatchSize'] = 2#8#16
        data['numTrainDatasetExamples'] = 20400
        data['numTestDatasetExamples'] = 2790
        data['logicalOutputSize'] = 6
        data['networkOutputSize'] = data['logicalOutputSize']*data['classificationModel']['binSize']
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss_nTuple"
        data['numTuple'] = 5
        
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
            data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
            data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        ####
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
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
####################################################################################
####################################################################################
####################################################################################
####################################################################################

def recompile_json_files(runName, itrNum):
    successItr = write_iterative(runName, itrNum, data)
    if successItr:
        print("JSON files updated")
    return successItr
