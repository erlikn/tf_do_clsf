from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

def add_loss_summaries(total_loss, batchSize):
    """Add summaries for losses in calusa_heatmap model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='Average')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Individual average loss
#    lossPixelIndividual = tf.sqrt(tf.multiply(total_loss, 2/(batchSize*4))) # dvidied by (8/2) = 4 which is equal to sum of 2 of them then sqrt will result in euclidean pixel error
#    tf.summary.scalar('Average_Pixel_Error_Real', lossPixelIndividual)

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + '_raw', l)
        tf.summary.scalar(l.op.name + '_average', loss_averages.average(l))

    return loss_averages_op

def _l2_loss(pred, tval): # batchSize=Sne
    """Add L2Loss to all the trainable variables.
    
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size, heatmap_size ]
    
    Returns:
      Loss tensor of type float.
    """
    l2_loss = tf.nn.l2_loss(tf.subtract(pred, tval), name="loss_l2")
    #tf.add_to_collection('losses', l2_loss)
    #return tf.add_n(tf.get_collection('losses'), name='loss_total')
    return l2_loss


def _weighted_L2_loss(tMatP, tMatT, activeBatchSize):
    mask = np.array([[100, 100, 100, 1, 100, 100, 100, 1, 100, 100, 100, 1]], dtype=np.float32)
    mask = np.repeat(mask, activeBatchSize, axis=0)
    tMatP = tf.multiply(mask, tMatP)
    tMatT = tf.multiply(mask, tMatT)
    return _l2_loss(tMatP, tMatT)

def _weighted_params_L2_loss(targetP, targetT, activeBatchSize):
    # Alpha, Beta, Gamma are -Pi to Pi periodic radians - mod over pi to remove periodicity
    #mask = np.array([[np.pi, np.pi, np.pi, 1, 1, 1]], dtype=np.float32)
    #mask = np.repeat(mask, kwargs.get('activeBatchSize'), axis=0)
    #targetP = tf_mod(targetP, mask)
    #targetT = tf_mod(targetT, mask)
    # Importance weigting on angles as they have smaller values
    mask = np.array([[100, 100, 100, 1, 1, 1]], dtype=np.float32)
    #mask = np.array([[1000, 1000, 1000, 100, 100, 100]], dtype=np.float32)
    mask = np.repeat(mask, activeBatchSize, axis=0)
    targetP = tf.multiply(targetP, mask)
    targetT = tf.multiply(targetT, mask)
    return _l2_loss(targetP, targetT)

def _weighted_params_L2_loss_nTuple_last(targetP, targetT, nTuple, activeBatchSize):
    # Alpha, Beta, Gamma are -Pi to Pi periodic radians - mod over pi to remove periodicity
    #mask = np.array([[np.pi, np.pi, np.pi, 1, 1, 1]], dtype=np.float32)
    #mask = np.repeat(mask, kwargs.get('activeBatchSize'), axis=0)
    #targetP = tf_mod(targetP, mask)
    #targetT = tf_mod(targetT, mask)
    # Importance weigting on angles as they have smaller values
    mask = np.array([[100, 100, 100, 1, 1, 1]], dtype=np.float32)
    mask = np.repeat(mask, activeBatchSize, axis=0)
    targetP = tf.multiply(targetP, mask)
    targetT = tf.multiply(targetT, mask)
    return _l2_loss(targetP, targetT)

def _weighted_params_L2_loss_nTuple_all(targetP, targetT, nTuple, activeBatchSize):
    # Alpha, Beta, Gamma are -Pi to Pi periodic radians - mod over pi to remove periodicity
    #mask = np.array([[np.pi, np.pi, np.pi, 1, 1, 1]], dtype=np.float32)
    #mask = np.repeat(mask, kwargs.get('activeBatchSize'), axis=0)
    #targetP = tf_mod(targetP, mask)
    #targetT = tf_mod(targetT, mask)
    # Importance weigting on angles as they have smaller values
    mask = np.array([[100, 100, 100, 1, 1, 1]], dtype=np.float32)
    mask = np.repeat(mask, nTuple-1, axis=0).reshape((nTuple-1)*6)
    mask = np.repeat(mask, activeBatchSize, axis=0)
    targetP = tf.multiply(targetP, mask)
    targetT = tf.multiply(targetT, mask)
    return _l2_loss(targetP, targetT)

def _params_classification_l2_loss_nTuple(targetP, targetT, nTuple, activeBatchSize):
    '''
    TargetT dimensions are [activeBatchSize, rows=6, cols=32, nTuple]
    '''
    # Alpha, Beta, Gamma are -Pi to Pi periodic radians - mod over pi to remove periodicity
    #mask = np.array([[np.pi, np.pi, np.pi, 1, 1, 1]], dtype=np.float32)
    #mask = np.repeat(mask, kwargs.get('activeBatchSize'), axis=0)
    #targetP = tf_mod(targetP, mask)
    #targetT = tf_mod(targetT, mask)
    # Importance weigting on angles as they have smaller values
    #mask = np.array([[100, 100, 100, 1, 1, 1]], dtype=np.float32)
    #mask = np.repeat(mask, nTuple-1, axis=0).reshape((nTuple-1)*6)
    #mask = np.repeat(mask, activeBatchSize, axis=0)
    #targetP = tf.multiply(targetP, mask)
    #targetT = tf.multiply(targetT, mask)
    targetT = tf.cast(targetT, tf.float32)
    return _l2_loss(targetP, targetT)

def _params_classification_softmaxCrossentropy_loss_nTuple(targetP, targetT, nTuple, activeBatchSize):
    '''
    Takes in the targetP and targetT and calculates the softmax-cross entropy loss for each parameter
    and sums them for each instance and sum again for each tuple in the batch
    TargetT dimensions are [activeBatchSize, rows=6, cols=32, nTuple]
    '''
    targetT = tf.cast(targetT, tf.float32)
    targetP = tf.cast(targetP, tf.float32)
    ############################
    # Alternatively, instead of sum, we could use squared_sum to penalize harsher
    ############################
    # ---> [activeBatchSize, rows=6, cols=32, nTuple]
    # Calculate softmax-cross entropy loss for each parameter (cols dimension -> cols)
    # ---> [activeBatchSize, rows=6, nTuple]
    # Then calculate sum of parameter losses for each batch (last 2 dimensions -> ntuple, rows), and returns an array of [activeBatchSize] size
    # ---> [activeBatchSize]
    smce_loss = tf.nn.l2_loss(tf.nn.softmax_cross_entropy_with_logits(logits=targetP, labels=targetT, dim=2), name="loss_smce_l2")
    #smce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=targetP, labels=targetT, dim=2), name="loss_smce_sum")
    return smce_loss

def _params_classification_gaussian_softmaxCrossentropy_loss_nTuple(targetP, targetT, nTuple, activeBatchSize):
    '''
    Takes in the targetP and targetT and calculates the softmax-cross entropy loss for each parameter
    and sums them for each instance and sum again for each tuple in the batch
    TargetT dimensions are [activeBatchSize, rows=6, cols=32, nTuple]
    '''
    targetT = tf.cast(targetT, tf.float32)
    targetP = tf.cast(targetP, tf.float32)
    ############################
    # find argmax of the target
    # find argmax of the loss
    # apply a gaussian based on index diff based on target and loss
    ############################
    softmaxLoss = tf.nn.softmax_cross_entropy_with_logits(logits=targetP, labels=targetT, dim=2)
    ### location sensetive classing, (distance^2)/5 ===> 5 means vicinity of index
    locationLoss = tf.multiply(tf.cast(tf.multiply((tf.argmax(targetP, axis=2)-tf.argmax(targetT, axis=2)),(tf.argmax(targetP, axis=2)-tf.argmax(targetT, axis=2))),tf.float32),(1/5))
    ### weight softmaxloss by location loss and get the l2 loss of batches
    smce_loss = tf.nn.l2_loss(tf.multiply(softmaxLoss, locationLoss), name="loss_glsmce_l2")
    #smce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=targetP, labels=targetT, dim=2), name="loss_smce_sum")
    return smce_loss

def _transformation_loss_nTuple_last(targetP, targetT, prevPred, nTuple, activeBatchSize):
    '''
    targetP = [activeBatchSize x 12]
    targetT = [activeBatchSize x 12]
    '''
    # 12 transformation matrix
    # Reshape each array to 3x4 and append [0,0,0,1] to each transformation matrix
    pad = tf.reshape( tf.tile( tf.constant([0, 0, 0, 1], dtype=tf.float32), [activeBatchSize]), [activeBatchSize, 4]) # [activeBatchSize x 4]
    targetP = tf.reshape(tf.concat([tf.reshape(targetP,[activeBatchSize, 12]),pad],1), [activeBatchSize, 4, 4])# [activeBatchSize x 12] -> [activeBatchSize x 16] -> [activeBatchSize x 4 x 4]
    targetT = tf.reshape(tf.concat([tf.reshape(targetT,[activeBatchSize, 12]),pad],1), [activeBatchSize, 4, 4])# [activeBatchSize x 12] -> [activeBatchSize x 16] -> [activeBatchSize x 4 x 4]
    #prevPred = tf.reshape(tf.concat([tf.reshape(prevPred,[activeBatchSize, 12]),pad],1), [activeBatchSize, 4, 4])# [activeBatchSize x 12] -> [activeBatchSize x 16] -> [activeBatchSize x 4 x 4]
    # Initialize points : 5 points that don't lie in a plane
    # [activeBatchSize x 4 x 5]
    points = tf.reshape( tf.tile(tf.constant([0, 1000,   0, 500, -230,
                                              0, 1000, 100, 500,   33,
                                              0, 1000,   0,   0,  672,
                                              1,    1,   1,   1,   1], dtype=tf.float32),
                                [activeBatchSize]), [activeBatchSize, 4, 5]) 
    # First get the source to current transformation: pPred x pCurrent
    #targetP = tf.matmul(prevPred, targetP) # [activeBatchSize x 4 x 4] * [activeBatchSize x 4 x 4] = [activeBatchSize x 4 x 4]
    # Transform points based on prediction
    pPoints = tf.matmul(targetP, points) # [activeBatchSize x 4 x 4] * [activeBatchSize x 4 x 5] = [activeBatchSize x 4 x 5]
    # Transform points based on target
    tPoints = tf.matmul(targetT, points) # [activeBatchSize x 4 x 4] * [activeBatchSize x 4 x 5] = [activeBatchSize x 4 x 5]
    # Get and return L2 Loss between corresponding points
    return _l2_loss(pPoints, tPoints)

def _params_transformation_loss_nTuple_last(targetP, targetT, prevPred, nTuple, activeBatchSize):
    '''
    Input:
        targetP = [activeBatchSize x 6]
        targetT = [activeBatchSize x 6]
    Description:
        tMatP = [activeBatchSize x 12]
        tmatT = [activeBatchSize x 12]
        use pre-existing _transformation_loss_nTuple_last to calculate the l2 loss
    '''
    # get transformation matrix from 6 parameters
    # get prediction tmat
    tMatP = tf.stack([tf.cos(targetP[:,0])*tf.cos(targetP[:,1]),  
                      (tf.cos(targetP[:,0])*tf.sin(targetP[:,1])*tf.sin(targetP[:,2]))-(tf.sin(targetP[:,0])*tf.cos(targetP[:,2])),
                      (tf.cos(targetP[:,0])*tf.sin(targetP[:,1])*tf.cos(targetP[:,2]))+(tf.sin(targetP[:,0])*tf.sin(targetP[:,2])),
                      targetP[:,3],
                      tf.sin(targetP[:,0])*tf.cos(targetP[:,1]),
                      (tf.sin(targetP[:,0])*tf.sin(targetP[:,1])*tf.sin(targetP[:,2]))+(tf.cos(targetP[:,0])*tf.cos(targetP[:,2])),
                      (tf.sin(targetP[:,0])*tf.sin(targetP[:,1])*tf.cos(targetP[:,2]))-(tf.cos(targetP[:,0])*tf.sin(targetP[:,2])),
                      targetP[:,4],
                      -tf.sin(targetP[:,1]),tf.cos(targetP[:,1])*tf.sin(targetP[:,2]),
                      tf.cos(targetP[:,1])*tf.cos(targetP[:,2]),
                      targetP[:,5]], 
                     axis=1)
    # get prevPred tmat
    tMatPrev = tf.stack([tf.cos(prevPred[:,0])*tf.cos(prevPred[:,1]),  
                      (tf.cos(prevPred[:,0])*tf.sin(prevPred[:,1])*tf.sin(prevPred[:,2]))-(tf.sin(prevPred[:,0])*tf.cos(prevPred[:,2])),
                      (tf.cos(prevPred[:,0])*tf.sin(prevPred[:,1])*tf.cos(prevPred[:,2]))+(tf.sin(prevPred[:,0])*tf.sin(prevPred[:,2])),
                      prevPred[:,3],
                      tf.sin(prevPred[:,0])*tf.cos(prevPred[:,1]),
                      (tf.sin(prevPred[:,0])*tf.sin(prevPred[:,1])*tf.sin(prevPred[:,2]))+(tf.cos(prevPred[:,0])*tf.cos(prevPred[:,2])),
                      (tf.sin(prevPred[:,0])*tf.sin(prevPred[:,1])*tf.cos(prevPred[:,2]))-(tf.cos(prevPred[:,0])*tf.sin(prevPred[:,2])),
                      prevPred[:,4],
                      -tf.sin(prevPred[:,1]),tf.cos(prevPred[:,1])*tf.sin(prevPred[:,2]),
                      tf.cos(prevPred[:,1])*tf.cos(prevPred[:,2]),
                      prevPred[:,5]], 
                     axis=1)
    # get gt tmat
    tMatT = tf.stack([tf.cos(targetT[:,0])*tf.cos(targetT[:,1]),  
                      (tf.cos(targetT[:,0])*tf.sin(targetT[:,1])*tf.sin(targetT[:,2]))-(tf.sin(targetT[:,0])*tf.cos(targetT[:,2])),
                      (tf.cos(targetT[:,0])*tf.sin(targetT[:,1])*tf.cos(targetT[:,2]))+(tf.sin(targetT[:,0])*tf.sin(targetT[:,2])),
                      targetT[:,3],
                      tf.sin(targetT[:,0])*tf.cos(targetT[:,1]),
                      (tf.sin(targetT[:,0])*tf.sin(targetT[:,1])*tf.sin(targetT[:,2]))+(tf.cos(targetT[:,0])*tf.cos(targetT[:,2])),
                      (tf.sin(targetT[:,0])*tf.sin(targetT[:,1])*tf.cos(targetT[:,2]))-(tf.cos(targetT[:,0])*tf.sin(targetT[:,2])),
                      targetT[:,4],
                      -tf.sin(targetT[:,1]),tf.cos(targetT[:,1])*tf.sin(targetT[:,2]),
                      tf.cos(targetT[:,1])*tf.cos(targetT[:,2]),
                      targetT[:,5]], 
                     axis=1)
    return _transformation_loss_nTuple_last(tMatP, tMatT, prevPred, nTuple, activeBatchSize)

def loss(pred, tval, prevPred, **kwargs):
    """
    Choose the proper loss function and call it.
    """
    lossFunction = kwargs.get('lossFunction')
    if lossFunction == 'L2':
        return _l2_loss(pred, tval)
    if lossFunction == 'Weighted_L2_loss':
        return _weighted_L2_loss(pred, tval, kwargs.get('activeBatchSize'))
    if lossFunction == 'Weighted_Params_L2_loss':
        return _weighted_params_L2_loss(pred, tval, kwargs.get('activeBatchSize'))
    if lossFunction == 'Weighted_Params_L2_loss_nTuple_last':
        return _weighted_params_L2_loss_nTuple_last(pred, tval, kwargs.get('numTuple'), kwargs.get('activeBatchSize'))
    if lossFunction == 'Weighted_Params_L2_loss_nTuple_all':
        return _weighted_params_L2_loss_nTuple_all(pred, tval, kwargs.get('numTuple'), kwargs.get('activeBatchSize'))
    if lossFunction == '_params_classification_l2_loss_nTuple':
        return _params_classification_l2_loss_nTuple(pred, tval, kwargs.get('numTuple'), kwargs.get('activeBatchSize'))
    if lossFunction == '_params_classification_softmaxCrossentropy_loss_nTuple':
        if kwargs.get('lastTuple'):
            return _params_classification_softmaxCrossentropy_loss_nTuple(pred, tval, 1, kwargs.get('activeBatchSize'))
        else:
            return _params_classification_softmaxCrossentropy_loss_nTuple(pred, tval, kwargs.get('numTuple'), kwargs.get('activeBatchSize')) # not complete
    if lossFunction == '_params_classification_gaussian_softmaxCrossentropy_loss_nTuple':
        if kwargs.get('lastTuple'):
            return _params_classification_gaussian_softmaxCrossentropy_loss_nTuple(pred, tval, 1, kwargs.get('activeBatchSize'))
        else:
            return _params_classification_gaussian_softmaxCrossentropy_loss_nTuple(pred, tval, kwargs.get('numTuple'), kwargs.get('activeBatchSize')) # not complete
    if lossFunction == '_transformation_loss_nTuple_last':
        if kwargs.get('lastTuple'):
            if prevPred is 0:
                return None
            return _transformation_loss_nTuple_last(pred, tval, prevPred, 1, kwargs.get('activeBatchSize'))
        else:
            return _transformation_loss_nTuple_last(pred, tval, prevPred, kwargs.get('numTuple'), kwargs.get('activeBatchSize')) # not complete
    if lossFunction == '_params_transformation_loss_nTuple_last':
        if kwargs.get('lastTuple'):
            if prevPred is 0:
                return None
            return _params_transformation_loss_nTuple_last(pred, tval, prevPred, 1, kwargs.get('activeBatchSize'))
        else:
            return _params_transformation_loss_nTuple_last(pred, tval, prevPred, kwargs.get('numTuple'), kwargs.get('activeBatchSize')) # not complete

