import numpy as np

def get_cdf(rngs, prob, maxRange):
    # get continuous cumulative distribution function as ranges differ 
    cdf = 0
    for bin in range(0,maxRange):
        cdf += (rngs[bin+1]-rngs[bin])*prob[bin]
    return cdf

def get_exp_sum(rngs, scores, maxRange):
    # calculates sum of exp(score_i) 
    csf = 0
    for bin in range(0,maxRange):
        csf += np.exp(scores[bin])
    return csf

def get_sqr_sum(rngs, scores, maxRange):
    # calculates sum of square(score_i) 
    csf = 0
    for bin in range(0,maxRange):
        csf += np.math.pow(scores[bin],2)
    return csf

def get_range_weighted_scores(rngs, scores, maxRange):
    # get range wiehgted scores (1/rngs_i)*score_i
    for bin in range(0,maxRange):
        scores[bin] = (1/(rngs[bin+1]-rngs[bin]))*scores[bin]
    # normalize between 0 and 1
    scores = (scores-np.min(scores)) / (np.max(scores)-np.min(scores))
    return scores

def get_softmax_ranging(rngs, scoresP, maxRange):
    # get softmax normalization  ===== vasucakkt exponential weighting
    scoresPWeighted = get_range_weighted_scores(rngs, scoresP, maxRange)
    #scoresPWeighted = scoresPWeighted - np.max(scoresPWeighted)
    sumExp = get_exp_sum(rngs, scoresPWeighted, maxRange) # finds sum of scores w.r.t ranges
    prob = scoresPWeighted.copy()
    for bin in range(0,maxRange): # converts scores to Softmax normalized vector
        prob[bin] = np.exp(scoresPWeighted[bin]) / sumExp
    sumProb = get_cdf(rngs, prob, maxRange) # calculates sum again that will always should be 1
    return prob, sumProb

def get_sqr_ranging(rngs, scoresP, maxRange):
    # get squared normalization  ===== squared norm weighting
    scoresPWeighted = get_range_weighted_scores(rngs, scoresP, maxRange)
    #scoresPWeighted = scoresPWeighted - np.max(scoresPWeighted)
    sumSqr = get_sqr_sum(rngs, scoresPWeighted, maxRange) # finds sum of scores w.r.t ranges
    prob = scoresPWeighted.copy()
    for bin in range(0,maxRange): # converts scores to square normalized vector
        prob[bin] = np.math.pow(scoresPWeighted[bin],2) / sumSqr
    sumProb = get_cdf(rngs, prob, maxRange) # calculates sum again that will always should be 1
    return prob, sumProb

def get_new_ranges(rngsIn, targetP, maxRange, weighting):
    '''
    Get new ranges based on scores using weighted softmax probabilities, so that new ranges have uniform probabilities
    Parameters:
        rngs = current ranges to be updated [n, m+1]
        targetP = predicted logits of each range from network, to be converted to probabilities [n, m]
        maxRange = maximum number of bins for each parameter
        weighting = 'softmax' | 'squared' 
                        # Softmax has a very heavy weighting around the guess region
                        # Squared has lighter weighting than Softmax around the guess region

    '''
    rngs = rngsIn.copy()
    # Convert network predicted logits to probabilities
    if weighting == 'softmax':
        # Softmax has a very heavy weighting around the guess region
        prob, sumWeightedProb = get_softmax_ranging(rngs, targetP, maxRange)
    if weighting == 'squared':
        # Squared has lighter weighting than Softmax around the guess region
        prob, sumWeightedProb = get_sqr_ranging(rngs, targetP, maxRange)
    
    probEach = sumWeightedProb/maxRange
    #print("each =", probEach)
    rngsOld = rngs.copy()
    probLocal = 0
    probRange = 0
    iOld = 0
    jNew = 1
    while jNew<maxRange:
        probLocal = (rngsOld[iOld+1]-rngsOld[iOld])*prob[iOld]
        #print("cLoc", probLocal)
        if probRange+probLocal < probEach: # if sumProb to this point smaller than probEach then add up and continue
            probRange+=probLocal
            iOld+=1
        else: # if sumProb > probEach then this is the iOld -> find cut, and keep rest for next jNew
            probRem = probEach-probRange # means -->  probRem < probLocal
            # ratio of distance we need from this prob region
            #      (ratio*range) this part is implicit, check probLocal definition + left range            
            hRng = (probRem/prob[iOld])+rngsOld[iOld]
            #print("hRng", hRng)
            #print("curRange=",probRange+(hRng*prob[iOld]))
            rngs[jNew] = hRng
            jNew+=1
            rngsOld[iOld] = hRng
            probRange = 0
            #print("rngs=",rngs[0:jNew])
            #print("rOld=",rngsOld[0:iOld])
            #print("prob=",prob[0:iOld])
    return rngs