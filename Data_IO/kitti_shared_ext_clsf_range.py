
def get_cdf(rng, prob, maxRange):
    # get continuous cumulative distribution function as ranges differ 
    cdf = 0
    for bin in range(0,maxRange):
        cdf += (rngs[bin+1]-rngs[bin])*prob[bin]
    return cdf

def get_exp_sum(rngs, scores, maxRange):
    # calculates sum of exp(socre_i) 
    csf = 0
    for bin in range(0,maxRange):
        csf += np.exp(scores[bin])
    return csf

def get_range_weighted_scores(rngs, scores, maxRange):
    # get range wiehgted scores (1/rngs_i)*score_i
    for bin in range(0,maxRange):
        scores[bin] = (1/(rngs[bin+1]-rngs[bin]))*scores[bin]
    return scores

def get_softmax(rngs, prob, maxRange):
    # get softmax normalization  ===== vasucakkt exponential weighting
    prob = get_range_weighted_scores(rngs, prob, maxRange)
    prob = prob - np.max(prob)
    sumExp = get_exp_sum(rngs, prob, maxRange) # finds sum of scores w.r.t ranges
    for bin in range(0,maxRange): # converts scores to Softmax normalized vector
        prob[bin] = np.exp(prob[bin]) / sumExp
    sumProb = get_cdf(rngs, prob, maxRange) # calculates sum again that will always should be 1
    return prob, sumProb

def get_new_ranges(rngs, prob, maxRange):
    # get new ranges based on probabilites, so that new ranges have uniform probabilities
    prob, sumWeightedProb = get_softmax(rngs, prob, maxRange)
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
        if probRange+probLocal < probEach:
            probRange+=probLocal
            iOld+=1
        else:
            probRem = probEach-probRange
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