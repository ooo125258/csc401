from preprocess import *
from lm_train import *
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary
	
	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""
	
	#TODO: Implement by student.
    if 'uni' not in LM and 'bi' not in LM:
        print("log_prob: warning: the result is incompleted")

    if not smoothing:
        delta = 0
        vocabSize = 0
    log_prob = 0
    lsTokens = sentence.split()
    nLen_tokens = len(lsTokens)
    
    for i in range(1, nLen_tokens):
        iNumo = delta
        iDeno = delta * vocabSize
        flLog_oneword = 0
        if lsTokens[i - 1] in LM['bi'] and lsTokens[i] in LM['bi'][lsTokens[i - 1]]:
            iNumo = LM['bi'][lsTokens[i - 1]][lsTokens[i]] + delta
        if lsTokens[i - 1] in LM['uni']:
            iDeno = LM['uni'][lsTokens[i - 1]] + delta * vocabSize
        if iNumo == 0 or iDeno == 0:
            flLog_oneword = float('-inf')
            log_prob += flLog_oneword
            break
        else:
            flLog_oneword = log(iNumo, 2) - log(iDeno, 2)
            log_prob += flLog_oneword
    return log_prob
