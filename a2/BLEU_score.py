import math

def BLEU_score(candidate, references, n, brevity=False):
    """
    Calculate the BLEU score given a candidate sentence (string) and a list of reference sentences (list of strings). n specifies the level to calculate.
    n=1 unigram
    n=2 bigram
    ... and so on
    
    DO NOT concatenate the measurments. N=2 means only bigram. Do not average/incorporate the uni-gram scores.
    
    INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.

	
	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""
	
	#TODO: Implement by student.
    #We only calculate p_n in this function.
    #@431, tie will not be tested. just find the closest
    # We assume it will start from
    lsCandidate_tokens = candidate.split()[1:-1]
    iCandidate_tokens_len = len(lsCandidate_tokens)
    iNum_ngrams = iCandidate_tokens_len - n + 1
    sReal_ref = [ref.split()[1:-1] for ref in references]
    iCi = 0
    for i in range(iNum_ngrams):
        sNgrams = ' '.join(lsCandidate_tokens[i : i + n])
        bExisted_in_ref = False
        for eachref in sReal_ref:
            #Clear the SENTSTART AND SENTEND, NOTICE, we assume it exists!!!
            
            if sNgrams in " ".join(eachref):
                iCi += 1
                break
    fP_n = (iCi * 1.0) / iNum_ngrams

    #Brevity for this only:
    if n == 1 and brevity == True:
        #Find the nearest length
        liLen_ref = [len(sReal_ref[i]) for i in range(len(sReal_ref))]
        #https:stackoverflow.com/questions/9706041
        iR_pos = min(range(len(liLen_ref)), key=lambda i : abs(liLen_ref[i] - iCandidate_tokens_len))
        if liLen_ref[iR_pos] <= iCandidate_tokens_len:#ri<ci
            bleu_score = 1 * fP_n
        else: #exp(1-ri/ci)
            bleu_score = math.exp(1 - liLen_ref[iR_pos] / iCandidate_tokens_len) * fP_n
    else:
        bleu_score = fP_n
    return bleu_score
