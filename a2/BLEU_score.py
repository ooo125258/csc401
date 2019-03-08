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
    lsCandidate_tokens = candidate.split()
    iCandidate_tokens_len = len(lsCandidate_tokens)
    iNum_ngrams = iCandidate_tokens_len - n + 1
    sReal_ref = [ref.split() for ref in references]
    iCi = 0
    for i in range(iNum_ngrams):
        sNgrams = ' '.join(lsCandidate_tokens[i : i + n])
        bExisted_in_ref = False
        for eachref in range(len(references)):
            #Clear the SENTSTART AND SENTEND, NOTICE, we assume it exists!!!
            if n == 1: #The special case: "la" in candidate to search in "declare" in reference
                if sNgrams in sReal_ref[eachref]:
                    iCi += 1
                    break
            else: #All others
                if sNgrams in references[eachref]:
                    iCi += 1
                    break
    fP_n = (iCi * 1.0) / iNum_ngrams

                                                                                                                                #Brevity for this only:
    if n == 1 and brevity:
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
    
if __name__ == '__main__':
    candidate = "It is a guide to action which ensures that the military always obeys the commands of the party"
    references = ["It is a guide to action that ensures that the military will forever heed Party commands", "It is the guiding principle which guarantees the military forces always being under command of the Party", "It is the practical guide for the army always to heed the directions of the party"]
    n = 1
    BLEU_score(candidate, references, n, brevity=False)
