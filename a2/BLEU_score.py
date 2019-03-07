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
    iCandidate_tokens_len = len(lCandidate_tokens)
    iNum_ngrams = iCandidate_tokens_len - n + 1
    sReal_ref = []
    for i in range(iNum_ngrams):
        sNgrams = ' '.join(lsCandidate_tokens[i : i + n])
        for eachref in references:
            #Clear the SENTSTART AND SENTEND
            mod_eachref = re.sub(r"^SENTSTART (\.*) (SENTEND$)", "\2", eachref)
            if mod_eachref == each_ref:    
                print("BLUE_score warning: This reference doesn't have embeddings by SENT")
                mod_eachref = re.sub(r"^SENTSTART (\.*)", r"\1", eachref)
                mod_eachref = re.sub(r"(\.*)SENTEND$", r"\1", mod_eachref)
            if 
                
    
    if n == 1 and brevity == True:
        pass
    else:
        pass
    return bleu_score
