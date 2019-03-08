from preprocess import *
from lm_train import *
from math import log


def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
    Compute the LOG probability of a sentencfrom preprocess import *
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
	# make sure LM has keys: uni and bi
	if 'uni' not in LM.keys():
		print("ERROR: LM contains NO key: uni.")
		return
	if 'bi' not in LM.keys():
		print("ERROR: LM contains NO key: bi.")
		return

	# make sure delta in range of [0, 1]
	if delta < 0 or delta > 1:
		print("ERROR: delta given is NOT in the range of [0, 1].")
		return

	# obtain tokens (words) list
	tokens = sentence.split()
	log_prob = 0

	# loop over tokens list, and compute the log probability for each token, then
	# sum up to get log probability of a sentence
	for i in range(1, len(tokens)):
		cur_token = tokens[i]
		prev_token = tokens[i-1]

		# define numerator
		if (prev_token in LM['bi'].keys()) and (cur_token in LM['bi'][prev_token].keys()):
			num_count = LM['bi'][prev_token][cur_token]
		else:
			num_count = 0

		# define denominator
		if prev_token in LM['uni'].keys():
			den_count = LM['uni'][prev_token]
		else:
			den_count = 0

		# if numerator or denominator is 0, then log_prob stays as 0
		if (den_count != 0) and (num_count !=0):
			# check whether add smoothing and use log base equals to 2
			if smoothing == False:
				log_prob += log(num_count / float(den_count), 2)
			else:
				num = num_count + delta
				den = den_count + (delta * vocabSize)
				log_prob += log(num / float(den), 2)
		else:
			return float("-inf")
	  
	return log_prob

# if __name__ == "__main__":
# 	sentence = "SENTSTART this is a test case ! SENTEND"
# 	fn_LM = 'LM_E'

# 	with open(fn_LM+'.pickle', 'rb') as handle:
# 		LM = pickle.load(handle)

# 	smoothing = False
# 	delta = 0
# 	vocabSize = len(LM["uni"])
# 	log_prob = log_prob(sentence, LM, smoothing, delta, vocabSize)
# 	print(log_prob)e, given a language model and whether or not to
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

    # TODO: Implement by student.
    # make sure LM has keys: uni and bi
    if 'uni' not in LM.keys():
        print("ERROR: LM contains NO key: uni.")
        return
    if 'bi' not in LM.keys():
        print("ERROR: LM contains NO key: bi.")
        return
    
    # make sure delta in range of [0, 1]
    if delta < 0 or delta > 1:
        print("ERROR: delta given is NOT in the range of [0, 1].")
        return
    
    # obtain tokens (words) list
    tokens = sentence.split()
    log_prob = 0
    
    # loop over tokens list, and compute the log probability for each token, then
    # sum up to get log probability of a sentence
    for i in range(1, len(tokens)):
        cur_token = tokens[i]
        prev_token = tokens[i - 1]
        
        # define numerator
        if (prev_token in LM['bi'].keys()) and (cur_token in LM['bi'][prev_token].keys()):
            num_count = LM['bi'][prev_token][cur_token]
        else:
            num_count = 0
        
        # define denominator
        if prev_token in LM['uni'].keys():
            den_count = LM['uni'][prev_token]
        else:
            den_count = 0
        
        # if numerator or denominator is 0, then log_prob stays as 0
        if (den_count != 0) and (num_count != 0):
            # check whether add smoothing and use log base equals to 2
            if smoothing == False:
                log_prob += log(num_count / float(den_count), 2)
            else:
                num = num_count + delta
                den = den_count + (delta * vocabSize)
                log_prob += log(num / float(den), 2)
        else:
            return float("-inf")
    
    return log_prob

# if __name__ == "__main__":
# 	sentence = "SENTSTART this is a test case ! SENTEND"
# 	fn_LM = 'LM_E'

# 	with open(fn_LM+'.pickle', 'rb') as handle:
# 		LM = pickle.load(handle)

# 	smoothing = False
# 	delta = 0
# 	vocabSize = len(LM["uni"])
# 	log_prob = log_prob(sentence, LM, smoothing, delta, vocabSize)
# 	print(log_prob)