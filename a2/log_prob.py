from preprocess import *
from lm_train import *
from math import log


# also, should this be positive or negative log prob?

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
    #         we can get the probability of a sentence simply by multiplying all the probabilities together . decompose using chain rule of probability and markov assumption

    # implement the vanilla log prob first:
    tokens = sentence.split()

    log_prob = 0
    prev_word = None

    delta_add = 0
    if smoothing:
        delta_add = delta


    for index, token in enumerate(tokens):
        unigram_count_curr = LM["uni"][token]



        if index != 0:
            # we should just skip the very first START OF SENTENCE word
            # log_prob += unigram_count_curr/vocabSize # would we also be able to compute vocabSize?


        # ok, now we add laplace smoothing

        # we are guarateed that the prev_word is set now

            # we only need one check, since it is impossible for LM-uni to contain it but LM bi to not contain it
            # what happens if token is not in the bigram tho?
            if (prev_word in LM["uni"] and prev_word in LM["bi"]) and token in LM["bi"][prev_word]:

                #  we can simply modify the laplace count here...
                # (we could also just recompute the terms entirely)
                # we could have a delta multiplier right on the very end
                unigram_count_prev = LM["uni"][prev_word] + delta_add * vocabSize
                bigram_count = LM["bi"][prev_word][token] + delta_add
                #what happens if token is not in the bigram tho?


                log_prob += log(bigram_count/unigram_count_prev,2)


            else:
                # at least one of the quantities in the division is 0,..
                # we migth need to consider all the cases
                # if only the count (w1) is 0
                # if only the count (w2,w1) is 0
                if smoothing:
                    nc_unigram_count_prev =  LM["uni"][prev_word] if prev_word in LM["uni"] else 0
                    nc_bigram_count_prev = LM["bi"][prev_word][token] if token in LM["bi"][prev_word] else 0
                    unigram_count_prev = nc_unigram_count_prev  + delta_add * vocabSize
                    bigram_count = nc_bigram_count_prev  + delta_add # we could also use the previous stuff as well


                    log_prob += log(bigram_count / unigram_count_prev, 2) # we could move this outside

                else:


                    log_prob = float("-inf")
                    break # exit the loop

        prev_word = token

    #     stil need to impleemnt: logs (base2)
    #  as well as smoothing
    # as well as we can need to account for words taht might not be there!

    return log_prob

