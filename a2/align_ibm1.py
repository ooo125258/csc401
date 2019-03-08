# from lm_train import *
# from log_prob import *
from preprocess import *
from math import log
import os
import pickle


def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model
	
	OUTPUT:
	AM :			(dictionary) alignment model structure
	
	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.
	
			LM['house']['maison'] = 0.5
	"""
    AM = {}
    
    # Read training data
    training_data = read_hansard(train_dir, num_sentences)
    
    # Initialize AM uniformly
    # initialize P(f | e)
    AM = initialize(training_data[0], training_data[1])
    
    # Iterate between E and M steps
    # for a number of iterations:
    temp_AM = AM
    for i in range(max_iter):
        temp_AM = em_step(temp_AM, training_data[0], training_data[1])
    temp_AM["SENTSTART"] = {}
    temp_AM["SENTSTART"]["SENTSTART"] = 1
    temp_AM["SENTEND"] = {}
    temp_AM["SENTEND"]["SENTEND"] = 1
    AM = temp_AM
    
    with open(fn_AM + '.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return AM


# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	
	Return:
	    (eng, fre) when each of them is a list of list of pre-processed eng or fre words in sentences of the train_dir
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	
	Make sure to read the files in an aligned manner.
	"""
    # TODO
    counter = 0
    training_set = {'eng': [], 'fre': []}
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if not (len(file) > 2 and file[-1] == 'e' and file[-2] == '.'):  # .e
                continue
            
            e_fullName = os.path.join(train_dir, file)
            f_fullName = e_fullName[:-1] + 'f'
            if not os.path.exists(f_fullName):
                # To remove eng without fre
                continue
            e_file = open(e_fullName)
            f_file = open(f_fullName)
            
            e_readLine = e_file.readline()
            f_readLine = f_file.readline()
            
            while e_readLine:  # "" is false directly
                if not f_readLine:
                    continue
                training_set['eng'].append(preprocess(e_readLine, 'e').split())
                training_set['fre'].append(preprocess(f_readLine, 'f').split())
                counter += 1
                
                if counter >= num_sentences:
                    # The time is now
                    e_file.close()
                    f_file.close()
                    return training_set['eng'], training_set['fre']
                e_readLine = e_file.readline()
                f_readLine = f_file.readline()
            e_file.close()
            f_file.close()
    return training_set['eng'], training_set['fre']


def initialize(eng, fre):
    '''
    Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
    :param eng: a list of english sentences
    :param fre: a list of french sentences
    :return: dict AM{eng_token:{fre_token:am_value}}
    '''
    
    # TODO
    # check inbalance - although it's impossible
    if len(eng) != len(fre):
        print("Function initialize: \
        unbalanced eng and fre len: {} and {}".format(len(eng), len(fre)))
    
    counting = {}
    AM = {}
    len_used = min(len(eng), len(fre))
    for i in range(len_used):
        
        # Count all relationship from each english word to each french word!
        for j in range(len(eng[i])):
            # remove these two
            if eng[i][j] == "SENTSTART" or eng[i][j] == "SENTEND":
                continue
            if eng[i][j] not in counting:
                counting[eng[i][j]] = {}
            for k in range(len(fre[i])):
                # There is a relation for this english word and all french word in the selected sentence
                if fre[i][k] == "SENTSTART" or fre[i][k] == "SENTEND":
                    continue
                if fre[i][k] not in counting[eng[i][j]]:
                    counting[eng[i][j]][fre[i][k]] = 1
                else:
                    counting[eng[i][j]][fre[i][k]] += 1
    
    for eng_token in counting:
        AM[eng_token] = {}
        length_fre_tokens_for_this_eng_token = len(counting[eng_token])
        if length_fre_tokens_for_this_eng_token == 0:
            # Although I don't think it can be zero, if there is a loop
            p = 0
        else:
            p = 1.0 / length_fre_tokens_for_this_eng_token
        for fre_token in counting[eng_token]:
            AM[eng_token][fre_token] = p
    
    return AM


def em_step(AM, eng, fre):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
    # TODO
    
    tcount = {}
    total = {}
    
    for e in AM:
        # set total(e) to 0 for all e
        total[e] = 0
        tcount[e] = {}
        for f in AM[e]:
            # set tcount(f, e) to 0 for all f, e
            tcount[e][f] = 0
    
    # for each sentence pair (F, E) in training corpus:
    for pair_idx in range(len(eng)):
        # for each unique word f in F:
        f_unique_words = unique_word(fre[pair_idx])
        for f in f_unique_words:
            if f == "SENTSTART" or f == "SENTEND":
                continue
            denom_c = 0
            e_unique_words = unique_word(eng[pair_idx])
            # for each unique word e in E:
            for e in e_unique_words:
                if e == "SENTSTART" or e == "SENTEND":
                    continue
                else:
                    # denom_c += P(f|e) * F.count(f)
                    denom_c += getAMef(AM, e, f) * f_unique_words[f]
            # for each unique word e in E:
            for e in e_unique_words:
                if e == "SENTSTART" or e == "SENTEND":
                    continue
                else:
                    AMef = getAMef(AM, e, f)
                if AMef == 0:
                    continue
                value_added = AMef * f_unique_words[f] * e_unique_words[e] / denom_c
                # tcount(f, e) += P(f|e) * F.count(f) * E.count(e) / denom_c
                tcount[e][f] += value_added
                # total(e) += P(f|e) * F.count(f) * E.count(e) / denom_c
                total[e] += value_added
    # for each e in domain(total(:)):
    for e in total:
        # for each f in domain(tcount(:,e)):
        for f in tcount[e]:
            # P(f|e) = tcount(f, e) / total(e)
            if e not in AM:
                AM[e] = {}
            if total[e] == 0:
                AM[e][f] = 0
            else:
                AM[e][f] = tcount[e][f] / total[e]
    return AM


def unique_word(sentence):
    '''
    Generate the dictionary for tokens in a sentences: the value is the times of appearance
    :param sentence: string. a sentence
    :return: dict count{word: #appearance}
    '''
    tokens = sentence  # .split() for already broken
    # remove duplicate by sets!
    # return list(set(tokens))
    
    # return a dict. The unique words are token and the times of appearance is value
    # https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-12.php
    counts = {}
    for word in tokens:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    
    return counts


def getAMef(AM, e, f):
    if len(AM) == 0:
        return 0
    if e not in AM:
        return 0
    if len(AM[e]) == 0:
        return 0
    if f not in AM[e]:
        return 0
    return AM[e][f]


if __name__ == "__main__":
    data_dir = "/u/cs401/A2_SMT/data/Hansard/Training/"
    saved_files = ''
    fn_AM = os.path.join(saved_files, "mine_fn_AM")
    AM = align_ibm1(data_dir, 1000, 10, fn_AM)
    print(AM)
