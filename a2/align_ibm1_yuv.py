from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os


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
    AM = initialize(training_data["eng"], training_data["fre"])
    # Iterate between E and M steps #todo implement the final algorithm
    t = 0

    for i in range(max_iter):
        em_step(AM, training_data["eng"], training_data["fre"])

    # Storing the AM in a pickle file
    with open(fn_AM+'.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return AM


# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
    Read up to num_sentences from train_dir.
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	Make sure to read the files in an aligned manner.
    :param train_dir: (string) The top-level directory name containing data
    :param num_sentences: (int) the maximum number of training sentences to consider
    :return:
    """
    counter = 0
    data = {"eng": {}, "fre": {}}
    for subdirs, dirs, files in os.walk(train_dir):
        for file in files:
            full_file_name = os.path.join(train_dir, file)
            if full_file_name[-1] == "e":
                open_file1 = open(full_file_name)
                open_file2 = open(full_file_name[:-2] + ".f")
                eng_line = open_file1.readline()
                fre_line = open_file2.readline()
                while eng_line != '':
                    eng_line = preprocess(eng_line, "e")
                    fre_line = preprocess(fre_line, "f")
                    counter += 1
                    data["eng"][counter] = eng_line
                    data["fre"][counter] = fre_line

                    if counter >= num_sentences:
                        open_file1.close()
                        open_file2.close()
                        break
                    eng_line = open_file1.readline()
                    fre_line = open_file2.readline()
                if counter >= num_sentences:
                    open_file1.close()
                    open_file2.close()
                    break
                open_file1.close()
                open_file2.close()
    return data


def initialize(eng, fre):
    """
    Initialize alignment model uniformly. Only set non-zero probabilities where word pairs appear
    in corresponding sentences.
    :param eng: (dict(int: str)) A dictionary containing the training English sentences
    :param fre: (dict(int: str)) A dictionary containing the training French sentences
    :return AM: (dict(str: dict(str: float))
    """
    AM = {}
    if not len(eng) == len(fre):
        print("Invalid training input.")
        return AM

    for i in range(1, len(eng)+1):
        english_words = eng[i].split()
        french_words = fre[i].split()
        for j in range(1, len(english_words) - 1):
            if english_words[j] in AM:
                for k in range(1, len(french_words) - 1):
                    AM[english_words[j]][french_words[k]] = 1
            else:
                AM[english_words[j]] = {}
                for k in range(1, len(french_words) - 1):
                    AM[english_words[j]][french_words[k]] = 1
        # Update the alignments
        for l in range(1, len(english_words) - 1):
            for word in AM[english_words[l]]:
                if float(len(AM[english_words[l]])) != 0:
                    AM[english_words[l]][word] = 1/float(len(AM[english_words[l]]))
                else:
                    AM[english_words[l]][word] = 0

    # SENTSTART and SENTEND cases
    AM["SENTSTART"] = {}
    AM["SENTSTART"]["SENTSTART"] = 1
    AM["SENTEND"] = {}
    AM["SENTEND"]["SENTEND"] = 1
    return AM


def em_step(AM, eng, fre):
    """
    One step in the EM algorithm. Follows the pseudo-code given in the tutorial slides.
    :param t:
    :param eng:
    :param fre:
    :return:
    """
    tcount_domain, tcount_value = {}, 0
    total_domain, total_value = {}, 0

    if not len(eng) == len(fre):
        print("Invalid training input.")
        return AM
    for i in range(1, len(eng)+1):
        french_sentence = fre[i].split()
        for fre_word in set(french_sentence):
            #print(word)
            if fre_word == "SENTSTART" or fre_word == "SENTEND":
                pass
            else:
                denom_c = 0
                english_sentence = eng[i].split()
                for eng_word in set(english_sentence):
                    if eng_word == "SENTSTART" or eng_word == "SENTEND":
                        pass
                    else:
                        denom_c = AM[eng_word][fre_word]*french_sentence.count(fre_word)
                for eng_word in set(english_sentence):
                    if eng_word == "SENTSTART" or eng_word == "SENTEND":
                        pass
                    else:
                        tcount_value += (AM[eng_word][fre_word]*french_sentence.count(fre_word) *
                                   english_sentence.count(eng_word)) / float(denom_c)
                        total_value += (AM[eng_word][fre_word] * french_sentence.count(fre_word) *
                                   english_sentence.count(eng_word)) / float(denom_c)
                        total_domain[eng_word] = total_value
                        if eng_word in tcount_domain:
                            tcount_domain[eng_word][fre_word] = tcount_value
                        else:
                            tcount_domain[eng_word] = {}
    for eng_word in total_domain:
        for fre_word in tcount_domain[eng_word]:
            AM[eng_word][fre_word] = tcount_domain[eng_word][fre_word]/ float(total_domain[eng_word])