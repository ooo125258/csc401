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
					e.g., '/data/Hansard/Testing/'
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
    training_set = read_hansard(train_dir, num_sentences)

    # Initialize AM uniformly
    english_s = []
    french_s = []

    for sent_pair in training_set:
        english_s.append(sent_pair[0])
        french_s.append(sent_pair[1])

    # the initialized AM
    AM = initialize(english_s, french_s)

    # for each iteration
    for i in range(max_iter):
        # initialize tcounts and total
        tcounts = {}
        total = {}
        for e in AM:
            total[e] = 0
            tcounts[e] = {}
            for f in AM[e]:
                tcounts[e][f] = 0

        # update tcounts and total
        for sent_pair in training_set:
            unique_f = unique_words(sent_pair[1])
            for f in unique_f:
                denom_c = 0
                unique_e = unique_words(sent_pair[0])
                for e in unique_e:
                    denom_c += AM[e][f] * unique_f[f]
                for e in unique_e:
                    tcounts[e][f] += AM[e][f] * unique_f[f] * unique_e[e] / denom_c
                    total[e] += AM[e][f] * unique_f[f] * unique_e[e] / denom_c

        # update alignment model
        for e in total:
            for f in tcounts[e]:
                AM[e][f] = tcounts[e][f] / total[e]

    AM['SENTSTART']['SENTSTART'] = 1
    AM['SENTEND']['SENTEND'] = 1

    # save AM
    with open(fn_AM +'.pickle', 'wb') as handle:
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
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	Make sure to read the files in an aligned manner.
	"""
    # training set has tuples of english sentences and corresponding french sentences
    training_set = []
    i = 0
    files = os.listdir(train_dir)
    for ffile in files:
        if ffile.split(".")[-1] == 'e':
            english_file = ffile
            french_file = ffile[:-1] + 'f'

            opened_e = open(train_dir+english_file, "r").read()
            opened_f = open(train_dir+french_file, "r").read()

            e_lines = opened_e.split("\n")
            f_lines = opened_f.split("\n")

            for j in range(len(e_lines)):
                if i != num_sentences:
                    training_set.append((preprocess(e_lines[j], 'e'), preprocess(f_lines[j], 'f')))
                    i += 1
    return training_set

def initialize(eng, fre):
    """
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
    init = {}

    # all the english words
    e_tokens = []
    for sent in eng:
        tokens = sent.split()
        for token in tokens:
            if token not in e_tokens:
                e_tokens.append(token)

    # for each english word give the sum of lengths of french sentences with the
    # english word in their corresponding english sentences
    e_lens = {}
    for e in e_tokens:
        for sent in range(len(eng)):
            if e in eng[sent].split():
                e_lens[e] = len(fre[sent].split())

    # for each english word in english sentence, for each french word in french
    # sentence init[eng_word][french_word] = 1 / e_lens[e_word]
    for sent in range(len(eng)):
        for eword in eng[sent].split():
            if eword not in init:
                init[eword] = {}
            for fword in fre[sent].split():
                init[eword][fword] = 1 / e_lens[eword]

    return init

def em_step(t, eng, fre):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
	# TODO

def unique_words(sentence):
    """
    Returns a dictionary with unique tokens and their count.
    """
    unique = {}
    tokens = sentence.split()
    for token in tokens:
        if token not in unique:
            unique[token] = 1
        else:
            unique[token] += 1
    return unique

if __name__ == "__main__":

    data_dir = '/u/cs401/A2_SMT/data/Hansard/Training/'
    saved_files = ''
    fn_AM = '{}sth_fn_AM'.format(saved_files)
    AM = align_ibm1(data_dir, 1000, 3, fn_AM)
    print(AM)