from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import pickle
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
	eng, fre = read_hansard(train_dir, num_sentences)

	# Initialize AM uniformly
	AM = initialize(eng, fre)

	# Iterate between E and M steps
	for i in range(0, max_iter):
		AM = em_step(AM, eng, fre)

	# Save Model
	with open(fn_AM+'.pickle', 'wb') as handle:
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
	# TODO
	# define output and count
	training_eng = []
	training_fren = []
	count = 0

	# get all filenames under the train_dir
	files_list = os.listdir(train_dir)

	for file_path in files_list:
		# check whether the file path is valid
		filename, file_extension = os.path.splitext(file_path)

		if file_extension == '.e':
			# open and read english and french files
			eng_path = train_dir + file_path
			fren_path = train_dir + filename + '.f'

			eng_file = open(eng_path, "r")
			eng_lines = eng_file.read().split('\n')
			eng_file.close()

			fren_file = open(fren_path, "r")
			fren_lines = fren_file.read().split('\n')
			fren_file.close()

			# loop over lines
			for i in range(0, len(eng_lines)):
				if count < num_sentences:
					proc_eng_line = preprocess(eng_lines[i], 'e')
					proc_fren_line = preprocess(fren_lines[i], 'f')

					training_eng.append(proc_eng_line)
					training_fren.append(proc_fren_line)

					count += 1
				else:
					# check whether have enough training data, if yes stop the
					# inner for loop
					break

			# check whether have enough training data, if yes stop the outer
			# for loop
			if count >= num_sentences:
				break

	return training_eng, training_fren

def initialize(eng, fre):
	"""
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
	# TODO

	# define output and special cases
	alignment_model = {'SENTSTART': {'SENTSTART':{1}}, 'SENTEND':{'SENTEND':{1}}}

	# get all english tokens
	eng_tokens = []
	for sentence in eng:
		tokens = sentence.split()[1:-1] # get rid of 'SENTSTART' and 'SENTEND'
		eng_tokens.extend(tokens)
	# get rid of the duplicated ones
	eng_tokens = list(set(eng_tokens))

	# loop over each english tokens and compute counts for its alignments
	align_info = {}
	for token in eng_tokens:
		align_list = []
		for i in range(0, len(eng)):
			eng_sen_tokens = eng[i].split()[1:-1]
			fren_sen_tokens = fre[i].split()[1:-1]

			# if the token not in current english sentence, then no french
			# alignment can be found in the current frech sentence
			if token not in eng_sen_tokens:
				continue

			align_list.extend(fren_sen_tokens)
		# get rid of duplicated alignments
		align_list = list(set(align_list))
		# store number of alignments for current token in counts dict
		# since token is unique, then token will not be in counts dict
		align_info[token] = align_list

	for token in align_info.keys():
		fren_list = align_info[token]
		alignment_model[token] = {}
		for fren_token in fren_list:
			alignment_model[token][fren_token] = 1 / float(len(fren_list))

	return alignment_model

def em_step(t, eng, fre):
	"""
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
	# TODO
	# define output and special case
	AM = {'SENTSTART': {'SENTSTART':{1}}, 'SENTEND':{'SENTEND':{1}}}

	# initialize tcount and total
	tcount = {}
	total = {}

	for eng_token in t.keys():
		if (eng_token == 'SENTSTART') or (eng_token == 'SENTEND'):
			continue
		# key is unique so no need to check
		total[eng_token] = 0
		tcount[eng_token] = {}
		for fren_token in t[eng_token]:
			tcount[eng_token][fren_token] = 0

	# compute tcount and count for each given e, f
	for i in range(0, len(eng)):
		# make sure each token is unique
		eng_token_list = eng[i].split()[1:-1]
		fren_token_list = fre[i].split()[1:-1]

		unique_eng_tokens = list(set(eng_token_list))
		unique_fren_tokens = list(set(fren_token_list))

		for fren_token in unique_fren_tokens:
			denom = 0
			for eng_token in unique_eng_tokens:
				denom += t[eng_token][fren_token] * fren_token_list.count(fren_token)
			for eng_token in unique_eng_tokens:
				tcount[eng_token][fren_token] += (t[eng_token][fren_token] *
					fren_token_list.count(fren_token) *
					eng_token_list.count(eng_token)) / float(denom)
				total[eng_token] += (t[eng_token][fren_token] *
					fren_token_list.count(fren_token) *
					eng_token_list.count(eng_token)) / float(denom)

		for eng_token in total.keys():
			AM[eng_token] = {}
			for fren_token in tcount[eng_token]:
				#print(tcount[eng_token][fren_token])
				#print(total[eng_token])
				if total[eng_token] !=  0 and tcount[eng_token][fren_token] != 0:
					AM[eng_token][fren_token] = tcount[eng_token][fren_token] / float(total[eng_token])
				else:
					AM[eng_token][fren_token] = 0

	return AM

if __name__ == "__main__":
	# align_ibm1(train_dir, num_sentences, max_iter, fn_AM)
	train_dir = "/u/cs401/A2_SMT/data/Hansard/Training/"
	num_sentences = 1000
	max_iter = 10 # 5, 6, 3
	fn_AM = 'am_1000'

	result = align_ibm1(train_dir, num_sentences, max_iter, fn_AM)
	print(result)