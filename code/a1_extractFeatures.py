import numpy as np
import sys
import argparse
import os
import json
from collections import defaultdict
from numpy import loadtxt
import re
import string

import csv


def wordTagSplit(token):
	format = re.match(r'^(.*)\/(\S+)$', token, re.I)
	if format is None:
		print(token)
	if format.lastindex == 2:
		word = format.group(1)
		tag = format.group(2)
	else:
		print("something strange input! word: {}".format(token))
		return format.group(1), format.group(2)
	return word, tag


def extract1(comment):
	# comment = " I/PRON You/PRON His/PRON and/CC did/VBD will/FUT going/ING to/CC ,/COMMA ,??/PUNCT noun/NN noun/NNS very/RB why/WP lmao/SL"
	# TODO:how do you deal with ?!?
	# TODO: if only one char, it's really possible to get error!
	# TODO: handle xx
	''' This function extracts features from a single comment

	Parameters:
		comment : string, the body of a comment (after preprocessing)

	Returns:
		feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
	'''
	print('TODO')
	# TODO: your code here
	PRP_1st = loadtxt("../Wordlists/First-person", comments="#", delimiter="\n", unpack=False, dtype=str)
	PRP_2nd = loadtxt("../Wordlists/Second-person", comments="#", delimiter="\n", unpack=False, dtype=str)
	PRP_3rd = loadtxt("../Wordlists/Third-person", comments="#", delimiter="\n", unpack=False, dtype=str)
	slang = loadtxt("../Wordlists/Slang", comments="#", delimiter="\n", unpack=False, dtype=str)

	# PRP_1st = loadtxt("/u/cs401/Wordlists/First-person", comments="#", delimiter="\n", unpack=False, dtype=str)
	# PRP_2nd = loadtxt("/u/cs401/Wordlists/Second-person", comments="#", delimiter="\n", unpack=False, dtype=str)
	# PRP_3rd = loadtxt("/u/cs401/Wordlists/Third-person", comments="#", delimiter="\n", unpack=False, dtype=str)
	# slang = loadtxt("/u/cs401/Wordlists/Slang", comments="#", delimiter="\n", unpack=False, dtype=str)

	future_tense_verbs = ["'ll", "will", "gonna"]  # The last one is going to VB, consider later

	# Now, start to check the words
	sentences = comment.split("\n")
	len_sentences = len(sentences)
	feats = [0] * 30  # This is for first 29.

	# counters fo  18
	total_char_len_nopunctonly_16 = 0
	total_tokens_num_nopunctonly_16 = 0

	# dict for 17+
	BG = {}
	RW = {}

	AOAs = []
	IMGs = []
	FAMs = []
	V = []
	A = []
	D = []

	# Statistics for xx:
	xx_file = open("xx.txt", "w+")
	with open("BristolNorms+GilhoolyLogie.csv") as BGcsv:
		reader = csv.reader(BGcsv)
		for line in reader:
			BG[line[1]] = line
	with open("Ratings_Warriner_et_al.csv") as RWcsv:
		reader = csv.reader(RWcsv)
		for line in reader:
			RW[line[1]] = line

	for sentence in sentences:
		# to avoid some silly empty string.
		if sentence == "" or sentence is None:
			len_sentences -= 1
			continue

		tokens = sentence.split()
		for i in range(len(tokens)):

			if tokens[i] == "":
				continue
			# The word should be word/tag format
			ret = wordTagSplit(tokens[i])
			word = ""
			tag = ""
			if ret is None:
				print("something strange input! sentence: {} word: {}".format(sentence, tokens[i]))
				continue
			else:
				(word, tag) = ret

			# 1,2,3, 1st/2nd/3rd person pronoun
			lower_word = word.lower()
			if lower_word in PRP_1st:
				feats[1] += 1
			if lower_word in PRP_2nd:
				feats[2] += 1
			if lower_word in PRP_3rd:
				feats[3] += 1
			# 4, CC
			if tag == "CC":
				feats[4] += 1
			# 5, VBD
			if tag == "VBD":
				feats[5] += 1
			# 6, Future
			# going to will be handled in present TODO: related to "will"
			if lower_word in future_tense_verbs:
				feats[6] += 1
			if (lower_word == "going" or (lower_word == "go" and tag == "VBG")) and i < len(tokens) - 2:
				# judge for to do
				# pos_for_TO:
				first_token = wordTagSplit(tokens[i + 1])
				second_token = wordTagSplit(tokens[i + 2])
				if first_token[0].lower() == "to" and first_token[1] == "TO" and second_token[1] in ["VB", "VBP"]:
					feats[6] += 1
			# 7, comma
			if lower_word == ',':
				feats[7] += 1
			# 8, multi-char punctuation tokens
			# multi #TODO: may be simplified to accelerate
			if len(lower_word) > 1 and all(i in string.punctuation for i in lower_word):
				feats[8] += 1
			# 9, common nouns
			if tag in ["NN", "NNS"]:
				feats[9] += 1
			# 10, proper nouns
			if tag in ["NNP", "NNPS"]:
				feats[10] += 1
			# 11, adverbs
			if tag in ["RB", "RBR", "RBS", "RP"]:
				feats[11] += 1
			# 12, wh- words
			if tag in ["WDT", "WP", "WP$", "WRB"]:
				feats[12] += 1
			# 13, slang acronyms
			if tag in slang:
				feats[13] += 1
			# 14, words in uppercase
			if len(word) >= 3 and word.isupper():
				feats[14] += 1

			# 18-20 if token
			if lower_word in BG:
				AOAs.append(int(BG[lower_word][3]))
				IMGs.append(int(BG[lower_word][4]))
				FAMs.append(int(BG[lower_word][5]))
			# 24-26
			if lower_word in RW:
				V.append(float(RW[lower_word][2]))
				A.append(float(RW[lower_word][5]))
				D.append(float(RW[lower_word][8]))

			if tag == "xx":
				print("unknown xx word: {} in sentence {}.".format(tokens[i], sentence), file=xx_file)

			# 16
			if not all(i in string.punctuation for i in lower_word):
				total_char_len_nopunctonly_16 += len(word)
				total_tokens_num_nopunctonly_16 += 1

	# 15, average len of sentences in tokens
	if len_sentences != 0:
		feats[15] = len(tokens) / len_sentences
	# 16 avg len of tokens, excluding punct-only
	if total_tokens_num_nopunctonly_16 > 0:
		feats[16] = total_char_len_nopunctonly_16 / total_tokens_num_nopunctonly_16
		# 17 num of sentences
		feats[17] = len_sentences
	# 18,21 avg aoa:
	if len(AOAs) > 0:
		feats[18] = np.mean(AOAs)
		feats[21] = np.std(AOAs)
	# 19, 22IMG
	if len(IMGs) > 0:
		feats[19] = np.mean(IMGs)
		feats[22] = np.std(IMGs)
	# 20,23
	if len(FAMs) > 0:
		feats[20] = np.mean(FAMs)
		feats[23] = np.std(FAMs)

	# 24,27
	if len(V) > 0:
		feats[24] = np.mean(V)
		feats[27] = np.std(V)
	# 25, 28IMG
	if len(A) > 0:
		feats[25] = np.mean(A)
		feats[28] = np.std(A)
	# 26,29
	if len(D) > 0:
		feats[26] = np.mean(D)
		feats[29] = np.std(D)

	return np.array(feats[1:])  # Return 29 bits


def main(args):
	data = json.load(open(args.input))
	feats = np.zeros((len(data), 174))

	# TODO: your code here
	alt_feats = np.load("../feats/Alt_feats.dat.npy")
	alt_IDs = np.loadtxt("../feats/Alt_IDs.txt", comments="#", delimiter="\n", unpack=False, dtype=str)
	print("doing 2")
	llwc_feats = np.loadtxt("../feats/feats.txt", comments="#", delimiter="\n", unpack=False, dtype=str)
	left_feats = np.load("../feats/Left_feats.dat.npy")
	print("doing 3")
	left_IDs = np.loadtxt("../feats/Left_IDs.txt", comments="#", delimiter="\n", unpack=False, dtype=str)
	right_feats = np.load("../feats/Right_feats.dat.npy")
	right_IDs = np.loadtxt("../feats/Right_IDs.txt", comments="#", delimiter="\n", unpack=False, dtype=str)
	print("doing 4")
	center_feats = np.load("../feats/Center_feats.dat.npy")
	center_IDs = np.loadtxt("../feats/Center_IDs.txt", comments="#", delimiter="\n", unpack=False, dtype=str)
	print("doing 5")
	'''
	alt_feats = np.load("/u/cs401/A1/feats/Alt_feats.dat.npy")
	alt_IDs = np.loadtxt("/u/cs401/A1/feats/Alt_IDs.txt", comments="#", delimiter="\n", unpack=False, dtype=str)
	print("doing 2")
	llwc_feats = np.loadtxt("/u/cs401/A1/feats/feats.txt", comments="#", delimiter="\n", unpack=False, dtype=str)
	left_feats = np.load("/u/cs401/A1/feats/Left_feats.dat.npy")
	print("doing 3")
	left_IDs = np.loadtxt("/u/cs401/A1/feats/Left_IDs.txt", comments="#", delimiter="\n", unpack=False, dtype=str)
	right_feats = np.load("/u/cs401/A1/feats/Right_feats.dat.npy")
	right_IDs = np.loadtxt("/u/cs401/A1/feats/Right_IDs.txt", comments="#", delimiter="\n", unpack=False, dtype=str)
	print("doing 4")
	center_feats = np.load("/u/cs401/A1/feats/Center_feats.dat.npy")
	center_IDs = np.loadtxt("/u/cs401/A1/feats/Center_IDs.txt", comments="#", delimiter="\n", unpack=False, dtype=str)
	print("doing 5")
	'''
	from tqdm import tqdm
	for i in tqdm(range(len(data))):
		# if (i % 100 == 0):
		#     print("complete: "+ str(i/float(len(data))*100) + "%")
		ret = extract1(data[i]["body"])
		feats[i][:29] = ret

		# It could be better ways, as the number is fixed.
		# But I haven't prove it.

		if data[i]["cat"] == "Left":
			itemindex, = np.where(left_IDs == data[i]["id"])
			thisFeat = left_feats[itemindex]
			feats[i][29:173] = thisFeat
			feats[i][-1] = 0
		elif data[i]["cat"] == "Center":
			itemindex, = np.where(center_IDs == data[i]["id"])
			thisFeat = center_feats[itemindex]
			feats[i][29:173] = thisFeat
			feats[i][-1] = 1
		elif data[i]["cat"] == "Right":
			itemindex, = np.where(right_IDs == data[i]["id"])
			thisFeat = right_feats[itemindex]
			feats[i][29:173] = thisFeat
			feats[i][-1] = 2
		elif data[i]["cat"] == "Alt":
			feats[i][-1] = 3
			itemindex, = np.where(alt_IDs == data[i]["id"])
			thisFeat = alt_feats[itemindex]
			feats[i][29:173] = thisFeat

	np.savez_compressed(args.output, feats)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process each .')
	parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
	parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
	args = parser.parse_args()

	main(args)
