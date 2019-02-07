import numpy as np
import sys
import argparse
import os
import json
import string
import csv
import bz2


def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 29-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    # TODO: your code here
    feats = np.zeros(174)
    first_pn = ["i","me","my","mine","we","us","our","ours"]
    second_pn = ["you", "your", "yours", "u", "ur", "urs"]
    third_pn = ["he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"]
    future_v = ["'ll", "gonna"]
    sentences = comment.split("\n")
    sentence_num = 0
    token_num = 0
    char_num = 0
    token_nopunc_num = 0
    BNGL_word_dict = {}
    W_dict = {}
    W_V = []
    W_A = []
    W_D = []
    BNGL_A = []
    BNGL_I = []
    BNGL_F = []
    with open('BristolNorms+GilhoolyLogie.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader: 
            if ((row[0] != "Source") and (row[0]!="")):
                BNGL_word_dict[row[1].lower()] = [float(row[3]),float(row[4]),float(row[5])]
    with open('Ratings_Warriner_et_al.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader: 
            if ((row[2] != "V.Mean.Sum") and (row[0]!="")):
                W_dict[row[1].lower()] = [float(row[2]),float(row[5]),float(row[8])]
    for sentence in sentences:
        if (sentence != ""):
            sentence_num +=1
            words = sentence.split(" ")
            for i in range(len(words)):
                if (words[i] != ""):
                    token_num +=1
                    tokens =  words[i].split("/")
                    if (tokens[0] != ""):
                        if (tokens[0].lower() in first_pn):
                            feats[0] +=1
                        if (tokens[0].lower() in second_pn):
                            feats[1] +=1
                        if (tokens[0].lower() in third_pn):
                            feats[2] +=1
                        if (len(tokens) > 1):
                            if (tokens[1] == "CC"):
                                feats[3] +=1
                            if (tokens[1] == "VBD"):
                                feats[4] +=1
                        if (tokens[0].lower() in future_v):
                            feats[5] +=1
                        if ((tokens[0].lower() == "will") and (i<len(words)-1)):
                            tokens1 = words[i+1].split("/")
                            if ((len(tokens1) > 1) and (tokens1[1] == "VB")):
                                feats[5] +=1
                        if ((len(tokens) > 1) and (tokens[1] == "VB") and (i>1)):
                            tokens1 = words[i-2].split("/")
                            tokens2 = words[i-1].split("/")
                            if ((tokens1[0].lower == "going") and (tokens2[0].lower == "to")):
                                feats[5] +=1
                        if ("," in tokens[0]):
                            feats[6] +=1
                        if ((len(tokens[0]) > 1) and (tokens[0][0] in string.punctuation) and
                            (tokens[0][-1] in string.punctuation)):
                            feats[7] +=1
                        else:
                            if ((len(tokens[0]) >1) or not(tokens[0][0] in string.punctuation)):
                                token_nopunc_num +=1
                                for w in tokens[0]:
                                    if not(w in string.punctuation):
                                        char_num +=1
                        if ((len(tokens) > 1) and ((tokens[1] == "NN") or (tokens[1] == "NNS"))):
                            feats[8] +=1
                        if ((len(tokens) > 1) and ((tokens[1] == "NNP") or (tokens[1] == "NNPS"))):
                            feats[9] +=1
                        if ((len(tokens) > 1) and (tokens[1] in ["RB","RBR","RBS"])):
                            feats[10] +=1
                        if ((len(tokens) > 1) and (tokens[1] in ["WDT", "WP", "WP$", "WRB"])):
                            feats[11] +=1
                        if (tokens[0].lower() in (open('Slang').read().lower().split("\n"))):
                            feats[12] +=1
                        if ((len(tokens[0])>3) and (tokens[0].isupper())):
                            feats[13] +=1
                        if (tokens[0].lower() in BNGL_word_dict.keys()):
                            BNGL_A.append(BNGL_word_dict[tokens[0].lower()][0])
                            BNGL_I.append(BNGL_word_dict[tokens[0].lower()][1])
                            BNGL_F.append(BNGL_word_dict[tokens[0].lower()][2])
                        if (tokens[0].lower() in W_dict.keys()):
                            W_V.append(W_dict[tokens[0].lower()][0])
                            W_A.append(W_dict[tokens[0].lower()][1])
                            W_D.append(W_dict[tokens[0].lower()][2]) 
    if (sentence_num != 0):
        feats[14] = token_num/float(sentence_num)
    if (token_nopunc_num != 0):
        feats[15] = char_num/float(token_nopunc_num)
    feats[16] = sentence_num
    if (BNGL_A != []):
        feats[17] = np.mean(np.array(BNGL_A))
    if (BNGL_I != []):
        feats[18] = np.mean(np.array(BNGL_I))
    if (BNGL_F != []):
        feats[19] = np.mean(np.array(BNGL_F))
    if (BNGL_A != []):
        feats[20] = np.std(np.array(BNGL_A))
    if (BNGL_I != []):
        feats[21] = np.std(np.array(BNGL_I))
    if (BNGL_F != []):
        feats[22] = np.std(np.array(BNGL_F))
    if (W_V != []):
        feats[23] = np.mean(np.array(W_V))
    if (W_A != []):
        feats[24] = np.mean(np.array(W_A))
    if (W_D != []):    
        feats[25] = np.mean(np.array(W_D))
    if (W_V != []):    
        feats[26] = np.std(np.array(W_V))
    if (W_A != []):    
        feats[27] = np.std(np.array(W_A))
    if (W_D != []):    
        feats[28] = np.std(np.array(W_D))
    return feats





def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))
    
    print("start")
    # TODO: your code here
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
    from tqdm import tqdm
    for i in tqdm(range(len(data))):
        # if (i % 100 == 0):
        #     print("complete: "+ str(i/float(len(data))*100) + "%")
        feats[i] = extract1(data[i]["body"])
        
        #It could be better ways, as the number is fixed.
        #But I haven't prove it.

        if (data[i]["cat"] == "Alt"):
            feats[i][-1] = 3
            itemindex, = np.where(alt_IDs == data[i]["id"])
            thisFeat = alt_feats[itemindex]
            feats[i][30:174] = thisFeat
        elif (data[i]["cat"] == "Center"):
            itemindex, = np.where(center_IDs == data[i]["id"])
            thisFeat = center_feats[itemindex]
            feats[i][30:174] = thisFeat
            feats[i][-1] = 1
        elif (data[i]["cat"] == "Left"):
            itemindex, = np.where(left_IDs == data[i]["id"])
            thisFeat = left_feats[itemindex]
            feats[i][30:174] = thisFeat
            feats[i][-1] = 0
        elif (data[i]["cat"] == "Right"):
            itemindex, = np.where(right_IDs == data[i]["id"])
            thisFeat = right_feats[itemindex]
            feats[i][30:174] = thisFeat
            feats[i][-1] = 2
    

    
    np.savez_compressed( args.output, feats)




    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

