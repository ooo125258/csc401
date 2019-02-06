import numpy as np
import sys
import argparse
import os
import json
from collections import defaultdict
from numpy import loadtxt

def wordTagSplit(token):
    format = re.match(r'^(.*)\/(\w+)$', token, re.I)
    if len(format) == 2:
        (word, tag) = format
    else:
        print("something strange input! word: {}".format(token))
        return None
    return word, tag
        
def helper_8(token):
    format = re.match(r'^(.*)\/(\w+)$', token, re.I)
        
def extract1( comment ):
    #TODO:how do you deal with ?!?
    #TODO: if only one char, it's really possible to get error!
    #TODO: handle xx
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    print('TODO')
    # TODO: your code here
    PRP_1st = loadtxt("/u/cs401/Wordlists/First-person", comments="#", delimiter="\n", unpack=False)
    PRP_2nd = loadtxt("/u/cs401/Wordlists/Second-person", comments="#", delimiter="\n", unpack=False)
    PRP_3rd = loadtxt("/u/cs401/Wordlists/Third-person", comments="#", delimiter="\n", unpack=False)
    future_tense_verbs = ["'ll", "will", "gonna"] #The last one is going to VB, consider later
    slang = loadtxt("/u/cs401/Wordlists/Slang", comments="#", delimiter="\n", unpack=False)
    #Now, start to check the words
    sentences = comment.split("\n")

    numbers = [0] * 30 #This is for first 29.
    for sentence in sentences:
        #to avoid some silly empty string.
        if sentence == "" or sentence is None:
            continue

        tokens = sentence.split()

        for i in range(len(tokens)):
            if tokens[i] = "":
                continue
            #The word should be word/tag format
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
                numbers[1] += 1
            elif lower_word in PRP_2nd:
                numbers[2] += 1
            elif lower_word in PRP_3rd:
                numbers[3] += 1
            #4, CC
            elif tag == "CC":
                numbers[4] += 1
            #5, VBD
            elif tag == "VBD":
                numbers[5] += 1
            #6, Future
            #going to will be handled in present TODO: related to "will"
            elif lower_word in future_tense_verbs:
                numbers[6] += 1
            elif lower_word == "going" and i <= len(tokens) - 2:
                #judge for to do
                #pos_for_TO: 
                first_token = wordTagSplit(tokens[i + 1])                
                second_token = wordTagSplit(tokens[i + 2])
                if first_token[0].lower() == "to" and first_token[1] in ["VB", "VBG", "VBP"]:
                    numbers[6] += 1
            #7, comma
            elif lower_word == ',':
                numbers[7] += 1
            #8, multi-char punctuation tokens
            #multi
            elif len(lower_word) > 1 and all(i in string.punctuation for i in lower_word):
                numbers[8] += 1
            #9, common nouns
            elif tag in ["NN", "NNS"]:
                numbers[9] += 1
            #10, proper nouns
            elif tag in ["NNP", "NNPS"]:
                numbers[10] += 1
            #11, adberbs
            elif tag in ["RB", "RBR", "RBS", "RP"]:
                numbers[11] += 1
            #12, wh- words
            elif tag in ["WDT", "WP", "WP$", "WRB"]:
                numbers[12] += 1
            #13, slang acronyms
            elif tag in slang:
                numbers[13] += 1
            #14, words in uppercase
            elif len(word) >= 3 and word.isupper():
                numbers[14] += 1
            else:
                if tag == "xx":
                    print("unknown xx word: {} in sentence {}.".format(tokens[i], sentence))
                else:
                    print("unknown other word: {} in sentence {}.".format(tokens[i], sentence))



def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    # TODO: your code here

    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

