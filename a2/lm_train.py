from preprocess import *
import pickle
import os
import re

def lm_train(data_dir, language, fn_LM):
    """
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM
	
	INPUTS:
	
    data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained
    
    OUTPUT
	
	LM			: (dictionary) a specialized language model
	
	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which 
	incorporate unigram or bigram counts
	
	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
		  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """
	
	# TODO: Implement Function
    LM = {'uni':{}, 'bi':{}}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not (len(file) > 2 and file[-1] == language and file[-2] == '.'):
                continue
            fDatafile = open(os.path.join(root, file), "r")
            sLine = fDatafile.readline()
            while sLine:
                sPre_proc = preprocess(sLine, language)
                lTokens = sPre_proc.split()
                #unigram
                for i in range(len(lTokens)):
                    if lTokens[i] in LM['uni']:
                        LM['uni'][lTokens[i]] += 1
                    else:
                        LM['uni'][lTokens[i]] = 1
                #bigram
                    if i + 1 >= len(lTokens):
                        continue
                    if lTokens[i] not in LM['bi']:
                        LM['bi'][lTokens[i]] = {lTokens[i + 1] : 1}
                    elif lTokens[i + 1] not in LM['bi'][lTokens[i]]:
                        LM['bi'][lTokens[i]][lTokens[i + 1]] = 1
                    else:
                        LM['bi'][lTokens[i]][lTokens[i + 1]] += 1
    language_model = LM                
    #Save Model
    with open(fn_LM+'.pickle', 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return language_model

if __name__ == "__main__":
    #lm_train(data_dir, language, fn_LM)
    data_dir = "/u/cs401/A2_SMT/data/Hansard/Training/"
    saved_files = '/h/u15/c4/00/sunchuan/csc401/a2/'
    fn_LMe = os.path.join(saved_files, "LMe")
    fn_LMf = os.path.join(saved_files, "LMf")
    lme = lm_train(data_dir, 'e', fn_LMe)
    lmf = lm_train(data_dir, 'f', fn_LMf)
    print(lme)
    print("\n================================================\n")
    print(lmf)
