from preprocess import *
import pickle
import os

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

    # get all the files in the data_dir, according to the language
    # we can use a glob.glob, or we can recursively walk through the tree and get everything

    import os
    language_model = {"uni": {} , "bi": {}}
    # could also be:
    # language_model = {"uni": defaultdict(int) , "bi": {}}

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".{}".format(language)):
                with open(os.path.join(root, file)) as read_file:
                    for line in read_file:
                        line = preprocess(line, language)
                        tokens = line.split()
                        prev_word = None

                        for index,token in enumerate(tokens):
                        #     we could make a safe_add function
                            if token in language_model["uni"]:  #wecould also construct from the ground up as well (i.e. bottom up)
                                language_model["uni"][token] +=1
                            else:
                                language_model["uni"][token] = 1

                        # get the next token as well...

                        #     now, check it for the bi dict
                            if prev_word is not None:
                                if prev_word in language_model["bi"]:  # wecould also construct from the ground up as well (i.e. bottom up)
                                    if token in language_model["bi"][prev_word]:
                                        language_model["bi"][prev_word][token] +=1
                                    else:
                                        language_model["bi"][prev_word][token] = 1
                                else:
                                    language_model["bi"][prev_word] = {token: 1}
                            prev_word = token




                        #         we could also look at the previous token

    print(language_model)
    #Save Model
    with open(fn_LM+'.pickle', 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return language_model

lm_train("/u/cs401/A2_SMT/data/Toy/","e", "eee")