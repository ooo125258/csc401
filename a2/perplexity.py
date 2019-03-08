from log_prob import *
from preprocess import *
import os


def preplexity(LM, test_dir, language, smoothing=False, delta=0):
    """
	Computes the preplexity of language model given a test corpus
	
	INPUT:
	
	LM : 		(dictionary) the language model trained by lm_train
	test_dir : 	(string) The top-level directory name containing data
				e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	language : `(string) either 'e' (English) or 'f' (French)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	"""
    print(test_dir)
    files = os.listdir(test_dir)
    
    pp = 0
    N = 0
    vocab_size = len(LM["uni"])
    counter = 0
    for ffile in files:
        if ffile.split(".")[-1] != language:
            continue
        
        opened_file = open(test_dir + ffile, "r")
        for line in opened_file:
            processed_line = preprocess(line, language)
            tpp = log_prob(processed_line, LM, smoothing, delta, vocab_size)
            
            if tpp > float("-inf"):
                pp = pp + tpp
                N += len(processed_line.split())
            else:
                counter += 1
        opened_file.close()
    print("-inf:" + str(counter))
    if N > 0:
        pp = 2 ** (-pp / N)
    return pp

lm_train_testdir = "/u/cs401/A2_SMT/data/Hansard/Testing/"

def test_MLE():
    with open("unsmoothed_perplexities.txt", "w") as file:
        for lang in ["e", "f"]:
            file.write("Lang: {}\n".format(lang))
            test_LM = lm_train(lm_train_testdir, lang, "lm_{}_test".format(lang))
            delta = 0
            perplexity = preplexity(test_LM, lm_train_testdir, lang, smoothing = False, delta=delta)
            file.write("{}\n".format( perplexity))
#         format out all the perplexities



def test_smoothings():
    with open("smoothed_perplexities.txt", "w") as file:
        # do it for each language, and for each delta
        for lang in ["e", "f"]:
            file.write("Lang: {}\n".format(lang))
            test_LM = lm_train(lm_train_testdir, lang, "lm_{}_test".format(lang))
            #
            for delta_percent in range(0, 100, 10):
                delta = float(delta_percent)/100
                perplexity = preplexity(test_LM, lm_train_testdir, lang, smoothing = True, delta=delta)
                file.write("Delta: {}, Perplex: {}\n".format(delta, perplexity))

            #     now call the perplexity




        pass

    pass


    
# test
if __name__ == "__main__":
    test_MLE()
    test_smoothings()
    print("------------------------------------")
    data_dir = "/u/cs401/A2_SMT/data/Hansard/Testing/"
    saved_files = '/h/u15/c4/00/sunchuan/csc401/a2/'
    fn_LMe = os.path.join(saved_files, "Task3_LMe")
    
    test_LM = lm_train(data_dir, 'e', fn_LMe)
    print(preplexity(test_LM, data_dir, "e"))
# test_LM = lm_train("lm_train_testdir/", "e", "e_temp")
# print(preplexity(test_LM, "lm_train_testdir/", "e"))
