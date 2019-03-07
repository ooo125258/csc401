import re

def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    # TODO: Implement Function

    import re

    SENT_START = "SENTSTART"
    SENT_END = "SENTEND"

    # send everything to lowercase as well
    in_sentence = in_sentence.lower()
    out_sentence = re.sub(r"([;,\(\)\-\+<>\'\"])", r" \1 ", in_sentence)



    out_sentence = re.sub(r"([\.\!\?]+)$", r" \1 ", out_sentence)
    out_sentence = "{} {} {}".format(SENT_START,out_sentence, SENT_END)

    if language != "f":

        # sentence final punctuation: we only want to separate out [!.] etc at the VERY end!
        # 


        #

        return out_sentence




    # separate leading l
    # thankfully, we can do all things at once! that is, there are no overlapping rules to apply
    #  we can do some replaces, using the groups, or we can do re.sub as well. (replace based on group)
    # we can also pass in a function as well

    import re

    out_sentence = re.sub("\b(l')",r"\1 ", out_sentence)

    # still need to add spaces to all punctuations as well

    out_sentence = re.sub("\b([cjt]')",r"\1 ", out_sentence)
    out_sentence = re.sub("\b(qu')",r"\1 ", out_sentence) #should we force a space here as well (checking that qu is not part of anything)?
    out_sentence = re.sub("'(on)",r" \1", out_sentence)


    return out_sentence


print(preprocess("l'homme; l'Afrique, l'allemagne, 3-4 je t'aime qu'on qu'il puisqu'on qu'on lorsqu'il!!!!", "f"))
