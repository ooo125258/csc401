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

    #lower-case and remove extra space
    modComm = in_sentence.lower().strip()

    modComm = re.sub(r"\s+", r" ", modComm)

    #split commas, colons,
    #semicolons, parentheses, dashes, mathmatical operators, quotationmarks
    modComm = re.sub(r"([\!\"\#\$\%\&\\\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~])", r" \1 ", modComm)

    if language == 'f':
        #l', or L'
        modComm = re.sub(r"(^|\s)(l\')(\w+)", "\1\2 \3", modComm)
        #single-consonnant words with e-muet #except some d
        modComm = re.sub(r"(^|\s)([bcdfghjkmnpqrstvxz]\')(\w+)", r"\1\2 \3", modComm)
        #que
        modComm = re.sub(r"(^|\s)(qu\')(\w+)", r"\1\2 \3", modComm)
        #conjuctions puisque and lorsque
        modComm = re.sub(r"(\')(on|il)", r"\1 \2", modComm)
    #Code from a1
        #The d word:
        def d_word_handler(matched):
            word = matched.group().strip()
            if word == "d'abord" or word == "d'accord" or word == "d'ailleurs" or word == "d'habitude'":
                return ' ' + word + ' '
            else:
                word = word.replace("'", " ' ")
                return ' ' + word + ' '
        modComm = re.sub(r"(^|\s)d\'\w+", d_word_handler, modComm)

        #strip and add tags
        modComm = modComm.strip()
        modComm = re.sub(r"\s+", r" ", modComm)
    out_sentence = "SENTSTART " + modComm + " SENTEND"
    return out_sentence
