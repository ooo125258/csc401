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

    modComm = re.sub(r'\s+', r' ', modComm)
    
    #Code from a1
    new_modComm = re.sub(  # Sorry, but when adding \., it will translate to \\. at once.
        r"(\W+)(Ala\.|Ariz\.|Assn\.|Atty\.|Aug\.|Ave\.|Bldg\.|Blvd\.|Calif\.|Capt\.|Cf\.|Ch\.|Co\.|Col\.|Colo\.|Conn\.|Corp\.|DR\.|Dec\.|Dept\.|Dist\.|Dr\.|Drs\.|Ed\.|Eq\.|FEB\.|Feb\.|Fig\.|Figs\.|Fla\.|Ga\.|Gen\.|Gov\.|HON\.|Ill\.|Inc\.|JR\.|Jan\.|Jr\.|Kan\.|Ky\.|La\.|Lt\.|Ltd\.|MR\.|MRS\.|Mar\.|Mass\.|Md\.|Messrs\.|Mich\.|Minn\.|Miss\.|Mmes\.|Mo\.|Mr\.|Mrs\.|Ms\.|Mx\.|Mt\.|NO\.|No\.|Nov\.|Oct\.|Okla\.|Op\.|Ore\.|Pa\.|Pp\.|Prof\.|Prop\.|Rd\.|Ref\.|Rep\.|Reps\.|Rev\.|Rte\.|Sen\.|Sept\.|Sr\.|St\.|Stat\.|Supt\.|Tech\.|Tex\.|Va\.|Vol\.|Wash\.|al\.|av\.|ave\.|ca\.|cc\.|chap\.|cm\.|cu\.|dia\.|dr\.|eqn\.|etc\.|fig\.|figs\.|ft\.|gm\.|hr\.|in\.|kc\.|lb\.|lbs\.|mg\.|ml\.|mm\.|mv\.|nw\.|oz\.|pl\.|pp\.|sec\.|sq\.|st\.|vs\.|yr\.|i\.e\.|e\.g\.)",
        r"\1 \2 ", modComm, flags=re.IGNORECASE)
    # Dot here is a problem. The one dot situation will be handled later. For all "..." and "?.?" will be handled
    # It's ridiculous, but ... has higher priority.
    new_modComm = re.sub(r"(\.\.\.)(\w|\s|$)", r" \1 \2", new_modComm)
    new_modComm = re.sub(
        r"(\w|\s\^)(\.\.\.|[\!\#\$\%\&\\\(\)\*\+\,\-\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~\.\']{2,}|[\!\#\$\%\&\\\(\)\*\+\,\-\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~])(\w|\s|$)",
        r"\1 \2 \3", new_modComm)
    # Special operation for brackets
    new_modComm = re.sub(r"(\W|\^)([\[\(\{\'\"])", r"\1 \2 ", new_modComm)
    # quote is a problem. when \w+\', you don't know if it's person's or the player'FLASH' or sth.
    # But you are sure that if \s\', it must be a quote for reference!
    new_modComm = re.sub(r"(\]|\)|\})(\W|\$|\.)", r" \1 \2", new_modComm)

    # Handle the dot problem. If find a word followed by dot, then check if it's a word in abbrs1001369404 list
    def periodHandler(matched):
        lst = abbrs1001369404_lower
        word = matched.group().strip()
        if word.lower() in lst:
            return " " + word + " "
        else:  # There is such a situation: apple.Tree e.g..Tree apple.E.g. So I choose the capital to be the identifier.
            return " " + word.replace(".", " . ")

    # e.g.. , e.g. something, etc. something, something.
    # Another situation is something.\nsomething, such that it's connected!
    new_modComm = re.sub(r"(^|\s)((\w+\.)+\.?)($|\s|\w+)", periodHandler, new_modComm, flags=re.IGNORECASE)

    return out_sentence
