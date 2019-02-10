import sys

import argparse
import html
import json
import os
import re
import spacy
from tqdm import tqdm

indir = '/u/cs401/A1/data/';

abbrs = []
abbrs_lower = []
abbs_withdot = ""
clitics = []
clitics_lower = []
stopwords = []
pn_abbrs = []
nlp = spacy.load('en', disable=['parser', 'ner'])


def init():
	global abbrs
	global abbrs_lower
	global abbs_withdot
	global clitics
	global clitics_lower
	global stopwords
	global pn_abbrs
	global nlp
	
	if nlp is None:
		nlp = spacy.load('en', disable=['parser', 'ner'])
	if len(abbrs) == 0:
		abbr_filename = "./abbrev.english"
		if not os.path.isfile(abbr_filename):
			abbr_filename = "/u/cs401/Wordlists/abbrev.english"
		with open(abbr_filename) as f:
			for line in f:
				abbrs.append(line.rstrip('\n'))
		
		abbrs_lower = [x.lower() for x in abbrs]
	
	if len(clitics) == 0:
		clitics_filename = "./clitics"
		if not os.path.isfile(clitics_filename):
			clitics_filename = "/u/cs401/Wordlists/clitics"
		with open(clitics_filename) as f:
			for line in f:
				clitics.append(line.rstrip('\n'))
		clitics_lower = [x.lower() for x in clitics]
	
	if len(stopwords) == 0:
		stopwords_filename = "./StopWords"
		if not os.path.isfile(stopwords_filename):
			stopwords_filename = "/u/cs401/Wordlists/StopWords"
		with open(stopwords_filename) as f:
			for line in f:
				stopwords.append(line.rstrip('\n'))
	# stopwords = set(stopwords)
	
	if len(pn_abbrs) == 0:
		pn_abbrs_filename = "./pn_abbrev.english"
		if not os.path.isfile(pn_abbrs_filename):
			pn_abbrs_filename = "/u/cs401/Wordlists/pn_abbrev.english"
		with open(pn_abbrs_filename) as f:
			for line in f:
				add_word = line.rstrip('\n')
				pn_abbrs.append(add_word)
				if add_word not in abbrs:
					abbrs.append(add_word)
	# pn_abbrs = set(pn_abbrs)


def preproc1(comment, steps=range(1, 11)):
	''' This function pre-processes a single comment

	Parameters:
		comment : string, the body of a comment
		steps   : list of ints, each entry in this list corresponds to a preprocessing step

	Returns:
		modComm : string, the modified comment
	'''
	init()
	modComm = ''
	
	modComm_tags = []
	modComm_text = []
	modComm_lemma = []
	global nlp
	doc = None  # doc is shared in one preproc1
	
	if 1 in steps:
		if comment == "":
			return ""
		# print('Remove all newline characters.')
		# modComm = modComm.replace('\n', '')
		modComm = re.sub(r"(\.?)(\r?\n)+", r" \1 ", comment)
	# modComm = comment.replace("\.\n", " \.")
	# modComm = comment.replace("\n", "")
	if 2 in steps:
		if modComm == "":
			return ""
		# modComm = html.unescape(comment)
		# print('Replace HTML character codes with their ASCII equivalent.')
		modComm = re.sub(r"(\&\w+\;)", r" \1 ",
		                 modComm)  # TODO: to add space around the HTML char. But DOES IT WORTH IT?
		modComm = html.unescape(modComm)
	
	if 3 in steps:
		if modComm == "":
			return ""
		# print('Remove all URLs http/www/https')
		# modified from : https://daringfireball.net/2010/07/improved_regex_for_matching_urls
		modComm = re.sub(
			r"([\(\[\{]?((((http[s]?):\/\/)|(www\.))[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:\/~\+#]*[\w\-\@?^=%&amp;\/~\+#])?)[\)\]\}]?)",
			r' ', modComm)
	# modComm = re.sub(
	#    r'(?:(?:http|https)?:\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&/=>]*)?',
	#    "", modComm, flags=re.MULTILINE)
	if 4 in steps:  # TODO: modify if still slow. remember the capital abbr words
		# skip abbr, or others
		
		"""
		What to replace here? We are using regex!
		... need to be together
		?!? need to be together
		Mr., e.g. need to be together
		others split
		
		So it's the only problem for . and ', besides the second problem
		"""
		if modComm == "":
			return ""
		# first, split the abbv seperately, with space. to do it better.
		# str = abbs_withdot
		# reStr = r"\b(" + r"|".join(abbrs) + r")\b"
		# reStr = re.sub(r"\.", r"\.", reStr)
		# restr = re.sub('\\', r"\\", reStr)
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
		
		# Handle the dot problem. If find a word followed by dot, then check if it's a word in abbrs list
		def periodHandler(matched):
			lst = abbrs_lower
			word = matched.group().strip()
			if word.lower() in lst:
				return " " + word + " "
			else:  # There is such a situation: apple.Tree e.g..Tree apple.E.g. So I choose the capital to be the identifier.
				return " " + word.replace(".", " . ")
		
		# e.g.. , e.g. something, etc. something, something.
		# Another situation is something.\nsomething, such that it's connected!
		new_modComm = re.sub(r"(^|\s)((\w+\.)+\.?)($|\s|\w+)", periodHandler, new_modComm, flags=re.IGNORECASE)
		
		# Special case:  .Tree
		new_modComm = re.sub(r"\s\.(\w+)", r" . \1", new_modComm)
		# special case: dogs'. or t'.
		modComm = new_modComm.replace("'.", "' .")
	
	if 5 in steps:  # split clitics
		# modComm = re.sub(r"([\w]+)(?=\'ve)|(?=\'ll)|(?=\'re)|(?=\'s)|(?=\'d)|(?=\'m)|(?=\'\s)",
		#                 lambda pat: pat.group(0) + ' ', modComm, flags=re.I)
		# modComm = re.sub(r"([\w]+)(?=\'t)|(?=\'T)", lambda pat: pat.group(0)[:-1] + ' ' + pat.group(0)[-1], modComm,
		#                 flags=re.I)
		# modComm = modComm.replace('\\', '')
		
		if modComm == "":
			return ""
		
		def citeHandler(matched):
			lst = clitics_lower
			word = matched.group().strip()
			word_lower = word.lower()
			for i in range(len(lst)):
				if lst[i] in word_lower:
					ret = " " + word.replace(lst[i], " " + lst[i] + " ") + " "
					return ret
			ret = word.replace("'", " ' ")
			return ret
		
		new_modComm = re.sub(r"(^|\s)(\w*\'\w*)($|\s)", citeHandler, modComm)
		modComm = new_modComm
	
	if 6 in steps:  # tags
		# We already know that 689 will be together.
		'''
		'''
		if modComm == "":
			return ""
		
		if nlp is None:
			print(
				"warning: trying to load spacy in a wrong place. It would be much slower if it happens a lot of time!")
			nlp = spacy.load('en', disable=['parser', 'ner'])
		doc = spacy.tokens.Doc(nlp.vocab, words=modComm.split())
		doc = nlp.tagger(doc)
		#print([x.text+"/"+x.tag_ for x in doc])
		document = modComm
		document = nlp(document)
		#print([x.text + "/" + x.tag_ for x in document])

		new_modComm_lst = []
		for token in doc:
			new_modComm_lst.append(token.text + "/" + token.tag_)
		modComm = " ".join(new_modComm_lst)

	if 7 in steps:  # Stop words
		# split to list, remove if in stopwords
		# analysis when it has or not the tags
		# replace beta/nng or beta/-lrc- or
		# start and end with white space or head/end
		if modComm == "":
			return ""
		# pattern = re.compile(r'\b\s(' + r'|'.join(stopwords) + r')\/\b')
		# modComm = pattern.sub(r' /', modComm)
		# pattern = re.compile(r"\s/[\w]+(?=\s)")
		# modComm = pattern.sub(r'', modComm)
		# re.sub(r'\b\s(' + r'|'.join(stopwords) + r')\/\b', r' /', modComm)
		reStr = re.compile(r"\b(" + r"(\s|^)(" + r"|".join(stopwords) + r")(\/\-?\S+\-?)?" + r"(\s|$)" + r")\b", flags=re.IGNORECASE)
		new_modComm = reStr.sub(" ", modComm)
		flag = False
		#Add a flag to minimize the inaccurate regex. I don't know why but it won't remove all stop words at once.
		#May due to overlap?
		while new_modComm != modComm or not flag:
			flag = new_modComm == modComm
			tmp = new_modComm
			new_modComm = reStr.sub(" ", modComm)
			modComm = tmp
		
		'''
		modComm_lst = modComm.split()
		if len(modComm_text) > 0:
			new_modComm_lst = []
			new_modComm_text = []
			new_modComm_tags = []
			new_modComm_lemma = []
			for idx in range(len(modComm_lst)):  # new_modComm_lst, modComm_text, modComm_tags have the same dimension
				if modComm_text[idx] not in stopwords:
					new_modComm_lst.append(modComm_lst[idx])
					new_modComm_text.append(modComm_text[idx])
					new_modComm_tags.append(modComm_tags[idx])
					new_modComm_lemma.append(modComm_lemma[idx])
			modComm_text = new_modComm_text
			modComm_tags = new_modComm_tags  # Save tags for later
			modComm_lemma = new_modComm_lemma
		else:
			new_modComm_lst = [x for x in modComm_lst if x.split('/')[0] not in stopwords]
		modComm = " ".join(new_modComm_lst)
		#print('TODO')
		'''
	
	if 8 in steps:  # lemmazation,
		# As piazza, 689 would be executed together.
		# So it must have tags
		# As we almost lemmazation all words, then it's actually useless to use regex here.
		# Warning! stopwords may or maynot be removed!
		
		if modComm == "":
			return ""
		
		modComm_lst = modComm.split()
		new_modComm_lst = []
		'''
		
		Yes, the stopwords may or may not removed but it won't be reduced.
		If a word happens n times, it will still be n times or zero times.
		And the order of all words keep the same.
		So I use the word in modComm, scan from the left to the right
		'''
		j = 0
		for i in range(len(modComm_lst)):
			pieces = modComm_lst[i].split("/")
			new_modComm_tag = pieces[-1]
			new_modComm_text = "/".join(pieces[:len(pieces) - 1])
			if j >= len(doc):
				print("ERROR! j out of bound in step 8. Result bypass!")
				break
			while not new_modComm_text == doc[j].text:
				debugger = doc[j].text
				j += 1
				if j >= len(doc):
					print("ERROR! j out of bound in step 8. Result bypass!")
					break
			# if j ends when i not, there would be a critical error!
			# Now I find the correct place
			token = doc[j]
			if token.lemma_[0] == '-' and token.text[0] != '-':  ##curcumstance to keep token
				new_modComm_lst.append(modComm_lst[i])  # It doesn't change at all!
			# elif token.lemma_[0] == token.text[0].lower():
			#    new_modComm_lst.append(token.text[0] + token.lemma_[1:] + "/" + new_modComm_tag)
			# Now we have word lists lemmaed.
			else:
				new_modComm_lst.append(token.lemma_ + "/" + new_modComm_tag)
			j += 1
		
		modComm = " ".join(new_modComm_lst)
	
	if 9 in steps:  # new line between sentences 4.2.4.
		# print('TODO')  # Mr./NNP Guardian/NNP Council/NNP .../: \n and/CC Supreme/NNP
		# The code in comment was 4.2.4. However... as 689 together, we can use tags...
		if modComm == "":
			return ""
		
		modComm = re.sub(r"(\S\/\.)(\s|$)", r"\1 \n\2", modComm)
		'''
		tag_scanner = r"\s*" #Handle tag exists or not
		if len(modComm_lst) == 0:
			print("ERROR here!")
		if len(modComm_lst[0].split('/')) == 2:  # Then there would be tags:
			tag_scanner = r"\/\S+\s*"
		#Place putative newline
		new_modComm = re.sub(r"([\.\?\!\;\:])(" + tag_scanner + r")", r"\1\2 \n ", modComm)
		#push newline after the following quotes
		new_modComm = re.sub(r"\n(\s*[\'\"])(" + tag_scanner + r")", r"\1\2 \n ", new_modComm)
	
		def abbrCheckForName(matched):
			word = matched.group().strip()
			word_lst = word.split()
			if len(word_lst) != 2:
				print("Step 9 Error in preprocess: request format: Prof. Rudzicz, actual word: " + word)
			if word_lst[0].split('/')[0] in pn_abbrs:
				return " " + word_lst[0] + " " + word_lst[1]
			elif ord(word_lst[1]) >=97 and ord(word_lst[1]) <= 122:
				return " " + word_lst[0] + " " + word_lst[1]
			else:
				return matched.group()
		#known abbr + . + Name/noneCapital?
		#Yes, I am speaking to you two: e.g. and i.e.
		new_modComm = re.sub(r"((\w+\.)+)" + r"(" + tag_scanner + r")" + r"\n(\s*\S)", abbrCheckForName, new_modComm)
		#Disqualify newline with ?! when following lowercase
		new_modComm = re.sub(r"([\?\!\:\;])" + r"(" + tag_scanner + r")" + r"\n(\s*^[A-Z])", r"\1\2\3", new_modComm)
		#Special case: ... word
		new_modComm = re.sub(r"(\s+...)" + r"(" + tag_scanner + r")" + r"\n(\s*[a-z])", r"\1\2\3", new_modComm)
	
		modComm = new_modComm
		'''
	if 10 in steps:  # lowercase
		
		if modComm == "":
			return ""
		
		
		# modComm = modComm.lower()
		def helper_rep(matched):
			# group: something/xx, group0: something/xx group1:something, group2:/xx
			return matched.group(1).lower() + matched.group(2)
		
		
		if 8 not in steps:
			modComm = re.sub(r"(\S+)(\/\-?\w+\$?\-?)", helper_rep, modComm)
	
	return modComm


def main(args):
	allOutput = []
	for subdir, dirs, files in os.walk(indir):
		for fl in files:
			fullFile = os.path.join(subdir, fl)
			print("Processing " + fullFile)
			
			data = json.load(open(fullFile))
			
			# TODO: select appropriate args.max lines
			if args.max > 10000:
				args.max = 10000
			# TODO: read those lines with something like `j = json.loads(line)`
			startpt = args.ID[0] % len(data)
			mydata = data[startpt: startpt + args.max]
			# TODO: choose to retain fields from those lines that are relevant to you
			# TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
			counter = 0
			for line in tqdm(mydata):
				# print(counter)
				# if counter == 5209:
				#    print(1)
				counter = counter + 1
				# if counter % 100 == 0:
				#    print("file {} counter {}".format(fl, counter))
				j = json.loads(line)
				newline = {}
				newline['id'] = j['id']
				newline['cat'] = fl
				
				# TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
				# TODO: replace the 'body' field with the processed text
				newline['body'] = preproc1(j['body'])
				allOutput.append(newline)
		# TODO: append the result to 'allOutput'
	
	fout = open(args.output, 'w')
	fout.write(json.dumps(allOutput))
	fout.close()


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Process each .')
	parser.add_argument('ID', metavar='N', type=int, nargs=1,
	                    help='your student ID')
	parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
	parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
	args = parser.parse_args()
	
	if (args.max > 200272):
		print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
		sys.exit(1)
	
	main(args)
