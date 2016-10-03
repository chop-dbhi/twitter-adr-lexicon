import nltk.data
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from functools import reduce

# GLOBAL SCRIPT CONSTANTS
# file to be preprocessed, assumed format is tab delimited rows where 
# first column is Ontology concept ID, second column is concept
ADE_FILE = ''
# output file name for file to be used in index lucene index generation
ADE_INDEX_FILE = ''

def tokens(text, splitContractions=False, contractChars = ["'"]):
    '''uses a regexpTokenizer to tokenize text to words. If splitContractions is true,
    the regex pattern is [\w]+ so that contractions are split, e.g. "I can't" -> ['I','can','t'],
    otherwise the regex pattern is [\w']+ so that contractions are not split, i.e. "I can't" -> ['I', "can't"]
    Additional contract characters, e.g. a hyphen, can be added by over riding the contractChars arg'''
    if splitContractions:
        pat = "[\w]+"
    else:
        pat = "[\w{0}]+".format(reduce(lambda x,y: x+y, contractChars, ""))
    return regexp_tokenize(text, pat, discard_empty=True)

def porter_stem(wordList):
    porter = nltk.PorterStemmer()
    return [porter.stem(w) for w in wordList]

def filter_stop_words(wordlist):
    sw = stopwords.words('english')
    return filter(lambda w: w not in sw, wordlist)

if __name__ == '__main__':
	lines_out = []
	with open(ADE_FILE, 'r') as fin:
		for line in fin.readlines():
			s = line.split('\t')
			t = s[1]
			toks = tokens(t)
			toks = porter_stem(toks)
			filtered_toks = filter_stop_words(toks)
			nl = reduce(lambda x,y: '{0} {1}'.format(x,y), filtered_toks, '{0}\t'.format(s[0]))
			if not nl in lines_out:
				lines_out.append(nl)
	with open(ADE_INDEX_FILE,'a') as fout:
		for line in lines_out:
			fout.write('{0}\n'.format(line))
