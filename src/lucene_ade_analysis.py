import sys
import lucene
from functools import reduce

import nltk.data
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
import pandas as pd

from sklearn_x.metrics import PerformanceMetrics
from sklearn_x import printers
from sklearn.metrics import confusion_matrix

from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import IndexReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version

# GLOBAL SCRIPT CONSTANTS
# direcotry containing lucene index files
INDEX_DIR='/data/index'
# input tweet files
TWEET_DATA_FILES = ['YOUR FILE NAMES HERE']
TWEET_DATA_FILE = TWEET_DATA_FILES[0]
# output file for storage
OUTPUT_FILE = '/data/output/lexicon_performance.txt'

lucene_initialized = False

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

def preprocess_text(text):
	toks = tokens(text)
	stems = filter_stop_words(toks)
	stems = porter_stem(stems)
	return reduce(lambda x,y: '{0} {1}'.format(x,y), stems, '')

def init_lucene(is_lucene_initialized=lucene_initialized):
	if not is_lucene_initialized:
		lucene.initVM()
		lucene_initialized = True

def create_searcher():
	init_lucene()
	reader = IndexReader.open(SimpleFSDirectory(File(INDEX_DIR)))
	return IndexSearcher(reader)

def create_query_parser():
	init_lucene()
	analyzer = StandardAnalyzer(Version.LUCENE_4_10_1)
	return QueryParser(Version.LUCENE_4_10_1, "text", analyzer)

def query_lucene(query_text, query_parser, searcher):	
	query = query_parser.parse(query_text)
	MAX = 1000
	return searcher.search(query, MAX)

def eval(tweet_data, false_positive_concept_dict = {}, false_negatives_span_dict = {},
         correctly_identified_ADRs = 0, predicted_ADRs = 0, actual_ADRs = 0):

    data_dict = {}
    for _, row in tweet_data.iterrows():
        id = row['id']
        semtype = row['semantic_type'].lower()
        span = row['span']
        tweet = row['text']
        if semtype in ['adr']:
            val = data_dict.get(id, (tweet, None))
            prev_span = val[1]
            if not prev_span:
                data_dict[id] = (tweet, span)
            else:
                data_dict[id] = (tweet, '{0}|{1}'.format(prev_span, span))

    for id, value in data_dict.iteritems():
        #id = row['id']
        #semantic_type = row['semantic_type'].lower()
        span_text = str(value[1]).lower()
        if '|' in span_text:
            spans = span_text.split('|')
        else:
            spans = [span_text]
        spans = [preprocess_text(s.strip()) for s in spans]
        actual_ADRs += len(spans)
        span_already_labeled = [False for _ in spans]
        all_span_tokens = [s.split(" ") for s in spans]
        tweet = value[0].decode('utf8').lower()
        tweet = preprocess_text(tweet.strip())
        tweet_tokens = tweet.split(" ")
        hits = query_lucene(tweet, qp, searcher).scoreDocs
        if not hits:
            false_negatives_span_dict[id]=(spans, tweet)
        else:
            # approx_match_already_found = False
            for hit in hits:
                doc = searcher.doc(hit.doc)
                doc_text = doc.get('text').encode('utf-8')
                concept_id, concept_text = doc_text.split('\t')
                concept_text = concept_text.strip().lower()
                concept_tokens = concept_text.strip().split(" ")

                # first check if concept is completely contained in the tweet
                tweet_match = True
                for ct in concept_tokens:
                    if not ct in tweet_tokens:
                        tweet_match = False
                # check concept against each span for approximate match
                if tweet_match:
                    # use this variable to track index of matched spans
                    matched_spans = []
                    # check for concept match against any ADR span
                    for span_idx, span_tokens in enumerate(all_span_tokens):
                        approx_match = False
                        for ct in concept_tokens:
                            if ct in span_tokens:
                                approx_match = True
                        if approx_match:
                            matched_spans.append(span_idx)
                    # score concept hit based on span match
                    if not matched_spans:
                        # no span match for this concept - False Postivie
                        predicted_ADRs += 1
                        tweet_fps = false_positive_concept_dict.get(id, ([],tweet))
                        concept_list = tweet_fps[0]
                        concept_list.append((concept_id, concept_text))
                        false_positive_concept_dict[id] = (concept_list, tweet)
                    else:
                        new_span_match_found = False
                        for idx in matched_spans:
                            sal = span_already_labeled[idx]
                            if not (new_span_match_found or sal):
                                new_span_match_found = True
                                span_already_labeled[idx] = True
                                correctly_identified_ADRs += 1
                                predicted_ADRs += 1
            for bidx, b in enumerate(span_already_labeled):
                if not b:
                    tweet_fn = false_negatives_span_dict.get(id, ([],tweet))
                    missed_span_list = tweet_fn[0]
                    missed_span_list.append(spans[bidx])
                    false_negatives_span_dict[id] = (missed_span_list, tweet)

    return (false_positive_concept_dict, false_negatives_span_dict,
            correctly_identified_ADRs, predicted_ADRs, actual_ADRs)

if __name__ == '__main__':
    tweet_data = pd.read_csv(TWEET_DATA_FILES[0])
    searcher = create_searcher()
    qp = create_query_parser()

    false_positive_concept_dict, false_negatives_span_dict, \
            correctly_identified_ADRs, predicted_ADRs, actual_ADRs = eval(tweet_data)

    for f in TWEET_DATA_FILES[1:]:
        tweet_data = pd.read_csv(f)
        false_positive_concept_dict, false_negatives_span_dict, \
            correctly_identified_ADRs, predicted_ADRs, actual_ADRs = eval(tweet_data, false_positive_concept_dict,
                                                                          false_negatives_span_dict,correctly_identified_ADRs,
                                                                          predicted_ADRs, actual_ADRs)


    precision = correctly_identified_ADRs / float(predicted_ADRs)
    recall = correctly_identified_ADRs / float(actual_ADRs)
    fmeasure = 2 * precision * recall / (precision + recall)
    printers.printsf('ADR spans actual: {0}'.format(actual_ADRs), OUTPUT_FILE)
    printers.printsf('ADR spans predicted: {0}'.format(predicted_ADRs), OUTPUT_FILE)
    printers.printsf('ADR spans correctly identified: {0}'.format(correctly_identified_ADRs), OUTPUT_FILE)
    printers.printsf('Precision: {0}'.format(precision), OUTPUT_FILE)
    printers.printsf('Recall: {0}'.format(recall), OUTPUT_FILE)
    printers.printsf('F1-Score: {0}'.format(fmeasure), OUTPUT_FILE)

    printers.printsf('\n\n{0}False Positive Concepts{0}\n'.format(50*'-'), OUTPUT_FILE)
    for tweet_id,v in false_positive_concept_dict.iteritems():
        concept_tpl_list, tweet = v
        line = ''
        for tpl in concept_tpl_list:
            cid, ct = tpl
            line += '{0}\t{1}\t{2}\t{3}\n'.format(cid, ct, tweet_id, tweet)
        printers.printsf(line, OUTPUT_FILE)

    printers.printsf('\n\n{0}False Negatives - Missed Spans{0}\n'.format(50*'-'), OUTPUT_FILE)
    for tweet_id, v in false_negatives_span_dict.iteritems():
        span_list, tweet = v
        line = ''
        for span in span_list:
            line += '{0}\t{1}\t{2}\n'.format(tweet_id, span, tweet)
        printers.printsf(line, OUTPUT_FILE)


