import os
import sys
import lucene
 
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
 
# GLOBAL SCRIPT CONSTANTS
# directory to store index files
INDEX_DIR='/data/index'

# input files from which to construct index
INPUT_DIR = 'YOUR INPUT DIRECTORY HERE'
# see also, ade_index_preprocess.py to generate these input files
FILES = ['YOUR INPUT FILES HERE']
INPUT_FILES = [os.path.join(INPUT_DIR,f) for f in FILES]


if __name__ == "__main__":
	lucene.initVM()
	indexDir = SimpleFSDirectory(File(INDEX_DIR))
	writerConfig = IndexWriterConfig(Version.LUCENE_4_10_1, StandardAnalyzer())
	writer = IndexWriter(indexDir, writerConfig)
  
	# write to index from input files
	line_count = 0
	for fn in INPUT_FILES:
		with open(fn,'r') as fin:
			for l in fin.readlines():
				line_count += 1
				doc = Document()
				doc.add(Field("text", l, Field.Store.YES, Field.Index.ANALYZED))
				writer.addDocument(doc)
	print("Indexed {0} lines from input files".format(line_count))
	writer.close()
