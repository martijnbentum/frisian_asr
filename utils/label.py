from texts.models import Text
from . import language_detection as ld
from . import lexicons
from sklearn.metrics import confusion_matrix,classification_report

from collections import Counter
import os
import re
import sys


def flatten(l):
	o = []
	for line in l:
		for item in line:
			o.append(item)
	return o


def eval_labeler():
	'''test the quality of the language labeler, however uses all available material
	also material used for training the word and sentence language classifiers
	might be an idea to improve on this by selecting test materials from the 
	classfiers and adding the code switch materials from the manually transcribed
	materials; alternatively test it on the second batch.
	'''
	texts = Text.objects.filter(text_type__name='manual transcription')
	l = Label()
	pred,test = [],[]
	for i,x in enumerate(texts):
		x.transcription
		ok = l.label(x.transcription.text_without_tags.replace('-',''))
		if not ok: continue
		l.extract_labels()
		pred.append(l.word_labels_sentences[0])
		test.append(x.transcription.tags_list)
	predf = flatten(pred)
	testf = flatten(test)
	cm = confusion_matrix(testf,predf)
	report = classification_report(testf,predf)
	print(report)
	return test,pred


class Label:
	'''Labels words in a text either dutch or frisian
	assumes cleaned tokenized text
	first pass a sentence is classified with the sentence classifier
	second pass each word not in the overlap set is classified with the word classifier
	the overlap consists of words present in dutch and frisian (based on emre's final lexicon
	words in the overlap set are given the sentence language label
	'''
	def __init__(self):
		self.classifier_sentences = ld.load('Dutch-Frisian_sentences')
		self.classifier_words= ld.load('Dutch-Frisian_words')
		self.overlap = lexicons.get_overlap_wordset()

	def label(self, text, extract_labels = False):
		self.nfrisian_words = 0
		self.ndutch_words = 0
		if text == '': return False
		self.sentences = text.split('\n')
		self.pred_sentences = self.classifier_sentences.predict(self.sentences)
		self.sentences_words = []
		self.pred_sentences_words = []
		for sentence in self.sentences:
			words = sentence.split(' ')
			self.sentences_words.append(words)
			self.pred_sentences_words.append(self.classifier_words.predict(words))
		self.labelled_sentences = []
		items = zip(self.sentences_words,self.pred_sentences_words,self.pred_sentences)
		d = {'Frisian':'-fr','Dutch':'-nl'}
		for sentence_words, pred_sentence_words, pred_sentence in items: 
			temp = [] 
			for word, pred in zip(sentence_words,pred_sentence_words):
				if word == '': continue
				if word in self.overlap: temp.append( word + d[pred_sentence] )
				else: temp.append( word + d[pred] )
				if pred == 'Frisian':self.nfrisian_words +=1
				elif pred == 'Dutch': self.ndutch_words +=1
			self.labelled_sentences.append(' '.join(temp))
		if extract_labels: self.extract_labels()
		self.labelled_text = '\n'.join([s for s in self.labelled_sentences if s])
		return True
	
	def extract_labels(self):
		'''create a set of language label for each word in a sentence, for each sentence
		in the text. Each sentence is represente by a list of language labels.
		'''
		self.dutch, self.frisian = [], []
		self.word_labels_sentences = []
		for sentence in self.labelled_sentences:
			temp = sentence.split(' ')
			self.dutch.append( [w.split('-')[0] for w in temp if w.split('-')[1] == 'nl'])
			self.frisian.append( [w.split('-')[0] for w in temp if w.split('-')[1] == 'fr'])
			self.word_labels_sentences.append([w.split('-')[1] for w in temp])

class Unlabel:
	'''remove language labels from a text.
	assumes - only occurs for language labels
	assumes -language_label is attached at the and of the word e.g. hello-eng
	'''
	def __init__(self,text = '',filename = '', seperator = '-'):
		if filename != '' and os.path.isfile(filename):
			with open(filename) as fin:
				text = fin.read()
		self.text = text
		self.seperator = seperator
		self._process()


	def __repr__(self):
		m = 'unlabeller | nwords:' + str(self.nwords) + ' | labels: ' + self.labels_str
		return m


	def _process(self):
		self.words = text2words(self.text)
		self.sentences = text2sentences(self.text)
		if self.seperator in self.text:
			self.labels = Counter([w.split(self.seperator)[-1] for w in self.words if '-' in w])
			self.unlabelled_text = re.sub('-[a-zA-Z]+','',self.text)
		else:
			self.labels = ''
			self.unlabelled_text = self.text
		self.labels_str = str(self.labels).split('{')[-1].split('}')[0]

		self.nwords = len(self.words)
		self.nsentences= len(self.sentences)

			






def text2words(text):
	'''assumes text is cleaned and tokenized'''
	return text.replace('\n',' ').split(' ')

def text2sentences(text):
	'''assumes text is cleaned and tokenized'''
	return text.split('\n')
