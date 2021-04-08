'''adapted from:
https://towardsdatascience.com/an-efficient-language-detection-model-using-naive-bayes-85d02b51cfbd
'''


from texts.models import Text
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import time
import copy
import glob
import pickle

class LangDetect:
	'''class to train language detection model based on naive bayes.
	number of languages is optional tested on frisian and dutch
	can train a model on sentences or word level (expects a list of sentences 
	or a list of words in the ll attribute (might need to change the data input
	for this, because now language are a list of transcriptions (see manual_transcriptions)
	the ngram range set the which ngram to include 2 - 5 seems to be optimal with current
	dataset.
	'''
	def __init__(self,languages = [], language_names = [], input_type = 'sentences',
		test_size = 0.33, ngram_range = (2,5)):
		if languages == []:
			self.languages= list(get_dutch_frisian_transcriptions())
			self.language_names = ['Dutch','Frisian']
		else: self.languages, self.language_names = languages,language_names
		self.input_type = input_type
		if input_type == 'sentences':func = make_sentences
		if input_type == 'words': func = make_words
		self.ll = [func(l) for l in self.languages]
		self.test_size = test_size
		self.ngram_range = ngram_range
		self._make_xy()
		self.training_prepped = False
		self.trained = False

	def _make_xy(self):
		x,self.ll_names = [], []
		for l,name in zip(self.ll,self.language_names):
			x.extend(l)
			self.ll_names.extend([name]*len(l))
		self.x = np.array(x)
		self.y = np.array(self.ll_names)

	def make_train_test(self,test_size=None):
		if test_size and type(test_size) == float: self.test_size = test_size
		temp = train_test_split(self.x, self.y, test_size = self.test_size, 
			random_state = 9)
		self.x_train, self.x_test, self.y_train, self.y_test = temp

	def set_ngram(self,ngram_range = None):
		if ngram_range and type(ngram_range) == tuple: self.ngram_range= ngram_range
		self.cnt = CountVectorizer(analyzer = 'char',ngram_range=self.ngram_range)

	def prep_train(self, test_size = None, ngram_range=None):
		self.make_train_test(test_size)
		self.set_ngram(ngram_range)
		self.training_prepped = True

	def train(self, test_size = None, ngram_range = None):
		if not self.training_prepped: self.prep_train(test_size,ngram_range)
		self.pipeline = Pipeline([
			('vectorizer',self.cnt),
			('model',MultinomialNB())
		])
		self.pipeline.fit(self.x_train,self.y_train)
		self.y_pred = self.predict(self.x_test)
		self.cm = confusion_matrix(self.y_test,self.y_pred)
		self._report = classification_report(self.y_test,self.y_pred)
		self.correctness = Correctness(self.x_test,self.y_test,self.y_pred)
		print(self._report)
		self.trained = True

	def predict(self, test):
		'''predicts new texts, test should be a list of sentences or words,
		dependent on the input_type training setting.'''
		if type(test) == str: test = [test]
		self.pred = self.pipeline.predict(test)
		return self.pred

	def predict_text(self,text):
		'''predicts a texts (assumes sentence level training)
		splits on \n, should be made more flexible to handle word level as well
		tested for sentences and works well'''
		pred = self.predict(text.split('\n'))
		all_pred = self.predict(text)
		self.output = Output(text,pred,self.language_names,all_pred)
		return self.output

	@property
	def report(self):
		'''print the classification report if it is available.'''
		if hasattr(self,'_report'): print(self._report)

	def save(self,filename= None,
		directory = '/vol/tensusers/mbentum/FRISIAN_ASR/LANGUAGE_DETECTION_MODELS/'):
		'''save the current model to disk. Uses pickle'''
		if not self.trained: 
			print('model not trained, nothing to save')
			return False
		if not filename: 
			filename = '-'.join(self.language_names) +'_'+self.input_type
		print('saving to ',directory+filename)
		output = copy.deepcopy(self)
		for attr in 'x,y,x_train,y_train,x_test,y_test,y_pred,languages'.split(','):
			setattr(output,attr,None)
		with  open(directory + filename,'wb') as fout:
			pickle.dump(output,fout)

def load(filename,directory = '/vol/tensusers/mbentum/FRISIAN_ASR/LANGUAGE_DETECTION_MODELS/'):
	'''load a pretraining langdetection model from disk'''
	if '/' in  filename or not directory: f = filename
	else: f = directory + filename
	print('loading model: ',f)
	with open(f,'rb') as fin:
		return pickle.load(fin)

def show_models(directory = '/vol/tensusers/mbentum/FRISIAN_ASR/LANGUAGE_DETECTION_MODELS/'):
	fn = [f.split('/')[-1] for f in glob.glob(directory + '*')]
	print('\n'.join(fn))


class Output:
	'''handle the output of .predict_text of the LangDetect class.
	seperates the Dutch and Frisian sentences. Word level -> work in progress.'''
	def __init__(self,text,predictions,language_names,overall_prediction = ['']):
		self.text = text
		self.predictions = predictions
		self.language_names = language_names
		# the language determined by .predict the whole text
		self.main_language_overall_prediction = overall_prediction[0]
		self.handle_pred()

	def __repr__(self):
		m = 'prediction output | '  
		m += ' | '.join([name + ' ' + str(getattr(self,name + '_count')) for name 
			in self.language_names])
		m += ' | main language: ' + self.main_language_overall_prediction
		return m


	def handle_pred(self):
		self._set_attributes()
		self._group_sentence_by_language()
		self._majority_language()

	def _set_attributes(self):
		for name in self.language_names:
			setattr(self,name,[])

	def _group_sentence_by_language(self):
		'''the sentences are grouped in a list on an attribute of the language name e.g. Dutch.'''
		sentences = self.text.split('\n')
		for sentence, pred in zip(sentences,self.predictions):
			if not sentence: continue
			getattr(self,pred).append(sentence)

	def _majority_language(self):
		'''counts the number of sentences for each language and sets majority language to
		the language with most sentences.'''
		self.majority_language = ''
		self.count = 0
		for name in self.language_names:
			count = len(getattr(self,name)) 
			setattr(self,name + '_count',count)
			if count > self.count: 
				self.count = count
				self.majority_language = name

	@property
	def found_languages(self):
		o = []
		for language in self.language_names:
			if len(getattr(self,language)) > 0: o.append(language)
		return o

			
		
def exclude_overlap_words(correctness,overlap):
	c = correctness
	count = 0
	test, pred = [] , []
	for x_test, y_test, y_pred in zip(c.x_test,c.y_test,c.y_pred):
		if x_test in overlap:
			count += 1
			continue
		test.append( y_test )
		pred.append( y_pred )

	print('before:')
	print( confusion_matrix(c.y_test,c.y_pred))
	print( classification_report(c.y_test,c.y_pred))
	print('after overlap words removal:')
	print( confusion_matrix(test,pred))
	print( classification_report(test,pred))
	print('removed:',count,'words from evaluation (these words occur in both dutch and frisian)')
	return test, pred
		



class Correctness:
	def __init__(self,x_test,y_test,y_pred):
		self.x_test = x_test
		self.y_test = y_test
		self.y_pred = y_pred
		self.labels = list(set(y_test))
		self._set_correctness()
		self.nitems = len(x_test)

	def __repr__(self):
		m = 'Correctness class | N items ' + str(self.nitems) + ' | N classes ' 
		m += str(len(self.labels))
		return m

	def _set_attributes(self):
		for label in self.labels:
			setattr(self,label +'_correct',[])
			setattr(self,label +'_incorrect',[])
		
	def _set_correctness(self):
		self._set_attributes()
		self.correctness_dict = {}
		for text, test_label, pred_label in zip(self.x_test,self.y_test,self.y_pred):
			if test_label == pred_label: getattr(self,test_label+'_correct').append(text)
			if test_label != pred_label: getattr(self,test_label+'_incorrect').append(text)
				

def get_texts(language= 'Dutch'):
	t =Text.objects.filter(text_type__name = 'manual transcription', 
		multiple_languages = False, main_language__name=language)
	return t

def get_transcriptions(language= 'Dutch'):
	t = get_texts(language)
	output = []
	for x in t:
		transcription = x.transcription
		if not transcription or type(transcription) == str: continue
		if not transcription.code_switched: output.append(transcription)
	return output

def get_dutch_frisian_transcriptions():
	dutch = get_transcriptions()
	frisian = get_transcriptions('Frisian')
	return dutch, frisian

def make_sentences(transcriptions):
	output = []
	for x in transcriptions:
		temp = x.text_without_tags
		if temp: output.append(temp)
	return output

def make_words(transcriptions):
	sentences = make_sentences(transcriptions)
	return ' '.join(sentences).split(' ')



