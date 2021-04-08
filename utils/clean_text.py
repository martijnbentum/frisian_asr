from texts.models import Text
from collections import Counter
from . import number2word
from . import lexicons
import string
from . import language_detection as ld
import re

'''ucto is a tokenizer and part of lamachine environment, can only be loaded if lamachine is active'''
try:
	import ucto
	tokenizer = ucto.Tokenizer("tokconfig-nld",lowercase = True,paragraphdetection=False)
except:
	print('could not load ucto, please activate lamachine?')
	tokenizer = ''

#object to map digits 11 to words elf (in dutch or frisian
n2w = number2word.Number2word()
dutch,frisian = lexicons.make_dutch_and_frysian_lexicon()
#lexicon with dutch and frisian lemma's to decide whether to include a dash word i.e e-mail
lexicon = set(frisian.words +dutch.words)


class Cleaner:
	'''clean a text with ucto
	numbers are mapped to written out words'''
	def __init__(self,text,remove_number_compounds =True):
		'''text 					text to clean
		remove_number_compounds 	whether 2013-2301 should be mapped to 2013 2301
		'''
		self.text = text
		if remove_number_compounds: text = re.sub('(\d+)-(\d+)-(\d+)',r'\1 \2 \3',text)
		if remove_number_compounds: text = re.sub('(\d+)-(\d+)',r'\1 \2',text)
		tokenizer.process(text)
		self.words = list(tokenizer)
		self.words_str = [str(x) for x in self.words]
		tokenizer.process(text)
		self.sentences = list(tokenizer.sentences())
		self.word_freq = Counter(self.words_str)
		self.suspect_words = []
		self.passed_words = []
		self.cs = ld.load('Dutch-Frisian_sentences') # classifier to determine language of numbers
		self.token_types = {} # for debug purposes
		self.clean()

	def clean(self, convert_number= True,split_on_comma = True, remove_punctuation = True,
			check_suspect = True, remove_suspect_words= False):
		'''steps to clean a text.
		ucto tokenizes a text into tokens
		these tokens have tokentypes, based on the token types a specific action is taken
		'''
		o = []
		for i,word in enumerate(self.words):
			if word.text == '': continue # empty words are skipped
			'''create a dictionary with word token types to store types and their instances'''
			if word.tokentype not in self.token_types.keys(): self.token_types[word.tokentype] = []
			'''do not store word instances in the token type dictionary'''
			if word.tokentype != 'WORD' and word.text not in self.token_types[word.tokentype]: 
				self.token_types[word.tokentype].append(word.text)
			'''if a token is a number try to map it to a written out form'''
			if word.tokentype == 'NUMBER':
				if convert_number:
					context = get_context(self.words,i)
					language = self.cs.predict_text(context).main_language_overall_prediction
					number = n2w.toword(word.text,language =language, spaces = True)
					o.append(number)
				else: o.append(w.text)
			# '''punctuation is typically skipped; a comma can be mapped to eos'''
			elif word.tokentype in ['PUNCTUATION','SYMBOL','PUNCTUATION-MULTI']:
				if word.text == ',' and not split_on_comma: pass
				elif remove_punctuation: 
					if word.text in ',.?:;!':o.append('\n')
					else:self.passed_words.append(word.text)		
				else: o.append(word.text)
			# '''a word is checked whether it contains punctuation charachter which can be skipped'''
			elif 'WORD' in word.tokentype: 
				if check_suspect or remove_suspect_words:
					suspect = self._check_suspect_word(word)
					if remove_suspect_words: continue
				'''if word have a dash, check whether it is already in the lexicon 
				otherwise remove dash'''
				if '-' in word.text and word.text not in lexicon: 
					if len(word.text) == 1: continue
					if word.text.endswith('-') or word.text.startswith('-'):
						word.text = word.text.replace('-','')
					else:word.text = word.text.replace('-',' ')
				o.append(word.text)
		#remove extra whitespace and some remaining characters
		self.clean_text = re.sub('\s+\n\s+','\n',' '.join(o))
		self.clean_text = re.sub('\n+','\n',self.clean_text)
		self.clean_text = re.sub('\|','',self.clean_text)
		self.clean_text = re.sub('\(','',self.clean_text)
		self.clean_text = re.sub('\)','',self.clean_text)
		self.clean_text = re.sub('\+','',self.clean_text)

	def _check_suspect_word(self,word):
		'''check whether a word contains punctuation characters (except -).'''
		for char in string.punctuation:
			if char == '-': continue
			if char in word.text:
				self.suspect_words.append(word.text)
				return True
		return False

			
def get_context(words,i,size = 10):
	'''gets the words surround a the word at i index, size sets the number before and after'''
	before = words[:i]
	after = words[i:]
	if len(before) > size : before = before[size*-1:]
	if len(after) > size : after = after[:size]
	words = [word.text for word in before + after]
	return ' '.join(words)

def simple_clean(text):
	'''minimal clean with just tokenization done by ucto.'''
	assert tokenizer != ''
	tokenizer.process(text)
	return '\n'.join(list(tokenizer.sentences()))


