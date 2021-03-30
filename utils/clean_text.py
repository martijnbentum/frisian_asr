from texts.models import Text
from collections import Counter
from . import number2word
import string
from . import language_detection as ld
import re

try:
	import ucto
	tokenizer = ucto.Tokenizer("tokconfig-nld",lowercase = True,paragraphdetection=False)
except:
	print('could not load ucto, please activate lamachine?')
	tokenizer = ''


n2w = number2word.Number2word()


class Cleaner:
	def __init__(self,text):
		self.text = text
		tokenizer.process(text)
		self.words = list(tokenizer)
		self.words_str = [str(x) for x in self.words]
		tokenizer.process(text)
		self.sentences = list(tokenizer.sentences())
		self.word_freq = Counter(self.words_str)
		self.suspect_words = []
		self.passed_words = []
		self.cs = ld.load('Dutch-Frisian_sentences')
		self.clean()

	def clean(self, convert_number= True,split_on_comma = True, remove_punctuation = True,
			check_suspect = True, remove_suspect_words= False):
		o = []
		for i,word in enumerate(self.words):
			if word.tokentype == 'NUMBER':
				if convert_number:
					context = get_context(self.words,i)
					language = self.cs.predict_text(context).main_language_overall_prediction
					number = n2w.toword(word.text,language =language)
					print(language,number)
					o.append(number)
				else: o.append(w.text)
			elif word.tokentype == 'PUNCTUATION':
				if word.text == ',' and not split_on_comma: pass
				elif remove_punctuation: 
					if word.text in ',.?:;!':o.append('\n')
					else:self.passed_words.append(word.text)		
				else: o.append(word.text)
			else: 
				if check_suspect or remove_suspect_words:
					suspect = self._check_suspect_word(word)
					if remove_suspect_words: continue
				o.append(word.text)
		self.clean_text = re.sub(' +\n +','\n',' '.join(o))

	def _check_suspect_word(self,word):
		for char in string.punctuation:
			if char in word.text:
				self.suspect_words.append(word.text)
				return True
		return False
			
		


		
def get_context(words,i,size = 10):
	before = words[:i]
	after = words[i:]
	if len(before) > size : before = before[size*-1:]
	if len(after) > size : after = after[:size]
	words = [word.text for word in before + after]
	return ' '.join(words)
			


def simple_clean(text):
	assert tokenizer != ''
	tokenizer.process(text)
	return '\n'.join(list(tokenizer.sentences()))


