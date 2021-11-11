import re
import statistics
from tqdm.auto import tqdm
import pickle
from collections import Counter
import os


'copied from file: /home/eyilmaz/main2/latest_ASR_exps/fame/data/local/dict/lexicon.txt'
f = '/vol/tensusers/mbentum/FRISIAN_ASR/LEXICONS/final_lexicon_emre.txt'
twente_f = '/vol/tensusers/mbentum/FRISIAN_ASR/LEXICONS/twente_lexicon.txt'
frisian_f = '/vol/tensusers/mbentum/FRISIAN_ASR/LEXICONS/Frysk.txt'
council_f = '/vol/tensusers/mbentum/FRISIAN_ASR/LEXICONS/lexicon_frisian_dutch_council.txt'
council_f_old = '/vol/tensusers/mbentum/FRISIAN_ASR/LEXICONS/lexicon_frisian_dutch_council_old.txt'
# council_cleaned_labelled = '/vol/tensusers/mbentum/FRISIAN_ASR/LM/council_notes_cleaned_labelled'

class Lexicon:
	def __init__(self,words=[],language='',prons= [],name = '', labelled = False,
		filename = ''):
		if filename:  
			self.text, words, prons, self.errors = read_lexicon_file(filename)
			self.filename = filename
		self.words= words 
		self.language = language
		self.prons= prons
		self.name = name
		self.labelled = labelled
		self._make_word2pron()
		self._make_wordpron()

	def _make_word2pron(self):
		self.word2pron = {}
		self.pron2word= {}
		if not self.prons or len(self.words) != len(self.prons): return False
		for i,word in enumerate(self.words):
			pron = self.prons[i]
			if word not in self.word2pron.keys(): self.word2pron[word] = []
			if pron not in self.pron2word.keys(): self.pron2word[pron] = []
			self.word2pron[word].append(pron)
			self.pron2word[pron].append(word)

	def _make_wordpron(self):
		self.Words = {}
		self.Prons = {}
		for word,pron in zip(self.words,self.prons):
			if word not in self.Words.keys():
				w = Word(word,labelled = self.labelled)
				self.Words[word] = w
			else: w = self.Words[word]
			if pron not in self.Prons.keys():
				p = Pronounciation(pron, labelled = self.labelled)
				self.Prons[pron] = p
			else: p = self.Prons[pron]
			w.add_pronounciation(p)
			p.add_word(w)

	def __repr__(self):
		m ='Lexicon: ' + self.language + ' ' + str(len(self.words))
		u = len(set(self.words)) 
		if len(self.words) == u:m += ' unique entries'
		else:m += ' entries ' + str(u)+  ' unique'
		if self.word2pron: m +=  ' | prons available'
		if self.name: m += ' | ' + self.name
		return m

	def overlap(self,other):
		assert type(self) == type(other)
		return list(set(self.words).intersection( set(other.words)))

	def unique(self,other):
		assert type(self) == type(other)
		return list(set(self.words) - set(other.words))


	def symmetric_unique(self,other):
		assert type(self) == type(other)
		return list(set(self.words).symmetric_difference(set(other.words)))
		

	@property
	def language_errors(self):
		return [w for w in self.Words.values() if w.language_error]

	def make_output(self, show_language= True, exclude_language_errors = True,
		select_language=''):
		words = self.Words.values() 
		print(len(words),'all words')
		if exclude_language_errors:
			words = [w for w in words if not w.language_error]
			print(len(words),'exclude_language_errors')
		if select_language:
			words = [w for w in words if w.language == select_language 
				or '<' in w.word]
			print(len(words),'select_language',select_language)
		output = []
		for w in words:
			output.extend(w.lexicon_entry(show_language = show_language))
		return '\n'.join(output)
		
		
	def save(self, filename = '', force_save = False, show_language =True, 
		exclude_language_errors=True, select_language = ''):
		if not filename: filename = self.filename
		if not force_save and os.path.isfile(filename):
			print(filename, 'already exists, use force_save to overwrite')
			print('doing nothing')
			return
		if not filename:
			print('please provide filename:',filename)
		o = self.make_output(show_language=show_language, 
			exclude_language_errors= exclude_language_errors,
			select_language = select_language)
		with open(filename,'w') as fout:
			fout.write(o)
		
		
		
		
	
	


class Word:
	def __init__(self,word,pronounciation=None,labelled = False):
		self.word_name = word
		self.language = 'unknown'
		self.word = word
		self.prons = []
		self.pron_doubles = []
		self.labelled = labelled
		if self.labelled and '-' in word: 
			self.language = word.split('-')[-1]
			self.word_name = '-'.join(word.split('-')[:-1]).strip('-')
		self.special_word = True if '-' not in word else False
		if pronounciation: self.add_pronounciation(pronounciation)

	def __repr__(self):
		m = self.word_name 
		if self.language != 'unknown':
			m+= ' | ' + self.language 
		m += ' | prons: ' + '   '.join([p.name() for p in self.prons])
		return m

	def __eq__(self,other):
		return self.word == other.word

	def add_pronounciation(self,pronounciation):
		if pronounciation not in self.prons: self.prons.append(pronounciation)
		else: self.pron_doubles.append(pronounciation)

	def name(self,show_language=True):
		return self.word if show_language else self.word_name

	def lexicon_entry(self, show_language = True):
		o = []
		for pron in self.prons:
			pron = pron.name(show_language = show_language)
			word = self.name(show_language)
			o.append(word + '\t' + pron)
		return o

	@property
	def language_error(self):
		for pron in self.prons:
			if pron.language != self.language: return True
		return False

		


class Pronounciation:
	def __init__(self,pron,word = None,labelled = False):
		self.pron = pron
		self.phonemes = [Phoneme(p, labelled) for p in pron.split(' ') if p]
		self.pron_name = ' '.join([p.phoneme_name for p in self.phonemes])
		self.words = []
		self.word_doubles = []
		self.labelled = labelled
		if word: self.add_word(word)
		self._set_language()

	def __repr__(self):
		m = self.pron + ' |  ' + self.language 
		m += ' | words: ' + ', '.join([w.name() for w in self.words])
		return m

	def __eq__(self,other):
		return self.pron == other.pron

	def _set_language(self):
		if self.labelled:
			languages = [p.language for p in self.phonemes]
			language_set = list(set(languages))
			l = ', '.join(language_set)
			self.language = l
			self._set_dominant_language(languages,language_set)
		else: self.language, self.dominant_language = 'unknown', 'unknown'
	

	def _set_dominant_language(self,languages,language_set):
		self.dominant_language = 'unknown'
		if len(language_set) == 1:
			self.dominant_language = language_set[0]
		highest = 0
		for language in language_set:
			n = languages.count(language)
			if n > highest:
				highest = n
				self.dominant_language = language
			
	def add_word(self,word):
		if word not in self.words: self.words.append(word)
		else: self.word_doubles.append(word)

	def name(self, show_language = True):
		return self.pron if show_language else self.pron_name
	


class Phoneme:
	def __init__(self, phoneme, labelled = False):
		self.phoneme_name = phoneme
		self.language = 'unknown'
		self.phoneme = phoneme
		self.labelled = labelled
		if self.labelled and '-' in self.phoneme:
			self.language = phoneme.split('-')[-1]
			self.phoneme_name = '-'.join(phoneme.split('-')[:-1]).strip('-')
		self.special_phoneme = True if not '-' in self.phoneme else False

	def __repr__(self):
		return self.phoneme
		
	def __eq__(self,other):
		return self.phoneme == other.phoneme
		
	def name(self,show_language):
		return self.phoneme if show_language else self.phoneme_name

			
				
				
			
			
			
		

def read_pronlex(filename):
	t = [line.split('\t') for line in open(filename).read().split('\n')]
	words = [line[0] for line in t if len(line) >= 2]
	prons = [line[1] for line in t if len(line) >= 2]
	errors= [line for line in t if len(line) < 2]
	return t,words, prons, errors

def read_lexicon_file(filename):
	text = open(filename).read().split('\n')
	words, prons, errors = [], [], []
	for i,line in enumerate(text):
		o = re.match('^\S*',line)
		if not o or not line:
			errors.append(line)
			continue
		words.append( o.group() )
		index = o.span()[1]
		prons.append( line[index:].strip() )
	return text, words, prons, errors


def council_lexicon():
	t, words, prons, errors = read_pronlex(council_f)
	return Lexicon(words,'Dutch, Frisian',prons,'council',labelled = True)

def council_lexicon_old():
	t, words, prons, errors = read_pronlex(council_f_old)
	return Lexicon(words,'Dutch, Frisian',prons,'council_old',labelled = True)



def read_frisian_dutch_lexicon(filename = f):
	print('reading lexicon from filename:',filename)
	t, words, prons, errors = read_pronlex(filename)
	language = []
	for w in words:
		if w.endswith('-nl'): language.append('Dutch')
		elif w.endswith('-fr'): language.append('Frisian')
		else: language.append('Unk')
	clean_words = [w[:-3] for w in words]
	clean_prons = re.sub('-nl','','\n'.join(prons))
	clean_prons = re.sub('-fr','',clean_prons).split('\n')
	return clean_words, clean_prons, language, t, words, prons,errors

def read_twente_lexicon():
	t, words, prons, errors = read_pronlex(twente_f)
	return words, prons, t, errors

def read_frisian_lexicon():
	t, words, prons, errors = read_pronlex(frisian_f)
	return words, prons, t, errors

		
def make_dutch_and_frisian_lexicon(lexicon_type = 'fame'):
	if lexicon_type == 'fame':filename = f
	elif lexicon_type == 'council': filename= council_f
	elif lexicon_type == 'council_old': filename= council_f_old
	else: raise ValueError(lexicon_type,'uknown use fame or council')
	print('creating lexicon:',lexicon_type)
	cw,cp,l,t,w,p,e = read_frisian_dutch_lexicon(filename)
	cwd,cwf,cpd,cpf = [],[],[],[]
	for word,pron,language in zip(cw,cp,l): 
		if language == 'Dutch':
			cwd.append(word)
			cpd.append(pron)
		elif language == 'Frisian':
			cwf.append(word)
			cpf.append(pron)
	dutch = Lexicon(cwd,'Dutch',cpd,lexicon_type)
	frisian= Lexicon(cwf,'Frisian',cpf,lexicon_type)
	return dutch, frisian


def make_twente_lexicon():
	w,p,t,e = read_twente_lexicon()
	return Lexicon(w,'Dutch',p,'Twente newspaper corpus lexicon')

def make_frisian_frysk_lexicon():
	w,p,t,e = read_frisian_lexicon()
	return Lexicon(w,'Frisian',p,'Frysk lexicon')

def get_overlap_wordset():
	f = '/vol/tensusers/mbentum/FRISIAN_ASR/LEXICONS/overlap_wordlist'
	t = open(f).read().split('\n')
	return set(t)

def find_new_words(text = '',words = None):
	if not words: words = words = re.sub('\s+',' ',text).split(' ')
	dutch, frisian = make_dutch_and_frysian_lexicon()
	overlap = get_overlap_wordset()
	fws = set(frisian.words)
	dws = set(dutch.words)
	new_overlap,new_dutch,new_frisian,error = [],[],[],[]
	for w in words:
		 #word, lang = w.split('-')
		word = w[:-3]
		if w.endswith('-nl'):lang = 'nl'
		if w.endswith('-fr'):lang = 'fr'
		if word in overlap: continue
		if lang == 'fr':
			if word in dws: new_overlap.append(w)
			elif word not in fws: new_frisian.append(w)
		elif lang == 'nl':
			if word in fws: new_overlap.append(w)
			elif word not in dws: new_dutch.append(w)
		else:error.append(w)
	return new_dutch,new_frisian,new_overlap,error, words


def label_lexicon(label, filename):
	lexicon = [line for line in open(filename).read().split('\n') if line]
	output = []
	for line in lexicon:
		word,pron = line.split('\t')
		wordo = word +label
		prono = ' '.join([p+label for p in pron.split(' ')])
		output.append(wordo + '\t' + prono)
	return '\n'.join(output)




	
def make_freq_dict(tokenized_text):
	'''create a dictionary that maps frequency to words of that frequency.'''
	word_freqs = Counter(tokenized_text)
	d = {}
	for word in word_freqs:
		freq = word_freqs[word]
		if freq not in d.keys(): d[freq] = []
		d[freq].append(word)
	return d

	


'''
	def find_doubles(self):
		from collections import Counter
		output = []
		for word in self.word2pron.keys():
			x = self.word2pron[word]
			counts = Counter(x)
			for pron,count in counts.items():
				if count > 1: output.append(pron)
				if count > 2: print(pron, 'occurs',count,'times')
		return output

	def remove_doubles(self):
		if not hasattr(self,'deleted'):self.deleted = []
		o = self.find_doubles()
		deleted = []
		if not o: 
			self._check_deleted()
			return 'done'
		print('double entries:',len(o),'found, deleting them')
		for pron in o:
			i = self.prons.index(pron)
			dp = self.prons.pop(i)
			dw = self.words.pop(i)
			deleted.append([dw,dp])
		print('removed',len(deleted),'double entries')
		if deleted: self.deleted.extend(deleted)
		self._make_word2pron()
		self.remove_doubles()

	def _check_deleted(self):
		if not self.deleted: print('ok, no entries were deleted')
		self.deleted_errors = []
		for word, pron in self.deleted:
			if word not in self.word2pron.keys():
				self.deleted_errors.append([word,pron,'word not found'])
				continue
			prons = self.word2pron[word]
			if pron not in prons:
				self.deleted_errors.append([word,pron,'pron not found'])
		if not self.deleted_errors:
			print('ok: all deleted doubles still in lexicon')
		else: print('found:',len(self.deleted_errors),'problems')
'''
