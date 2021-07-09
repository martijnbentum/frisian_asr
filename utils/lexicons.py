import re
import statistics
from tqdm.auto import tqdm
import pickle
from collections import Counter


'copied from file: /home/eyilmaz/main2/latest_ASR_exps/fame/data/local/dict/lexicon.txt'
f = '/vol/tensusers/mbentum/FRISIAN_ASR/LEXICONS/final_lexicon_emre.txt'
twente_f = '/vol/tensusers/mbentum/FRISIAN_ASR/LEXICONS/twente_lexicon.txt'
frisian_f = '/vol/tensusers/mbentum/FRISIAN_ASR/LEXICONS/Frysk.txt'
council_f = '/vol/tensusers/mbentum/FRISIAN_ASR/LEXICONS/lexicon_frisian_dutch_council.txt'
council_cleaned_labelled = '/vol/tensusers/mbentum/FRISIAN_ASR/LM/council_notes_cleaned_labelled'

class Lexicon:
	def __init__(self,words,language,prons= [],name = '', labelled = False):
		self.words= words 
		self.language = language
		self.prons= prons
		self.name = name
		self.labelled = labelled
		if prons and len(words) == len(prons): 
			self.word2pron = dict([[word,pron] for word, pron in zip(words,prons)])
		else:self.word2pron = {}

	def word2pron(self):
		pass

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
		
		

def read_pronlex(filename):
	t = [line.split('\t') for line in open(filename).read().split('\n')]
	words = [line[0] for line in t if len(line) >= 2]
	prons = [line[1] for line in t if len(line) >= 2]
	errors= [line for line in t if len(line) < 2]
	return t,words, prons, errors

def council_lexicon():
	t, words, prons, errors = read_pronlex(council_f)
	return Lexicon(words,'Dutch, Frisian',prons,'council',labelled = True)




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


def _make_indices(nwords, frequency):
	'''indices to slice a text in to the necessary number of chunks to compute arf.'''
	step = round(nwords/frequency)
	divisable = True if nwords % frequency == 0 else False
	if divisable: 
		indices = list(range(0,nwords,step))
		indices.append(nwords)
		return indices
	indices = []
	count,extra = 0,0
	diff = (frequency * step) - nwords
	x = 1 if diff < 0 else -1 
	diff = diff *-1
	for i in range(frequency+1):
		if i == frequency - diff: extra = x
		indices.append(count)
		count += step + extra
	return indices


def compute_arf_word(word, tokenized_text):
	'''compute arf of a single word, terrible implementation of arf, very slow, correct ???'''
	assert type(tokenized_text) == list
	frequency = tokenized_text.count(word)
	if frequency in [0,1]: return frequency, frequency
	nwords = len(tokenized_text)
	indices = _make_indices(nwords,frequency)
	rf,rfs = [], []
	for _ in range(frequency):
		rf = []
		for i in range(len(indices)):
			if i == len(indices) -1: break
			chunk = tokenized_text[indices[i]:indices[i+1]]
			rf.append(word in chunk)
		rfs.append(sum(rf))
		tokenized_text.append(tokenized_text.pop(0))
	arf = statistics.mean(rfs)
	print(word,frequency,arf,nwords, indices[-40:], len(indices))
	return arf

def compute_arf_words(words, tokenized_text , frequency, use_rf =False):
	'''compute arf of set of words, terrible implementation of arf, very slow, correct ???
	reduced frequency is feasible in processing time
	'''
	assert type(tokenized_text) == list
	rfs,arf = {}, {}
	if frequency in [0,1]: 
		for word in words:
			arf[word] = frequency
		return arf
	nwords = len(tokenized_text)
	indices = _make_indices(nwords,frequency)
	step = round(nwords/frequency)
	n = 1 if use_rf else step
	for _ in range(n):
		rf= {}
		for i in range(len(indices)):
			if i == len(indices) -1: break
			chunk = tokenized_text[indices[i]:indices[i+1]]
			for word in words:
				if word not in rf.keys(): rf[word] = []
				rf[word].append(word in chunk)
		for word in words:
			if word not in rfs.keys():rfs[word] =[]
			rfs[word].append(sum(rf[word]))
		tokenized_text.append(tokenized_text.pop(0))
	for word in rfs.keys():
		arf[word] = statistics.mean(rfs[word])
	return arf
			
	
def make_freq_dict(tokenized_text):
	'''create a dictionary that maps frequency to words of that frequency.'''
	word_freqs = Counter(tokenized_text)
	d = {}
	for word in word_freqs:
		freq = word_freqs[word]
		if freq not in d.keys(): d[freq] = []
		d[freq].append(word)
	return d

def get_freq_arf(filename_text = council_cleaned_labelled, text = None, 
	'''get frequency and arf for all words in a text. Arf computation time is not great'''
	load_filename = '', use_rf = False):
	if load_filename: return load_arf_file(load_filename)
	if not text:
		text = open(filename_text).read().replace('\n',' ').split(' ')
	words = list(set(text))
	freq_dict = make_freq_dict(text)
	output ={} 
	for frequency in tqdm(sorted(freq_dict.keys())):
		words = freq_dict[frequency]
		arf = compute_arf_words(words=words,tokenized_text=text,frequency=frequency, use_rf =use_rf)
		for i,word in enumerate(words):
			output[word] = [arf[word],frequency]
	pickle.dump(output,open('council_cleaned_labelled_arf','wb'))
	return output
	

def load_arf_file(filename = 'council_cleaned_labelled_arf'):
	'''load pickled arf dict, maps word to arf and frequency value.
	by default this is based on cleaned council materials
	'''
	return pickle.load(open(filename,'rb'))
	
	
