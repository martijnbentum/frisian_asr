import re


'copied from file: /home/eyilmaz/main2/latest_ASR_exps/fame/data/local/dict/lexicon.txt'
f = '/vol/tensusers/mbentum/FRISIAN_ASR/LEXICONS/final_lexicon_emre.txt'
twente_f = '/vol/tensusers/mbentum/FRISIAN_ASR/LEXICONS/twente_lexicon.txt'
frisian_f = '/vol/tensusers/mbentum/FRISIAN_ASR/LEXICONS/Frysk.txt'
council_f = '/vol/tensusers/mbentum/FRISIAN_ASR/LEXICONS/lexicon_frisian_dutch_council.txt'

class Lexicon:
	def __init__(self,words,language,prons= [],name = ''):
		self.words= words 
		self.language = language
		self.prons= prons
		self.name = name
		if prons and len(words) == len(prons): 
			self.word2pron = dict([[word,pron] for word, pron in zip(words,prons)])
		else:self.word2pron = {}

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
	return Lexicon(words,'Dutch, Frisian',prons,'council')




def read_frisian_dutch_lexicon():
	t, words, prons, errors = read_pronlex(f)
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

		
def make_dutch_and_frysian_lexicon():
	cw,cp,l,t,w,p,e = read_frisian_dutch_lexicon()
	cwd,cwf,cpd,cpf = [],[],[],[]
	for word,pron,language in zip(cw,cp,l): 
		if language == 'Dutch':
			cwd.append(word)
			cpd.append(pron)
		elif language == 'Frisian':
			cwf.append(word)
			cpf.append(pron)
	dutch = Lexicon(cwd,'Dutch',cpd,'fame')
	frisian= Lexicon(cwf,'Frisian',cpf,'fame')
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
	
