import glob
import os
import progressbar as pb
import re
import sys
import textract
from texts.models import Text, Language, TextType, Source
from utils import make_tables as mt
import progressbar as pb


'''
[frl:       = Codeswitch naar Fries voor nederlandse spreker
[nl: = Codeswitch naar Nederlands vanuit het Fries
[frl-??:    = Niet specifiek herkend Fries dialect of Fries uitgesproken woord dat qua spelling of uitspraak niet als correct Fries wordt aangemerkt maar wel degelijk Fries klinkt. Altijd fonetisch genoteerd. 
[frl_??:    = zelfde als frl-?? (wordt in uiteindelijk dataset nog opgeschoond / genormaliseerd)
[frl-nl:    = Vernederlandst Fries woord
[nl-frl:    = Verfriest Nederlands woord
[frl-sw:    = Herkend als Zuid-west Fries dialect
[nl-overlap: = Betekenis nu niet bekend, ingevoerd door taalspecialist. Betekenis wordt nagevraagd.
'''

output_dir = '/vol/tensusers/mbentum/FRISIAN_ASR/'

def get_all_council_transcriptions_with_tags(error = False,save = False, with_original=False):
	'''
	save 		whether to save the file
	with_orig.. whether to add the orginal label (for debugging) 
	'''
	tt = Text.objects.filter(source__name = 'frisian council transcripts').filter(error = error)
	o = []
	for t in tt:
		#provides a line with wav filename start time end time label
		line = t.transcription.line_with_tags
		if line:o.append(line) #only append line if there is something in the label
		if with_original:
			o.append(t.transcription.line)
			o.append('-'*9)
	if save: 
		filename = output_dir + 'labels_with_tags'
		if error: filename += '_error'
		if with_original:filename += '_with_original'
		filename += '.txt'
		with open(filename,'w') as fout:
			fout.write('\n'.join(o))
	return o


nsn = 'nsn,geluid,oversturing,ruis,klok,hamer,foto,applaus,microfoon,klop'
nsn += ',tafel,koffie,stoel,water,typen,pen,papier,geklap,telefoon'
nsn = nsn.split(',')
spn = 'spn,spraak,kuch,lach,zucht,za,slikken,geroezemoes,ademhaling'
spn = ',lipsmack,he,hè,eh,nies,hoest'
spn = spn.split(',')


filename_labels = '/vol/tensusers3/Frisiansubtitling/Downloads-Humainr/second_batch/labels.txt'
filename_protocol= '/vol/tensusers3/Frisiansubtitling/Downloads-Humainr/First_Batch_20210218/Transcriptions/protocol_en_tags.txt'

unknown_language= Language.objects.get(name='unknown')
dutch = Language.objects.get(name='Dutch')
frisian = Language.objects.get(name='Frisian')
dutch_frisianized = Language.objects.get(name='Dutch_frisianized')
frisian_dutchized = Language.objects.get(name='Frisian_dutchized')
english = Language.objects.get(name='English')
ld = {'FRL':frisian,'NL':dutch,'frl':frisian,'fr':frisian,'nl':dutch,'frl-nl':frisian_dutchized}
ld.update({'nl-frl':dutch_frisianized, 'frl-sw':frisian,'en':english,'frl_??':'frl-??'})
ld.update({'frl-??':'frl-??'})

cs = 'frl,nl,frl-??,frl_??,frl-nl,nl-frl,frl-sw,nl-overlap,overlap-nl,en'.split(',')
tags = dict([[x,'code_switch'] for x in cs])
tags.update({'++':'enrich','--':'cleaning','eh':'eh','he':'eh','hè':'eh'})

council_source = Source.objects.get(name= 'frisian council transcripts')
radio_source = Source.objects.get(name= 'frisian radio broadcasts')
text_type = TextType.objects.get(name = 'manual transcription')

def get_council_transcriptions_text():
	return open(filename_labels).read().split('\n')

def make_council_transcriptions():
	transcription = get_council_transcriptions_text()
	output = []
	for line in transcription:
		output.append( Transcription(line,'council') )
	return output

def make_radio_transcriptions():
	t = mt.Tables()
	output = []
	bar = pb.ProgressBar()
	bar(range(len(t.combined_table.lines)))
	print('creating transcription objects')
	for i,line in enumerate(t.combined_table.lines):
		bar.update(i)
		output.append(Transcription(line,'radio'))
	return output


def load_transcriptions_in_database(transcriptions=None, save = False,
	line_type= 'council',start = 0, check_db=True):
	if not transcriptions: 
		if line_type == 'council':transcriptions = make_council_transcriptions()
		if line_type == 'radio': transcriptions = make_radio_transcriptions()
	o = []
	if line_type == 'council':source = council_source
	if line_type == 'radio': source = radio_source
	print('loading transcriptions in database',line_type)
	bar = pb.ProgressBar()
	bar(range(len(transcriptions[start:])))
	for i,t in enumerate(transcriptions[start:]):
		bar.update(i)
		o.append(add_transcription(t,save,source,check_db))
	return o

def add_transcription(t, save = False, source = council_source, check_db = True):
		error = t.get_bracket_error or t.bracket_error or t.tag_error
		multiple_languages = True if len(t.languages) > 1 else False
		if check_db:
			o = Text.objects.filter(start_time = t.start, end_time = t.end, wav_filename = t.wav)
			if o:
				print('transcription already stored, returning object from database')
				return o
		o =Text(filetype = 'txt',raw_text = t.text, transcription_meta = t.line, 
			main_language = t.language, source = source, text_type= text_type,
			start_time = t.start, end_time = t.end, wav_filename = t.wav,
			multiple_languages = multiple_languages, error = error)
		if not save: return o
		try: o.save()
		except:
			print('could not save:',t)
			print(sys.exc_info())
		else:
			for language in t.languages:
				o.all_languages.add(language)
		return o
		

class Transcription:
	def __init__(self,line,line_type = 'council'):
		self.line = line
		self.line_type = line_type
		if line_type == 'council':
			self.wav, self.l, self.start, self.end, self.text= line.split('\t')
		if line_type == 'radio':
			self.wav = line.filename + '.wav'
			self.l = line.language
			self.start, self.end = line.start, line.end
			self.text = line.label
			self.line = line.__dict__
		self.start = float(self.start)
		self.end = float(self.end)
		self.duration = self.end - self.start
		try:self.language = ld[self.l]
		except: self.language = unknown_language
		self.extract_brackets_and_words()

	def __repr__(self):
		t = self.text if len(self.text) < 75 else self.text[:75] + '... '
		m = t.ljust(80) + ' | ' + self.language.name.ljust(8) 
		m += ' | ' + str(round(self.duration))
		return m

	def extract_brackets_and_words(self):
		self.brackets = []
		brackets, self.line_without_brackets,word_chunks,error = get_brackets(self.text)
		self.get_bracket_error = error
		for b in brackets:
			self.brackets.append(Bracket(b,line_type = self.line_type))
		self.words = []
		i = 0
		for chunk in word_chunks:
			for w in chunk:
				self.words.append(Word(w,self.language,False))
			if i < len(self.brackets):
				b = self.brackets[i]
				if b.error:continue
				self.words.extend(b.words)
				i +=1

	@property
	def code_switched(self):
		cs = False
		for w in self.words:
			if w.code_switched: cs= True
		return cs

	@property
	def bracket_error(self):
		e1 = sum([b.error for b in self.brackets]) > 0
		return e1 or self. get_bracket_error
		
	@property
	def tag_error(self):
		return sum([b.tag_error for b in self.brackets]) > 0

	@property
	def languages(self):
		return list(set([w.language for w in self.words if w.language and 
			type(dutch) == type(w.language)]))

	@property
	def text_with_tags(self):
		d = {dutch:'-nl',frisian:'-fr',dutch_frisianized:'-nl',frisian_dutchized:'-fr',
			english:'-eng'}
		output = []
		for word in self.words:
			if word.word == '$$': continue
			if word.is_word and word.language: 
				if word.language == 'frl-??': output.append('<UNK>')
				elif word.language and word.language.name == 'unknown': output.append('<UNK>')
				else: output.append(word.word + d[word.language])
			else:output.append(word.word)
		return ' '.join(output).lower()

	@property
	def text_without_tags(self):
		output = []
		for word in self.words:
			if word.word == '$$': continue
			if word.is_word and word.language: output.append(word.word)
		return ' '.join(output).lower()

	@property
	def tags_list(self):
		d = {dutch:'nl',frisian:'fr',dutch_frisianized:'nl',frisian_dutchized:'fr',
			english:'eng'}
		output = []
		for word in self.words:
			if word.word == '$$': continue
			if word.is_word and word.language: 
				if word.language == 'frl-??': output.append('fr')
				else: output.append(d[word.language])
		return output
		
	@property
	def line_with_tags(self):
		if not self.text_with_tags: return False
		return '\t'.join([self.wav,str(self.start),str(self.end),self.text_with_tags])

	@property
	def dutch_words(self):
		o = []
		for word in self.words:
			if word.language in [dutch,dutch_frisianized]: o.append(word)
		return o

	@property
	def frisian_words(self):
		o = []
		for word in self.words:
			if word.language in [frisian,frisian_dutchized]: o.append(word)
		return o


class Word:
	def __init__(self,word,language, code_switched,tag = '',is_word = True):
		if word == '<eh>':
			if tag == '': tag = 'eh' 
			language = ''
			code_switched = False
			word = '<UNK>'
		self.word = word
		self.original_word = word
		self.language = language
		self.code_switched = code_switched
		self.tag = tag
		self.is_word = is_word
		self.check()

	def __repr__(self):
		cs = ' | code_switched' if self.code_switched else ''
		l = self.language.name if self.language else 'Unk'
		return self.word.ljust(20) + ' | ' + l.ljust(8) + cs
	
	def check(self):
		for char in list('[]{}:'):
			if char in self.word:
				self.is_word = False
				self.word = '<nsn>'

class Bracket:
	def __init__(self,bracket,line_type = 'council'):
		self.bracket = bracket
		self.line_type = line_type
		if line_type == 'council':
			self.handle_council_bracket()
		elif line_type == 'radio':
			self.handle_radio_bracket()
		else: raise ValueError('unknown line_type ' + line_type + ' should be council or radio')
		if not self.error: self.set_info()

	def handle_radio_bracket(self):
		self.text = self.bracket[1:-1]
		self.error, self.tag_error = False, False
		self.t, self.tag_text = self.text, self.text
		if self.text[:2] == 'nl':
			self.t = 'nl'
		elif self.text[:2] == 'fr':
			self.t = 'fr'
		elif self.text[:2] == 'en':
			self.t = 'en'
		elif self.text == 'spn':
			self.t = 'spn'
		elif self.text == 'nsn':
			self.t = 'nsn'
		else: self.error = True
		if not self.error: self.tag_text = self.text[2:].strip()
			

	def handle_council_bracket(self):
		self.text = re.sub(':+',':',self.bracket[1:-1])
		self.error = False
		self.tag_error = False
		self.t, self.tag_text = self.text, self.text
		if ':' in self.text:
			if self.text.count(':') == 1: self.t,self.tag_text = self.text.split(':') 
			elif self.text.count(':') == 2: 
				self.t,temp,self.tag_text = self.text.split(':') 
				self.tag_text = temp + ':' +self.tag_text
			else: self.error = True
		else: error = True

	def set_info(self):
		try:self.tag = tags[self.t]
		except: 
			self.tag = self.t
			self.tag_error = True
		self.code_switch = self.t in cs
		self.language = None
		self.words = []
		if self.code_switch: 
			try: self.language = ld[self.t]
			except: pass 
			self.make_words()
		else: self.make_tagword()

	def make_tagword(self):
			label = ''
			for item in nsn:
				if item in self.tag_text: label = '<UNK>'
			if not label:
				for item in spn:
					if item in self.tag_text: label = '<UNK>'
			if not label: 
				print('could not categorize tag text:',self.tag_text,'setting word to UNK')
				label = '<UNK>'
			self.words.append(Word(label,'',False,False))


	def __repr__(self):
		return self.text + ' | ' + self.tag + ' | '  + str(self.error) + ' | ' + str(self.tag_error)

		
	def make_words(self):
		for w in self.tag_text.split(' '):
			self.words.append(Word(w,self.language,True,self.tag))
	
	
def get_brackets(line):
	line = line.replace('[eh]','<eh>')
	line = re.sub('[^ ]<eh>',' <eh>',line)
	line = re.sub('<eh>[^ ]','<eh> ',line)
	line_without = []
	brackets = []
	st, et,old_et = -1,-1,0
	error =False 
	while True:
		st_check = line.find('[',st+1)
		if 0 < st_check < et: 
			error = True
		st = line.find('[',et+1)
		et = line.find(']',et+1)
		if 0 <= st < et:
			line_without.append( line[old_et:st] )
			brackets.append(line[st:et+1])
			old_et = et+1
		elif st > et: 
			st = et+1
			error = True
		else:
			line_without.append( line[old_et:] )
			break
		if error:
			break
	words = []
	for chunk in line_without:
		words.append([w for w in chunk.split(' ') if w])
	return brackets,line_without,words,error

class Audio:
	def __init__(self, audio_filename, texts):
		self.audio_filename = audio_filename
		self.texts = texts
		self.nsegments = len(texts)
		self.duration = round(sum([t.transcription.duration for t in texts]),2)
		self.nwords = sum([len(t.transcription.words) for t in texts])

	def __repr__(self):
		m = self.audio_filename + ' ' + str(self.duration) + ' ' + str(self.nsegments)
		m += ' ' + str(self.nwords)
		return m

def analyse_audio_recordings(wav_dict = None):
	if not wav_dict:
		wav_dict = {}
		t = Text.objects.filter(source__name='frisian council transcripts')
		for x in t:
			if x.wav_filename not in wav_dict.keys(): wav_dict[x.wav_filename] = []
			wav_dict[x.wav_filename].append(x)
	audios = []
	for key in wav_dict:
		audios.append(Audio(key,wav_dict[key]))
	return audios
