import glob
import os
import pickle
import progressbar as pb
import re
import random
import sys
import textract
from texts.models import Text, Language, TextType, Source
from utils import make_tables as mt
from utils.prefix_meeting_names import prefix_meeting_names
import progressbar as pb


'''
[frl:       = Codeswitch naar Fries voor nederlandse spreker
[nl: = Codeswitch naar Nederlands vanuit het Fries
[frl-??:    = Niet specifiek herkend Fries dialect of Fries uitgesproken woord 
			dat qua spelling of uitspraak niet als correct Fries wordt aangemerkt 
			maar wel degelijk Fries klinkt. Altijd fonetisch genoteerd. 
[frl_??:    = zelfde als frl-?? (wordt in uiteindelijk dataset nog opgeschoond 
			/ genormaliseerd)
[frl-nl:    = Vernederlandst Fries woord
[nl-frl:    = Verfriest Nederlands woord
[frl-sw:    = Herkend als Zuid-west Fries dialect
[nl-overlap: = Betekenis nu niet bekend, ingevoerd door taalspecialist. 
			Betekenis wordt nagevraagd.
'''

output_dir = '/vol/tensusers/mbentum/FRISIAN_ASR/'


def get_segments(name = 'council',partition = 'train'):
	if name == 'council': name = 'frisian council transcripts'
	elif name == 'fame': name = 'frisian radio broadcasts'
	else:
		raise ValueError('name should be council or fame')
	if partition not in 'train,dev,test'.split(','): 
		raise ValueError('partition should be train/dev/test')
	t = Text.objects.filter(source__name = name)
	return t.filter(partition = partition)
	

def make_all_council_transcriptions_with_tags(error = False,save = False, 
	with_original=False, kaldi= True, lm=False):
	'''
	save 		whether to save the file
	with_orig.. whether to add the orginal label (for debugging) 
	kaldi 		whether to create a line for kaldi AM training
	lm 			wether to create a line for srilm LM training
	'''
	if lm: kaldi = False
	if kaldi: name = 'labels_with_tags_'
	elif lm: name = 'manual_transcriptions_'
	print('creating partitions [train/dev/text] and storing as:',name)
	train_dev_test = []
	for partition in 'train,dev,test'.split(','):
		segments = get_segments('council',partition)
		o = []
		for segment in segments:
			#provides a line with wav filename start time end time label
			if kaldi:line = segment.transcription.line_with_tags
			if lm:line = segment.transcription.text_with_tags
			#only append line if there is something in the label
			if line:o.append(line)
			if with_original:
				o.append(t.transcription.line)
				o.append('-'*9)
		if save:
			filename = output_dir + name + partition
			if error: filename += '_error'
			if with_original:filename += '_with_original'
			filename += '.txt'
			with open(filename,'w') as fout:
				fout.write('\n'.join(o))
		train_dev_test.append(o)	
		print(partition,len(o),'nsegments')
	return train_dev_test


nsn = 'nsn,geluid,oversturing,ruis,klok,hamer,foto,applaus,microfoon,klop'
nsn += ',tafel,koffie,stoel,water,typen,pen,papier,geklap,telefoon'
nsn = nsn.split(',')
spn = 'spn,spraak,kuch,lach,zucht,za,slikken,geroezemoes,ademhaling'
spn = ',lipsmack,he,hè,eh,nies,hoest'
spn = spn.split(',')


filename_labels = '/vol/tensusers/mbentum/FRISIAN_ASR/labels.txt'
filename_protocol= '/vol/tensusers3/Frisiansubtitling/Downloads-Humainr/First_Batch_20210218/Transcriptions/protocol_en_tags.txt'

unknown_language= Language.objects.get(name='unknown')
dutch = Language.objects.get(name='Dutch')
frisian = Language.objects.get(name='Frisian')
dutch_frisianized = Language.objects.get(name='Dutch_frisianized')
frisian_dutchized = Language.objects.get(name='Frisian_dutchized')
english = Language.objects.get(name='English')
ld = {'FRL':frisian,'NL':dutch,'frl':frisian,'fr':frisian,'nl':dutch}
ld.update({'nl-frl':dutch_frisianized, 'frl-sw':frisian,'en':english})
ld.update({'frl-??':'frl-??','frl-nl':frisian_dutchized,'frl_??':'frl-??'})
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
			o = Text.objects.filter(start_time = t.start, end_time = t.end, 
				wav_filename = t.wav)
			if o:
				print('transcription already stored, returning object from database')
				return o
		o =Text(filetype = 'txt',raw_text = t.text, transcription_meta = t.line, 
			main_language = t.language, source = source, text_type= text_type,
			start_time = t.start, end_time = t.end, wav_filename = t.wav,
			multiple_languages = multiple_languages, error = error, 
			file_id = t.file_id)
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
			l = line
			self.file_id,self.wav,self.l,self.start,self.end,self.text=l.split('\t')
		if line_type == 'radio':
			self.wav = line.filename + '.wav'
			self.l = line.language
			self.start, self.end = line.start, line.end
			self.text = line.label
			self.line = line.__dict__
			self.file_id = ''
		if line_type == 'cgn':
			self.wav = line.filename
			self.l = 'NL'
			self.start = line.start
			self.end = line.end
			self.text = line.label
			self.file_id = line.file_id
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
		brackets,self.line_without_brackets,word_chunks,error=get_brackets(self.text)
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
		d = {dutch:'-nl',frisian:'-fr',dutch_frisianized:'-nl',
			frisian_dutchized:'-fr',english:'-eng'}
		output = []
		for word in self.words:
			if word.word == '$$': continue
			if word.is_word and word.language: 
				if word.language == 'frl-??': output.append('<spn>')
				elif word.language and word.language.name == 'unknown': 
					output.append('<spn>')
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
		twt = self.text_with_tags
		return '\t'.join([self.wav,str(self.start),str(self.end),twt])

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
			word = '<spn>'
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
		else: 
			m = 'unknown line_type ' + line_type + ' should be council or radio'
			raise ValueError(m)
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
			if self.text.count(':') == 1: 
				self.t,self.tag_text = self.text.split(':') 
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
				if item in self.tag_text: label = '<nsn>'
			if not label:
				for item in spn:
					if item in self.tag_text: label = '<nsn>'
			if not label: 
				print('could not categorize tag text:',self.tag_text,
					'setting word to spn')
				label = '<spn>'
			self.words.append(Word(label,'',False,False))


	def __repr__(self):
		m = self.text + ' | ' + self.tag + ' | '  + str(self.error) + ' | ' 
		m += str(self.tag_error)
		return m

		
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
	'''code to analyse the segment audio files, contains multiple segements
	and is a part of a meeting.
	'''
	def __init__(self, audio_filename, texts):
		self.audio_filename = audio_filename
		self.texts = texts
		self.nsegments = len(texts)
		self.duration = round(sum([t.transcription.duration for t in texts]),2)
		self.nwords = sum([len(t.transcription.words) for t in texts])

	def __repr__(self):
		m = self.audio_filename + ' ' + str(self.duration) + ' ' 
		m += str(self.nsegments) + ' ' + str(self.nwords)
		return m

def analyse_audio_recordings(wav_dict = None):
	'''analyse duration of segment audio files.'''
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



def make_meetings():
	labels = [line.split('\t') for line in open(filename_labels).read().split('\n')]
	o = {}
	bads = []
	for x in labels:
		count = 0
		prefs = []
		found = False
		for prefix in prefix_meeting_names:
			if prefix not in o.keys():
				o[prefix] = []
			if prefix in x[0]:
				o[prefix].append(x)
				count +=1
				prefs.append(prefix)
				found = True
		if not found:bads.append(x)
		if count >1:print(prefs,'\n',x,count)
	meetings = {}
	for prefix in o.keys():
		meetings[prefix] = Meeting(prefix,o[prefix])
	return meetings,o, bads

class Meeting:
	'''represents a specific meeting based on a specific prefix 
	(part of the meeting_wav.
	the meetings contains consecutive segments (i.e. seg_wav), a seg_wav contains
	multiple segements.
	A specific segment is stored in a Line.
	''' 
	def __init__(self,prefix,lines):
		self.prefix = prefix
		self._org_lines = lines
		self._set_info()
		
	def _set_info(self):
		duration = 0
		self.lines = []
		for i,line in enumerate(self._org_lines):
			self.lines.append(Line(line,i+1,duration, self))
			duration += self.lines[-1].duration

	def __repr__(self):
		m = str(make_time(self.duration)) + ' | ' + str(self.nsegments).ljust(4) 
		m += ' | ' + str(self.nlabels)
		return m

	@property
	def duration(self):
		return sum([x.duration for x in self.lines])

	@property
	def nsegments(self):
		return len(list(set([x.seg_wav for x in self.lines])))

	@property
	def nlabels(self):
		return len(self.lines)

class Line:
	'''represents a segment in a Meeting.'''
	def __init__(self,line,index,start_in_meeting = None,meeting = None):
		if type(line) == str: line = line.split('\t')
		self.line = '\t'.join(line)
		l = line
		self.meeting_wav,self.seg_wav,self.language,self.start,self.end,self.text=l
		self.start = float(self.start)
		self.end= float(self.end)
		self.duration = self.end - self.start
		self.start_in_meeting = start_in_meeting
		self.index = index
		self.meeting = meeting

	def __repr__(self):
		m = self.meeting_wav + ' | ' + make_time(self.duration) 
		m += ' | ' + make_time(self.start_in_meeting)
		m += ' | ' + str(self.index)
		return m

	@property
	def transcription(self):
		if not hasattr(self,'_transcription'): 
			self._transcription = Transcription(self.line)
		return self._transcription

	@property
	def id_line(self):
		l = self.transcription.line_with_tags
		o = [self.meeting.prefix,str(self.index),self.meeting_wav,self.seg_wav]
		meeting_start = make_time(self.start_in_meeting)
		o += [str(self.start),str(self.end),self.text,meeting_start]
		if not l:
			o += [str(self.start_in_meeting),self.line]
		else:
			o += [str(self.start_in_meeting),self.transcription.line_with_tags]
		return '\t'.join(o)

	def equal_to_text(self,text):
		if self.text == text.raw_text and self.start == text.start_time and self.end == text.end_time:
			return self.meeting_wav == text.file_id and self.seg_wav == text.wav_filename
		return False
		

def make_time(seconds):
	'''helper function to turn seconds in to hours:minutes:seconds'''
	seconds = int(seconds)
	h = str(seconds //3600)
	seconds = seconds % 3600
	m = str(seconds // 60)
	s = str(seconds % 60)
	if len(h) == 1:h =  '0' + h
	if len(m) == 1:m =  '0' + m
	if len(s) == 1:s = '0' + s
	return ':'.join([h,m,s])
			

def make_train_dev_test(meetings,perc = 0.1, seed = 1111,
	meeting_test_duration = 1800,save = False):
	
	'''partitions the meetings in a training, dev and test set. 
	perc 					the amount of materials in dev and in test partition
	seed 					number to freeze the randomization
	meeting_test_duration 	length from specific meetings reserved for dev or test
	save 					whether to save the files
	returns segments for training, dev, test
	segments from dev and test are consecutive segments 
	(upto meeting_test_duration length)
	meetings used for dev are not used for test
	all remaining segments are put into training
	'''
	print(perc,seed,meeting_test_duration,save)
	random.seed(seed)
	total_duration = sum([meetings[k].duration for k in meetings.keys()])
	dev_test_duration = total_duration * perc
	n_test_meetings = int(round(dev_test_duration / meeting_test_duration,0))
	print('total duration:',make_time(total_duration))
	print('dev test duration:',make_time(dev_test_duration))
	print('n dev test meetings:',n_test_meetings)
	dev_test_meetings = random.sample(meetings.keys(),n_test_meetings *2)
	dev_meetings = dev_test_meetings[:n_test_meetings]
	test_meetings = dev_test_meetings[n_test_meetings:]
	print('\ndev meetings:\n','\n'.join(dev_meetings))
	print('\ntest meetings:\n','\n'.join(test_meetings))
	train,dev, test = [],[],[]
	for k in meetings.keys():
		if k not in dev_test_meetings:train.extend(meetings[k].lines)
		elif k in dev_meetings:
			temp_dev,other = _extract_segments_from_meeting(meetings[k],
				meeting_test_duration)
			dev.extend(temp_dev)
			train.extend(other)
		elif k in test_meetings:
			temp_test,other = _extract_segments_from_meeting(meetings[k],
				meeting_test_duration)
			test.extend(temp_test)
			train.extend(other)
	check_train_dev_test(train,dev,test)
	if save:
		_save_set(train,'train')
		_save_set(dev,'dev')
		_save_set(test,'test')
	return train,dev,test
		



def _extract_segments_from_meeting(meeting,meeting_test_duration):
	'''extracts consecutive segments from a single meeting. 
	The start time is random, but
	starts no later that meeting duration minus meeting_test_duration. 
	return the selected segments (within random start + meeting_test_duration) 
	as output all other segments are returned in other
	'''
	duration = meeting.duration
	if duration < meeting_test_duration: return meeting.lines
	start_time = random.randint(0,int(duration - meeting_test_duration))
	output = []
	other = []
	total_line_duration = 0
	for line in meeting.lines:
		if start_time > line.start_in_meeting + line.duration: other.append(line)
		elif total_line_duration > meeting_test_duration:other.append(line)
		else:
			output.append(line)
			total_line_duration += line.duration
	print('\nmeeting:',meeting.prefix,'\n')
	print('duration meeting:',meeting.duration, start_time)
	print('duration selected lines:',sum([l.duration for l in output]))
	print('goal duration:',meeting_test_duration)
	print('nlines:',len(output),'\n')
	return output,other


def _save_set(lines,set_type = 'train'):
	'''saves the train, dev and test partitions.
	in addition saves an additional info file to identify the segments and a not used
	file for the segments that were excluded'''
	output = []
	output_id = []
	not_used = []
	for line in lines:
		x = line.transcription.line_with_tags
		if x:
			output.append(x)
			output_id.append(line.id_line)
		else:not_used.append(line.id_line)
	with open('../council_'+set_type,'w') as fout:
		fout.write('\n'.join(output))
	with open('../council_'+set_type+'_id','w') as fout:
		fout.write('\n'.join(output_id))
	with open('../council_'+set_type+'_not_used','w') as fout:
		fout.write('\n'.join(not_used))

def _in_other(source,other):
	for x in source:
		if x in other: return True
	return False
	
def check_train_dev_test(train,dev,test):
	'''checks if there is no overlap in segments between the train, 
	dev and test partitions'''
	no_overlap =True
	if _in_other(train,dev):
		print('lines in train are found in dev')
		no_overlap = False
	if _in_other(train,test):
		print('lines in train are found in test')
		no_overlap = False
	if _in_other(test,dev):
		print('lines in test are found dev')
		no_overlap = False
	if no_overlap: print('no overlap between sets')

	print('train segments duration:',make_time(sum([x.duration for x in train])))
	print('dev segments duration:',make_time(sum([x.duration for x in dev])))
	print('test segments duration:',make_time(sum([x.duration for x in test])))
	

def find_partition(texts,train,dev,test, save = False):
	'''stores the partition a text belongs to in the sqlite database'''
	train_count,dev_count,test_count = 0,0,0
	bar = pb.ProgressBar()
	bar(range(len(texts)))
	ntrain, ndev, ntest = len(train), len(dev), len(test)
	for i,text in enumerate(texts):
		bar.update(i)
		found = False
		for j,line in enumerate(test):
			if line.equal_to_text(text):
				text.partition = 'test'
				test_count += 1
				found = True
				test.pop(j)
				break
		if not found:
			for j,line in enumerate(dev):
				if line.equal_to_text(text):
					text.partition = 'dev'
					dev_count += 1
					found = True
					dev.pop(j)
					break
		if not found:
			for j,line in enumerate(train):
				if line.equal_to_text(text):
					text.partition = 'train'
					train_count += 1
					found = True
					train.pop(j)
					break
		if save:
			text.save()
	print('train:',ntrain,train_count)
	print('dev:',ndev,dev_count)
	print('test:',len(test),test_count)


def make_cgn_transcriptions(tables = None):
	from SPEAK_RECOG import files, speakers
	if not tables: tables = files.make_tables()
	tables = [table for table in tables if table.audio_info.sample_rate >= 16000]
	s2g= load_speaker2gender()
	bar = pb.ProgressBar()
	bar(range(len(tables)))
	print('creating transcription objects for cgn')
	output = []
	for i,table in enumerate(tables):
		bar.update(i)
		for line in table.lines:
			if line.speaker_id == 'BACKGROUND' or line.speaker_id == 'COMMENT':
				continue
			line.file_id = line.table.file_id
			g = s2g[line.speaker_id] if line.speaker_id in s2g.keys() else ''
			line.gender = g
			line.language = 'nl'
			line.label = clean_text(line.text)
			line.filename = line.audio_fn
			output.append(Transcription(line,'cgn'))
	return output

def load_cgn_in_database(cgn_transcriptions = None, save = False):
	if not cgn_transcriptions: cgn_transcriptions = make_cgn_transcriptions()
	cgn_source = Source.objects.get(name= 'cgn')
	text_type = TextType.objects.get(name = 'manual transcription')
	output = []
	bar = pb.ProgressBar()
	bar(range(len(cgn_transcriptions)))
	for i,t in enumerate(cgn_transcriptions):
		bar.update(i)
		error = t.get_bracket_error or t.bracket_error or t.tag_error
		o =Text(filetype = 'txt',raw_text = t.text, transcription_meta = t.line, 
			main_language = t.language, source = cgn_source, text_type= text_type,
			start_time = t.start, end_time = t.end, wav_filename = t.wav,
			multiple_languages = False, error = error, file_id = t.file_id,
			speaker_id = t.line['speaker_id'],speaker_gender = t.line['gender'])
		if save:o.save()
		output.append(o)
	return output

def load_speaker2gender():
	return pickle.load(open('speakers2gender','rb'))

def clean_text(text):
	for char in '.,?!:;':
		text =text.replace(char,'')
	return text 

