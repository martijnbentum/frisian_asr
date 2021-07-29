'''
create training development and test sets for KALDI training.
transcriptions are stored in a sqlite database interfaced with django
the database entries contain information on language (frisian / dutch)
and origin (council, fame, cgn). Also on the location of the audio files.
based on this information all kaldi resources are created.
'''


from utils import manual_transcriptions as mt
from texts.models import Text
import os
from SPEAK_RECOG import files
import time

def make(normal = True, cgn = False, fame_dev_test = True,language_split = True,
	save = False):
	output = {}
	if normal:
		print('''making standard kaldi resource files Fame + council in training
			only council in test and dev''',time.asctime())
		f_normal = Filemaker()
		f_normal.make_all(save=save)
		output['normal']=f_normal
	if cgn:
		print('''making CGN + Fame + council in training
			only council in test and dev''',time.asctime())
		f_cgn= Filemaker(use_cgn_train = True)
		f_cgn.make_all(save=save)
		output['cgn']=f_normal
	if fame_dev_test:
		print('making fame test and dev files',time.asctime())
		f_fame= Filemaker(fame_dev_test = True)
		f_fame.make_all(save=save)
		output['fame']=f_fame
	if fame_dev_test and language_split:
		print('making fame test and dev files, split on language',time.asctime())
		f_fls = Filemaker(fame_dev_test = True, language_split =True)
		f_fls.make_all(save=save)
		output['fame_ls'] = f_fls
	if normal and language_split:
		print('making council test and dev files, split on language',
			time.asctime())
		f_ls = Filemaker(language_split =True)
		f_ls.make_all(save=save)
		output['normal_ls'] = f_ls
	return output
	
	
	

data_dir = '/vol/tensusers3/Frisiansubtitling/COUNCIL/data/'
council_wav_dir = '/vol/tensusers3/Frisiansubtitling/COUNCIL/wav/'
fame_wav_dir = '/vol/tensusers/mbentum/FRISIAN_ASR/corpus/fame/wav/'

class Filemaker:
	'''create kaldi resources.'''
	def __init__(self,data_dir = data_dir, use_cgn_train = False, 
		language_split = False,fame_dev_test = False):
		'''
		data_dir 		council data dir - directory resources are saved to
		use_cgn_train 	whether to use cgn data in training
		language_split 	whether to create dev and test sets split on language
		fame_dev_test 	whether to create fame as dev and test set
		'''
		self.data_dir = data_dir
		self.use_cgn_train = use_cgn_train
		self.language_split = language_split
		self.fame_dev_test = fame_dev_test
		self.council = 'frisian council transcripts'
		self.fame= 'frisian radio broadcasts'
		self.cgn = 'cgn'
		if language_split or self.fame_dev_test: 
			self.partitions = 'dev,test'.split(',')
		else:self.partitions = 'train,dev,test'.split(',')
		self._load_segments()
		self.checked = False

	def _load_segments(self):
		'''load data from sqlite database (with django interface).'''
		f = Text.objects.filter
		for partition in self.partitions:
			if partition == 'train': 
				if self.fame_dev_test: continue
				if self.use_cgn_train:
					t=f(source__name=self.council)|f(source__name=self.fame)|f(source__name=self.cgn)
				else:
					t=f(source__name = self.council)|f(source__name=self.fame)
			else:
				if self.fame_dev_test:
					t=f(source__name=self.fame)
				#only use council materials for dev and test
				else:t = f(source__name = self.council) 
			setattr(self,partition,t.filter(partition = partition))

	def _check(self):
		'''remove bad or very short transcriptions.'''
		for partition in self.partitions:
			print('checking partition:',partition)
			p = getattr(self,partition)
			output,rejected = [],[]
			if self.language_split:nl,fr,mx,unk_languages = [],[],[],[]
			for text in p:
				if (text.transcription.duration < 0.1 or 
						not text.transcription.line_with_tags):
					rejected.append(text)
				else:
					if self.language_split:
						if self._check_language_split(text) == 'mx': 
							mx.append(text)
						elif self._check_language_split(text) == 'nl':
							nl.append(text)
						elif self._check_language_split(text) == 'fr':
							fr.append(text)
						else: unk_languages.append(text.languages)
					else:output.append(text)
			if self.language_split:
				output = {'nl':nl,'fr':fr,'mx':mx}
				print(len(unk_languages),'unk language text',
					list(set(unk_languages)))
			setattr(self,partition + '_rejected',rejected)
			setattr(self,partition,output)
		self.checked = True

	def _check_language_split(self,text):
		t = text.transcription.text_with_tags
		if '-nl' in t and '-fr' in t: return 'mx'
		if '-nl' in t: return 'nl'
		if '-fr' in t: return 'fr'
		return 'unk'

	def _make(self,make = 'text', save=False):
		'''create text, wav.scip, segments and uttspk files.'''
		if not self.checked: self._check() 
		if make == 'text': f = self._partition2text
		elif make == 'wav.scp': f = self._partition2wavscp
		elif make == 'segments': f = self._partition2segments
		elif make == 'utt2spk': f = self._partition2utt2spk
		else: raise ValueError(make + ' unknown')
		if self.fame_dev_test: extra = 'fame_'
		elif self.use_cgn_train: extra = 'cgn_'
		else: extra = ''
		for partition in self.partitions:
			directory = self.data_dir + extra + partition + '/'
			print( 'making ' + make + ' for ' + partition, 'extra:',extra)
			if save and not os.path.isdir(directory) and not self.language_split:
				os.mkdir(directory)
			if self.language_split: 
				self._handle_language_split(partition,make,f,save)
				continue
			output = f(partition)
			setattr(self,partition + '_' + make.replace('.','') ,partition)
			self._save(directory,make,output,save)

	def _handle_language_split(self,partition,make,f,save):
		'''splits dev and test on language i.e. dutch or frisian or mixed.'''
		for l in 'nl,fr,mx'.split(','):
			extra = 'fame_' if self.fame_dev_test else ''
			directory = self.data_dir + extra + partition + '_' + l + '/'
			if save and not os.path.isdir(directory):
				os.mkdir(directory)
			output = f(partition,l)
			setattr(self,partition + '_'+l+'_'+make.replace('.','') ,partition)
			self._save(directory,make,output,save)
			
	def _save(self,directory, make, output, save):
		if save:
			print('saving to ' + directory + make, len(output) , 'nlines')
			with open(directory + make,'w') as fout:
				fout.write('\n'.join(output))
			


	def make_all(self,save=False):
		self.make_text(save)
		self.make_wav_scp(save)
		self.make_segments(save)
		self.make_utt2spk(save)

	def make_text(self,save=False):
		self._make(make='text',save=save)

	def make_wav_scp(self,save=False):
		self._make(make='wav.scp',save=save)
	
	def make_segments(self, save=False):
		self._make(make='segments',save=save)

	def make_utt2spk(self, save=False):
		self._make(make='utt2spk',save=save)

	def _partition2text(self,partition,l = None):
		if self.language_split: p = getattr(self,partition)[l]
		else: p = getattr(self,partition)
		output = []
		for text in p:
			output.append(text.utterance_id + ' ' +
				text.transcription.text_with_tags)
		return output
			
	def _partition2wavscp(self,partition, l = None):
		if self.language_split: p = getattr(self,partition)[l]
		else: p = getattr(self,partition)
		output = []
		for text in p:
			if text.source.name == self.council:
				wav_filename = council_wav_dir + text.wav_filename
			elif text.source.name == self.fame:
				if partition == 'dev': partition += 'el'
				wav_filename = fame_wav_dir+ partition + '/' + text.wav_filename
			elif text.source.name == self.cgn:
				nchannels = files.audios[text.file_id].channels
				if nchannels == 2: 
					wav_filename = 'sox -t wav ' + text.wav_filename 
					wav_filename += ' -b 16 -t wav - remix - |'
				else:wav_filename = text.wav_filename
			else: raise ValueError(text.source.name + ' not recognized')
			if text.source.name == self.cgn:
				if not os.path.isfile(text.wav_filename):
					raise ValueError(wav_filename + ' no file found')
			elif not os.path.isfile(wav_filename):
				raise ValueError(wav_filename + ' no file found')
			output.append(text.wav_filename+ ' ' + wav_filename)
		return output

	def _partition2segments(self,partition, l = None):
		if self.language_split: p = getattr(self,partition)[l]
		else: p = getattr(self,partition)
		output = []
		for text in p:
			l =[text.utterance_id,text.wav_filename,str(text.start_time),
				str(text.end_time)] 
			output.append(' '.join(l))
		return output

	def _partition2utt2spk(self,partition, l = None):
		if self.language_split: p = getattr(self,partition)[l]
		else: p = getattr(self,partition)
		output = []
		for text in p:
			if text.speaker_id: speaker_id = text.speaker_id
			else: speaker_id = text.wav_filename.split('.')[0]
			output.append(text.utterance_id + ' ' + speaker_id)
		return output

