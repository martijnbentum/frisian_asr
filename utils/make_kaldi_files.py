from utils import manual_transcriptions as mt
from texts.models import Text
import os

data_dir = '/vol/tensusers3/Frisiansubtitling/COUNCIL/data/'
council_wav_dir = '/vol/tensusers3/Frisiansubtitling/COUNCIL/wav/'
fame_wav_dir = '/vol/tensusers/mbentum/FRISIAN_ASR/corpus/fame/wav/'

class Filemaker:
	def __init__(self,data_dir = data_dir):
		self.data_dir = data_dir
		self.council = 'frisian council transcripts'
		self.fame= 'frisian radio broadcasts'
		self.partitions = 'train,dev,test'.split(',')
		self._load_segments()
		self.checked = False

	def _load_segments(self):
		corpora= 'fame,council'.split(',')
		for partition in self.partitions:
			f = Text.objects.filter
			if partition == 'train': t = f(source__name = self.council) | f(source__name=self.fame)
			else:t = f(source__name = self.council) #only use council materials for training
			setattr(self,partition,t.filter(partition = partition))

	def _check(self):
		for partition in self.partitions:
			print('checking partition:',partition)
			p = getattr(self,partition)
			output,rejected = [],[]
			for text in p:
				if text.transcription.duration < 0.1 or not text.transcription.line_with_tags:
					rejected.append(text)
				else:output.append(text)
			setattr(self,partition + '_rejected',rejected)
			setattr(self,partition,output)
		self.checked = True

	def _make(self,make = 'text', save=False):
		if not self.checked: self._check() 
		if make == 'text': f = self._partition2text
		elif make == 'wav.scp': f = self._partition2wavscp
		elif make == 'segments': f = self._partition2segments
		elif make == 'utt2spk': f = self._partition2utt2spk
		else: raise ValueError(make + ' unknown')
		for partition in self.partitions:
			directory = self.data_dir + partition + '/'
			print( 'making ' + make + ' for ' + partition )
			if save and not os.path.isdir(directory):
				os.mkdir(directory)
			output = f(partition)
			setattr(self,partition + '_' + make.replace('.','') ,partition)
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

	def _partition2text(self,partition):
		p = getattr(self,partition)
		output = []
		for text in p:
			output.append(text.utterance_id + ' ' + text.transcription.text_with_tags)
		return output
			
	def _partition2wavscp(self,partition):
		p = getattr(self,partition)
		output = []
		for text in p:
			if text.source.name == self.council:
				wav_filename = council_wav_dir + text.wav_filename
			elif text.source.name == self.fame:
				if partition == 'dev': partition += 'el'
				wav_filename = fame_wav_dir + partition + '/' + text.wav_filename
			else: raise ValueError(text.source.name + ' not recognized')
			if not os.path.isfile(wav_filename):
				raise ValueError(wav_filename + ' no file found')
			output.append(text.wav_filename+ ' ' + wav_filename)
		return output

	def _partition2segments(self,partition):
		p = getattr(self,partition)
		output = []
		for text in p:
			l =[text.utterance_id,text.wav_filename,str(text.start_time),str(text.end_time)] 
			output.append(' '.join(l))
		return output

	def _partition2utt2spk(self,partition):
		p = getattr(self,partition)
		output = []
		for text in p:
			speaker_id = text.speaker_id if text.speaker_id else text.wav_filename.split('.')[0]
			output.append(text.utterance_id + ' ' + speaker_id)
		return output

