
import glob
import random
import re
import subprocess
import os


audio_dir = '/vol/tensusers3/Frisiansubtitling/Downloads-Humainr/second_batch/Audio/all_audio/'
audio_info = dict([line.split('\t') for line in open('../audio_info.txt').read().split('\n') if line])
kaldi_wav_dir = '/vol/tensusers3/Frisiansubtitling/COUNCIL/wav/'
#audios = make_audios()

def make_audios():
	audios = {}
	for k in audio_info.keys():
		# print(k,audio_info[k])
		audios[k] = Audio(k)
		if not audios[k].ok: print(k)
	return audios

def downsample_audio(source_dir = audio_dir, goal_dir = kaldi_wav_dir, sample_rate = 16000,
	nchannels = 1, execute = False):
	fn = glob.glob(audio_dir + '*.wav')
	for f in fn:
		f = f.split('/')[-1]
		command = 'sox -v 0.98 ' + audio_dir + f + ' -r ' + str(sample_rate) 
		command += ' -c ' + str(nchannels) + ' ' +  goal_dir + f
		if execute: os.system(command)
		else:print(command)
	if not execute:print('to execute command set execute to True')
	else: print('converted:',len(fn), 'to',goal_dir,'sr',sample_rate,'nchannels',nchannels)

class Audio:
	def __init__(self,file_id):
		self.file_id = file_id
		self._set_info()
		if self.ok:
			t = self.info.split(',')
			self.path = t[0].split(':')[1]
			self.channels= int(t[1].split(':')[1])
			self.sample_rate= int(t[2].split(':')[1])
			self.precision= t[3].split(':')[1]
			self.duration= t[4].strip('Duration:').split('=')[0]
			self.samples= int(t[4].strip('Duration:').split('=')[1].split('samples')[0])
			self.file_size= t[5].split(':')[1]
			self.bit_rate = t[6].split(':')[1]
			self.sample_encoding = t[7].split(':')[1]
			self.seconds = self.samples / self.sample_rate
			self.comp = self.path.split('/')[-3]
			self.region= self.path.split('/')[-2]

	def __repr__(self):
		m = 'Audio: ' + self.file_id + ' | ' + self.duration + ' | ' + str(self.channels) 
		m += ' | ' + self.comp + ' | ' + self.region
		return m
			

	def _set_info(self):
		self.ok =True
		try: self.info = audio_info[self.file_id]
		except:
			print(file_id,'not found in audio_info')
			self.ok = False
		if 'Duration' not in self.info:self.ok = False


def make_time(seconds):
	seconds = int(seconds)
	h = str(seconds //3600)
	seconds = seconds % 3600
	m = str(seconds // 60)
	s = str(seconds % 60)
	if len(h) == 1:h =  '0' + h
	if len(m) == 1:m =  '0' + m
	if len(s) == 1:s = '0' + s
	return ':'.join([h,m,s])


def _make_audio_info():
	fn = glob.glob(audio_dir + '**/*.wav',recursive=True)
	output = []
	for f in fn:
		file_id = f.split('/')[-1].split('.')[0]
		o = subprocess.check_output('sox --i ' + f, shell =True).decode().replace('\n','\t').strip()
		o = o.replace('\t',',').replace("'",'')
		o = re.sub('\s+','',o)
		output.append(file_id + '\t' + o)
	with open('../audio_info.txt','w') as fout:
		fout.write('\n'.join(output))
	return output
