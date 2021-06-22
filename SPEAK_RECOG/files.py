import glob
import random
import re
import subprocess
import os


table_dir= 'TABLES/'
audo_dir = '/vol/bigdata/corpora2/CGN2/data/audio/wav/'
audio_info = dict([line.split('\t') for line in open('audio_info.txt').read().split('\n') if line])
ort_dir = '/vol/bigdata/corpora2/CGN2/data/annot/text/ort/'
audios = make_audios()

def make_audios():
	audios = {}
	for k in audio_info.keys():
		audios[k] = Audio(k)
		if not audios[k].ok: print(k)
	return audios


def make_tables(d = table_dir,n = None):
	fn = glob.glob(table_dir+ '*.Table')
	if n: fn = random.sample(fn,n)
	o, bad = [],[]
	for f in fn:
		t = Table(f)
		if t.ok: o.append(t)
		else: bad.append(t)
	print(len(fn),'files, and',len(bad),'failed to read:\n','\n'.join(bad))
	return o




class Table:
	def __init__(self,path = ''):
		try: self.table_list = open(path).read().split('\n') 
		except: 
			self.ok = False
			self.table_list = []
		else: self.ok = True
		self.path = path
		self.file_id = path.split('/')[-1].split('.')[0]
		self._set_lines()
		self.nlines = len(self.lines)
		self.duration = sum([l.duration for l in self.lines])
		self.speakers = list(set([l.speaker_id for l in self.lines]))
		self.nspeakers = len(self.speakers)
		self.audio_info = audios[self.file_id]

	def __repr__(self):
		m = self.file_id + ' | ' + make_time(self.duration) + ' | ' + str(self.nspeakers)
		m += ' | ' + self.comp + ' | ' + self.region
		return m

	def _set_lines(self):
		self.lines = []
		self.bad_lines = []
		for line in self.table_list:
			l = Line(line,self)
			if l.ok: self.lines.append(l)
			else: self.bad_lines.append(line)

	def get_speaker_lines(self, speaker_id):
		output = []
		for line in self.lines:
			if line.speaker_id == speaker_id: output.append(line)
		return output

	@property
	def audio_fn(self):
		return self.audio_info.path

	@property
	def comp(self):
		return self.audio_info.comp

	@property
	def region(self):
		return self.audio_info.region
		


class Line:
	def __init__(self,line, table = None):
		self.line = line
		self.table = table
		temp = line.split('\t')
		try: 
			self.start  = float(temp[0])
			self.speaker_id = temp[1]
			self.text = temp[2]
			self.end = float(temp[3])
			self.ok = True
			self.duration = self.end - self.start
		except: self.ok = False

	def __repr__(self):
		if self.ok:
			return self.speaker_id+ ' | ' + self.text  + ' | ' + make_time(self.duration)
		return self.line + ' | ' + str(self.ok) 


	@property
	def audio_fn(self):
		if self.table: return self.table.audio_info.path

	@property
	def comp(self):
		if self.table: return self.table.audio_info.comp

	@property
	def region(self):
		if self.table: return self.table.audio_info.region

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
	fn = glob.glob(audo_dir + '**/*.wav',recursive=True)
	output = []
	for f in fn:
		file_id = f.split('/')[-1].split('.')[0]
		o = subprocess.check_output('sox --i ' + f, shell =True).decode().replace('\n','\t').strip()
		o = o.replace('\t',',').replace("'",'')
		o = re.sub('\s+','',o)
		output.append(file_id + '\t' + o)
	with open('audio_info.txt','w') as fout:
		fout.write('\n'.join(output))
	return output


def _make_tables_from_ort(fn = None):
	if not fn:fn = glob.glob(ort_dir + '**/.*.ort.gz')
	for f in fn:
		filename = f.split('/')[-1]
		file_id = filename.split('.')[0]
		if os.path.isfile('TABLES/' + file_id + '.Table'):
			print(file_id,'table already created, skipping')
			continue
		os.system('cp ' + f + ' ORT/')
		os.system('gunzip ' + 'ORT/' + filename)
		os.system('praat ort_2_table.praat ORT/' + filename.strip('.gz') + ' TABLES/')


