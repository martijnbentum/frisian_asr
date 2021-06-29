f = '/vol/bigdata/corpora/CGN2/data/meta/text/speakers.txt'

from .files import make_tables
from .files import make_time
import random
from tqdm import tqdm

gender_dict = {'sex1':'male','sex2':'female'}

speakers = [line.split('\t') for line in open(f).read().split('\n') if line]
col_names = 'type,creator,version,update,ID,sex,birthYear,birthPlace,birthRegion'
col_names += ',language,firstLang,homeLang,workLang,resPlace,resRegion,eduPlace'
col_names += ',eduRegion,eduSize,education,eduLevel,occupation,occLevel,notes'
col_names = col_names.split(',')

def make_speakers(tables = None, min_duration = 0, min_sr = 0, verbose = False, force_all =False):
	'''creates a list of speakers based on tables (see files.py).
	min_duration 	minimal duration for a segment to be included
	min_sr 			minimal sample rate for a segment to be included
	'''
	if not tables: tables = make_tables()
	output= []
	for line in tqdm(speakers[1:]):
		s = Speaker(line, tables,min_duration=min_duration, min_sr=min_sr)
		if len(s.lines) < 2 and not force_all: 
			if verbose: print(s,'no audio')
			continue
		output.append(s)
	return output
	
		

class Speaker:
	def __init__(self,speaker_info, tables = None, min_duration=0,min_sr=0):
		self.speaker_info = speaker_info
		self.min_duration = min_duration
		self.min_sr = min_sr
		s = speaker_info
		self._set_info()
		self.tables = []
		self.lines, self.rejected_lines = [], []
		if tables: find_tables(self,tables)		


	def __repr__(self):
		m = self.id + ' | ' + self.sex
		if self.age: m += ' | ' + str(self.age) 
		m += ' | ' + self.duration
		return m

	def _get_info(self,col_name):
		return self.speaker_info[col_names.index(col_name)]
		print(col_name, 'not part of:',col_names)

	def _set_info(self):
		self.id = self._get_info('ID')
		self.sex = gender_dict[self._get_info('sex')]
		self.birth_year = self._get_info('birthYear')
		self.language = self._get_info('language')
		try:
			self.birth_year = int(self.birth_year)
			self.age = 2000 - self.birth_year
		except:
			self.birth_year = None
			self.age = None


	@property
	def seconds(self):
		return sum([line.duration for line in self.lines])

	@property
	def duration(self):
		return make_time(self.seconds)

	def sample(self,n = 1):
		return random.sample(self.lines,n)



def find_tables(speaker,tables):
	for t in tables:
		if speaker.id in t.speakers:
			speaker.tables.append(t)
			lines = t.get_speaker_lines(speaker.id)
			for line in lines:
				if line.duration >= speaker.min_duration and \
						line.table.audio_info.sample_rate >= speaker.min_sr:
					speaker.lines.append(line)
				else: speaker.rejected_lines.append(line)

