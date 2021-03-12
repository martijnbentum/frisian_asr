''' Create tables from text grid annotations made for the Fame corpus
provides the class Tables to load all tables for all annotations
work in progress:
	linking annotations to wav file
	extact word from wav file
	extracting code switching [language code words annotated] [nl hoe gaat het] 
		[fr WurkProgram kolleezje]
'''



import glob
import os
import progressbar as pb


input_dir = '/vol/tensusers/mbentum/FRISIAN_ASR/corpus/fame/annot/all_textgrids/'
output_dir = '/vol/tensusers/mbentum/FRISIAN_ASR/TABLES/'

input_dir_spk = '/vol/tensusers/mbentum/FRISIAN_ASR/corpus/fame/annot/all_textgrids/'
output_dir_spk = '/vol/tensusers/mbentum/FRISIAN_ASR/TABLES/SPK/'


def get_files(directory=input_dir, extension = '*.TextGrid'):
	if not directory.endswith('/'): directory += '/'
	return glob.glob(directory+ extension)

def handle_file(name, input_dir = input_dir, output_dir = output_dir):
	filename = name.split('/')[-1]
	praat_script = 'praat ort_2_table.praat' 
	command =  ' '.join([praat_script, filename,  input_dir, output_dir]) + ' > temp'
	# print('executing: ',command)
	os.system(command)

def handle_input_dir(input_dir= input_dir, output_dir = output_dir):
	fn = get_files(input_dir)
	bar = pb.ProgressBar()
	bar(range(len(fn)))
	for i,f in enumerate(fn):
		handle_file(f, input_dir = input_dir, output_dir = output_dir)
		bar.update(i)

def handle_input_spk_dir():
	pass
	#handle_input_dir(input_dir = input_dir_spk, output_dir= output_dir_spk)

def get_tables(input_dir= output_dir):
	fn = get_files(directory=input_dir,extension='*Table')
	return fn


def read_table(f):
	t = open(f).read()
	return t


class Tables:
	def __init__(self,directory = output_dir):
		self.directory = directory
		self.fn = get_files(directory=output_dir,extension='*Table')
		self.tables = []
		self.make()

	def __repr__(self):
		return 'Table '+ ' ' + str(len(self.tables)) + ' tables\nDirectory:' + self.directory


	def make(self):
		bar = pb.ProgressBar()
		bar(range(len(self.fn)))
		for i,f in enumerate(self.fn):
			self.tables.append(Table(f))
			bar.update(i)
		self.combined_table = Table()
		for table in self.tables:
			self.combined_table.lines.extend(table.lines)


class Table:
	def __init__(self,f = 'all'):
		self.lines,self.rejected_lines, self.background_lines = [], [], []
		self.f =f 
		if f == 'all': self.filename = 'Combined table'
		else:
			self.filename = f.split('/')[-1].split('.')[0]
			self.t = read_table(f)
			self.header = self.t.split('\n')[0]
			self.body = self.t.split('\n')[1:]
			self.handle_table_lines()

	def __repr__(self):
		return 'Table '+ ' ' + str(len(self.lines)) + ' lines\tFilename:' + self.filename 

	def handle_table_lines(self):
		for line in self.body:
			if line == ['']: continue
			tl = TableLine(line)
			if tl.ok: self.lines.append(TableLine(line,self.filename))
			elif tl.background: self.background_lines.append(tl)
			else:self.rejected_lines.append(tl)

class TableLine:
	def __init__(self,line, filename = ''):
		self.ok,self.background,self.label = False,False, ''
		if type(line) == str: line = line.split('\t')
		if len(line) < 4 : 
			return 
		self.line = line
		self.tmin = line[0]
		self.tmax = line[-1]
		self.tier = line[1]
		self.label = line[2]
		self.handle_tier()
		self.filename = filename

	def __repr__(self):
		return self.label 

	def __str__(self):
		m = 'label: ' + self.label + '\n'
		m += 'language: ' + str(self.language)+ '\n'
		m += 'gender: ' + str(self.gender)+ '\n'
		m += 'speaker_id: ' + str(self.speaker_id)+ '\n'
		m += 'filename: ' + str(self.filename)+ '\n'
		m += 'ok: ' + str(self.ok)+ '\n'
		return m
		

	def handle_tier(self):
		self.gender, self.language, self.speaker_id = False,False,False
		self.unknown_tier_item = []
		self.background = True if 'background' in self.tier else False
		if self.background: return

		for item in self.tier.split('/'):
			if item[:2] == 'sp': self.speaker_id = item
			elif item in ['male', 'female']: self.gender = item
			elif item in ['fr', 'nl']: self.language = item
			else: self.unknown_tier_item.append(item)

		self.ok = True if len(self.unknown_tier_item) == 0 else False

			
def make_tableline_from_dict(d):
	if type(d) == str: 
		try: eval(d)
		except: raise ValueError('could not convert to TableLine dict '+ d)
	x = TableLine('')
	x.__dict__ = d
	return x

		
		
	
