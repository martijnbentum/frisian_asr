import glob
import os
import progressbar as pb
import sys
import textract
from texts.models import Text, Language, TextType, Source
from utils import language_detection as ld


source = Source.objects.get(name='frisian council notes')
texttype= TextType.objects.get(name='council notes')
dutch = Language.objects.get(name='Dutch')
frisian = Language.objects.get(name='Frisian')

directory = '/vol/tensusers/mbentum/FRISIAN_ASR/frisian_council_notes/'
fn = []

language_dict = {
	'enkel in fries':[frisian],
	'enkel in nederlands':[dutch],
	'mix van fries en nederlands in zelfde document':[frisian,dutch],
	'notulen waarin frl en nl in combi':[frisian,dutch],
	'zelfde documenten in fries en nederlands':[frisian,dutch],
	'NA':[]
}

def get_filenames():
	global fn
	if fn == []:fn = glob.glob(directory + '**/*',recursive = True)
	files = [f for f in fn if os.path.isfile(f)]
	return files

def get_directories():
	global fn
	if fn == []:fn = glob.glob(directory + '**/*',recursive = True)
	directories = [f for f in fn if os.path.isdir(f)]
	return directories

def handle_files(save = False):
	files = get_filenames()
	bar = pb.ProgressBar()
	bar(range(len(files)))
	for i,filename in enumerate(files):
		bar.update(i)
		language_folder= find_language_folder(filename)
		languages = language_dict[language_folder]
		multiple_languages = True if len(languages) > 1 else False
		main_language = languages[0] if len(languages) == 1 else None
		filetype = filename.split('.')[-1]
		incorrect_ft= False if filetype.lower() in 'doc,docx,pdf,txt,rtf'.split(',') else True
		if incorrect_ft: continue
		try: 
			raw_text = textract.process(filename).decode()
			error = False	
		except: 
			raw_text = ''
			error = True
		t = Text(filename = filename, filetype = filetype, raw_text = raw_text, 
			main_language = main_language, multiple_languages = multiple_languages, 
			source = source, text_type = texttype, error = error)
		if save:
			try:t.save()
			except:
				print('could not save:',t)
				print(sys.exc_info())
				continue
			for language in languages:
				t.all_languages.add(language)
			
		
			


def find_language_folder(d):
	folders = d.split('/')
	for f in folders:
		f = f.lower()
		if 'fries' in f or 'nederlands' in f or 'frl' in f or 'NL' in f:
			return f
	return 'NA'


def export_unk_files(goal_dir = '../unk_files/'):
	t = Text.objects.filter(main_language = None,multiple_languages=False)
	for i,x in enumerate(t):
		filename = str(i+1) + '___' + x.filename.split('/')[-1].split('.')[0]
		text = x.raw_text
		with open(goal_dir + filename + '.txt','w') as fout:
			fout.write(text)

def export_mixed_files(goal_dir = '../mixed_files/'):
	t = Text.objects.filter(multiple_languages=True)
	for i,x in enumerate(t):
		filename = str(i+1) + '___' + x.filename.split('/')[-1].split('.')[0]
		text = x.raw_text
		with open(goal_dir + filename + '.txt','w') as fout:
			fout.write(text)

def make_text_frisian_minutes(save = False):
	d = {}
	directory = '/vol/tensusers/mbentum/FRISIAN_ASR/frisian_minutes/' 
	fn = glob.glob( directory + '*.pdf')
	for f in fn:
		raw_text = textract.process(f).decode()
		d[f] = raw_text
		filename = f.split('/')[-1].split('.')[0] + '.txt'
		if save:
			with open(directory + 'txt/' + filename,'w') as fout:
				fout.write(raw_text)
	return d



def load_frisian_minutes_in_db(d = None, save = False):
	'''set of pdf's scanned by jelske.'''
	source = Source.objects.get(name='frisian council minutes')
	texttype= TextType.objects.get(name='council notes')
	output = []
	language_dict = {'Frisian':frisian,'Dutch':dutch}
	c = ld.load('Dutch-Frisian_sentences')
	if not d: d = make_text_frisian_minutes()
	for f,text in d.items():
		print(f)
		t = Text.objects.filter(filename__exact=f)
		if t:
			print(f.split('/')[-1],'already found in database',9)#t)
			output.append(t)
			continue
		o = c.predict_text(text)
		main_language = language_dict[o.main_language_overall_prediction]
		multiple_languages = True
		t = Text(filename = f, filetype = 'pdf', source = source, text_type = texttype,
			raw_text = text,main_language = main_language, multiple_languages = multiple_languages) 
		output.append(t)
		if save:
			try:t.save()
			except:
				print('could not save:',10)#t)
				print(sys.exc_info())
				continue
			for language in [frisian,dutch]:
				t.all_languages.add(language)
	return output

