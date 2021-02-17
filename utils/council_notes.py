import glob
import os
import progressbar as pb
import sys
import textract
from texts.models import Text, Language, TextType, Source


source = Source.objects.get(name='frysian council notes')
texttype= TextType.objects.get(name='council notes')
dutch = Language.objects.get(name='Dutch')
frysian = Language.objects.get(name='Frysian')

directory = '/vol/tensusers/mbentum/FRYSIAN_ASR/frysian_council_notes/'
fn = []

language_dict = {
	'enkel in fries':[frysian],
	'enkel in nederlands':[dutch],
	'mix van fries en nederlands in zelfde document':[frysian,dutch],
	'notulen waarin frl en nl in combi':[frysian,dutch],
	'zelfde documenten in fries en nederlands':[frysian,dutch],
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
			raw_text = textract.process(filename)
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

'''

	filename = models.CharField(max_length=1000,default='',unique=True)
	filetype = models.CharField(max_length=10,default='')
	raw_text = models.TextField(default='')
	clean_text = models.TextField(default='')
	titel = models.CharField(max_length=1000,default='')
	main_language =  models.ForeignKey(Language, **dargs,related_name = 'main_language')
	all_languages = models.ManyToManyField(Language,blank=True, related_name = 'all_languages')
	multiple_languages = models.BooleanField()
	source=  models.ForeignKey(Source, **dargs)
	text_type = models.ForeignKey(TextType, **dargs)
	error= models.BooleanField(default = False)
	note= models.CharField(max_length=300,default='')

'''



