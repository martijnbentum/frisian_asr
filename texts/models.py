from django.db import models
import re

# Create your models here.

class Language(models.Model):
	name= models.CharField(max_length=1000,default='',unique =True)
	abbreviation= models.CharField(max_length=1000,default='')

	def __repr__(self):
		return self.name

class Source(models.Model):
	name= models.CharField(max_length=1000,default='',unique=True)
	description = models.TextField(default='')

	def __repr__(self):
		return self.name

class TextType(models.Model):
	'''model to describe text, e.g. transcription, council notes, wikipedia article, 
	automatic transcription'''
	name= models.CharField(max_length=1000,default='',unique=True)
	description = models.TextField(default='')

	def __repr__(self):
		return self.name



class Text(models.Model):
	dargs = {'on_delete':models.SET_NULL,'blank':True,'null':True}
	filename = models.CharField(max_length=1000,default='')
	filetype = models.CharField(max_length=10,default='')
	raw_text = models.TextField(default='')
	clean_text = models.TextField(default='')
	transcription_meta= models.TextField(default='')
	main_language =  models.ForeignKey(Language, **dargs,
		related_name = 'main_language')
	all_languages = models.ManyToManyField(Language,blank=True, 
		related_name = 'all_languages')
	multiple_languages = models.BooleanField(default = False)
	source=  models.ForeignKey(Source, **dargs)
	text_type = models.ForeignKey(TextType, **dargs)
	error= models.BooleanField(default = False)
	note= models.CharField(max_length=300,default='')
	speaker_id = models.CharField(max_length=300,default='')
	speaker_gender = models.CharField(max_length=300,default='')
	start_time = models.FloatField(default = None,null=True)
	end_time = models.FloatField(default = None,null=True)
	wav_filename = models.CharField(max_length=1000,default='')
	title = models.CharField(max_length=300,default='')
	file_id= models.CharField(max_length=100,default='')
	partition = models.CharField(max_length=20,default='')
	duration = models.FloatField(default = None,null=True)
	text_without_tags = models.TextField(default='')
	n_words_text_without_tags = models.PositiveIntegerField(null=True,blank=True)

	def __repr__(self):
		# f = self.filename.split('/')[-1] if self.filename else ''
		try:s = self.source.name.ljust(27)
		except:s =''.ljust(27)
		if self.clean_text: text = self.clean_text
		elif self.raw_text: text = self.raw_text
		else: text = ''
		text = re.sub('\s+',' ',text[:500])
		t = text[:60] + '...' if len(text) > 60 else text
		m =  s + ' | ' + t.ljust(65) + ' | ' + str(self.raw_word_count()) + ' words'
		# l = self.languages
		# if l: m += ' | ' + l
		return m
		

	def raw_word_count(self):
		if self.clean_text: text = self.clean_text
		elif self.raw_text: text = self.raw_text
		else: text = ''
		return len(re.sub('\s+',' ',text).split(' '))

	@property
	def languages(self):
		languages = self.all_languages.all()
		if languages: return ', '.join([language.name for language in languages])
		if self.main_language: return self.main_language.name
		return ''

	@property
	def transcription(self):
		from utils.manual_transcriptions  import Transcription
		from utils.make_tables import make_tableline_from_dict
		if not hasattr(self,'_transcription'):
			if self.source.name == 'frisian council transcripts': 
				self._transcription = Transcription(self.transcription_meta,line_type='council')
			elif self.source.name == 'frisian radio broadcasts': 
				tableline = make_tableline_from_dict(self.transcription_meta)
				self._transcription = Transcription(tableline,line_type = 'radio')
			elif self.source.name == 'cgn': 
				tableline = make_tableline_from_dict(self.transcription_meta)
				self._transcription = Transcription(tableline,line_type = 'cgn')
			else: return 'not available, source is not a radio or council transcription'
		return self._transcription

	@property
	def utterance_id(self):
		wf = self.wav_filename.split('.wav')[0]
		utt = wf + '_pk-' + add_zeros(self.pk)
		if self.speaker_id: return self.speaker_id + '@-@' + utt
		else: return wf + '@-@' + utt



def add_zeros(n,output_length = 6):
	sn = str(n)
	return '0' * (output_length - len(sn)) + sn
		
		
