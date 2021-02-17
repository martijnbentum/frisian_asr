from django.db import models

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
	filename = models.CharField(max_length=1000,default='',unique=True)
	filetype = models.CharField(max_length=10,default='')
	raw_text = models.TextField(default='')
	clean_text = models.TextField(default='')
	titel = models.CharField(max_length=1000,default='')
	main_language =  models.ForeignKey(Language, **dargs,related_name = 'main_language')
	all_languages = models.ManyToManyField(Language,blank=True, related_name = 'all_languages')
	multiple_languages = models.BooleanField(default = False)
	source=  models.ForeignKey(Source, **dargs)
	text_type = models.ForeignKey(TextType, **dargs)
	error= models.BooleanField(default = False)
	note= models.CharField(max_length=300,default='')
	speaker_id = models.CharField(max_length=300,default='')
	start_time = models.FloatField(default = None,null=True)
	end_time = models.FloatField(default = None,null=True)
	wav_filename = models.CharField(max_length=1000,default='')

	def __repr__(self):
		return self.filename.split('/')[-1] + ' ' + self.titel

	def raw_word_count(self):
		return len(self.raw_text.split(' '))

