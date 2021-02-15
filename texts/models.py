from django.db import models

# Create your models here.

class Language(models.Model):
	name= models.CharField(max_length=1000,default='',unique =True)
	abbreviation= models.CharField(max_length=1000,default='')

class Source(models.Model):
	name= models.CharField(max_length=1000,default='',unique=True)
	description = models.TextField(default='')

class TextType(models.Model):
	'''model to descripte text, e.g. transcription, council notes, wikipedia article, 
	automatic transcription'''
	name= models.CharField(max_length=1000,default='',unique=True)
	description = models.TextField(default='')


class Text(models.Model):
	dargs = {'on_delete':models.SET_NULL,'blank':True,'null':True}
	org_filename = models.CharField(max_length=1000,default='',unique=True)
	org_filetype = models.CharField(max_length=1000,default='')
	raw_text = models.TextField(default='')
	clean_text = models.TextField(default='')
	titel = models.CharField(max_length=1000,default='')
	main_language =  models.ForeignKey(Language, **dargs,related_name = 'main_language')
	all_languages = models.ManyToManyField(Language,blank=True, related_name = 'all_languages')
	multiple_languages = models.BooleanField()
	source=  models.ForeignKey(Source, **dargs)
	text_type = models.ForeignKey(TextType, **dargs)

