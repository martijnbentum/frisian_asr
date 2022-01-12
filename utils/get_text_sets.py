from collections import Counter
from texts.models import Text
from .wav2vec2_data import remove_special_characters
import re

f = Text.objects.filter
council_name = 'frisian council transcripts'
fame_name = 'frisian radio broadcasts'
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\<\>\“\%\‘\”\�\\u200b\$\’\´\`\'0-9]'
letter_diacritics_removal_dict={'ö':'o','ä':'a','à':'a','ü':'u','ù':'u','ó':'o'}
letter_diacritics_removal_dict.update({'è':'e','ï':'i','ë':'e','é':'e'})


def remove_special_characters(s):
	s = re.sub(chars_to_remove_regex,'',s).lower()
	for key, value in letter_diacritics_removal_dict.items():
		s = re.sub(key,value,s)
	return s

def get_council_texts():
	return f(source__name=council_name)

def get_fame_texts():
	return f(source__name=fame_name)

def get_train_texts():
	texts = f(source__name=council_name)|f(source__name=fame_name)
	train = texts.filter(partition='train')
	return f(source__name=council_name)

def get_council_fame_texts():
	return f(source__name=council_name)|f(source__name=fame_name)

def texts_to_text_without_tags(texts):
	output = []
	for x in texts:
		output.append(x.text_without_tags)
	return output

def texts_to_letter_dict(texts, clean_text = remove_special_characters):
	s = ' '.join(texts_to_text_without_tags(texts))
	if clean_text: s = clean_text(s)
	return Counter(s)
	

		

