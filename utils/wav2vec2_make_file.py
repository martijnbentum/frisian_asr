from datasets import load_dataset
import json
import librosa 
from texts.models import Text
import re
council_name = 'frisian council transcripts'
fame_name = 'frisian radio broadcasts'

cache_dir = '../wav2vec2data/'
council_wav_dir = '/vol/tensusers3/Frisiansubtitling/COUNCIL/wav/'
fame_wav_dir = '/vol/tensusers/mbentum/FRISIAN_ASR/corpus/fame/wav/'

vocab_dir= '../wav2vec2data/'
vocab_filename = vocab_dir + 'vocab.json'

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\<\>\“\%\‘\”\�\\u200b\$\’\´\`\'0-9]'
letter_diacritics_removal_dict={'ö':'o','ä':'a','à':'a','ü':'u','ù':'u','ó':'o'}
letter_diacritics_removal_dict.update({'è':'e','ï':'i','ë':'e','é':'e'})

def remove_special_characters(s):
	s = re.sub(chars_to_remove_regex,'',s).lower()
	for key, value in letter_diacritics_removal_dict.items():
		s = re.sub(key,value,s)
	return s


def load_audio(filename, sampling_rate=None):
	np_array, sampling_rate = librosa.load(filename,sr=sampling_rate)
	return np_array, sampling_rate

def load_audio_section(start_time,end_time,filename='',audio=None,
	sampling_rate=None):
	if not audio: audio, sampling_rate = load_audio(filename,sampling_rate)
	return audio[int(start_time*sampling_rate):int(end_time*sampling_rate)]

def filter_texts(qs,minimum_duration,maximum_duration, minimum_nwords = None):
	qs = qs.filter(duration__gte = minimum_duration)
	qs = qs.filter(duration__lte = maximum_duration)
	qs = qs.exclude(text_without_tags='')
	if minimum_nwords: 
		qs = qs.exclude(n_words_text_without_tags__lt = minimum_nwords)
	return qs

def get_train_dev_test_texts(minimum_duration = 1,maximum_duration = 7,
	minimum_nwords = 2, enforce_duration_contraints = True):
	f = Text.objects.filter
	texts = f(source__name=council_name)|f(source__name=fame_name)
	train = texts.filter(partition='train')
	council= f(source__name=council_name)
	dev = council.filter(partition='dev')
	test = council.filter(partition='test')
	if enforce_duration_contraints:
		train = filter_texts(train,minimum_duration,maximum_duration,minimum_nwords)
		dev= filter_texts(dev,minimum_duration,maximum_duration,minimum_nwords)
		test= filter_texts(test,minimum_duration,maximum_duration,minimum_nwords)
	return train, dev, test

def _array_to_float_list(array):
	return [float(x) for x in array]

def text_to_dict(text, load_audio = False):
	if text.source.name == council_name: directory = council_wav_dir
	if text.source.name == fame_name: directory = fame_wav_dir + 'train/'
	filename = directory + text.wav_filename
	d = {'pk':text.pk}
	d = {'sentence':remove_special_characters(text.text_without_tags)}
	d.update({'audiofilename':filename})
	if load_audio:
		audio = load_audio_section(text.start_time,text.end_time,filename,
			sampling_rate = 16000)
		d.update({'audio':_array_to_float_list(audio)})
	else:
		d.update({'sampling_rate':16000})
		d.update({'start_time':text.start_time})
		d.update({'end_time':text.end_time})
	return d

def make_json(texts, name, cache_dir = cache_dir, save = True,load_audio=False):
	data = []
	for text in texts:
		data.append(text_to_dict(text,load_audio))
	output = {'data':data}
	if not name.endswith('.json'): name += '.json'
	if save:
		print('saving file to:',cache_dir+name)
		with open(cache_dir + name,'w') as fout:
			json.dump(output,fout)
	return data

def make_train_dev_test(minimum_duration = 1, maximum_duration = 7,minimum_nwords=2,
	enforce_duration_contraints = True, cache_dir = cache_dir, save = True, 
	make_vocab= True):
	output = {}
	train, dev, test = get_train_dev_test_texts(minimum_duration,
		maximum_duration,minimum_nwords,enforce_duration_contraints)
	output['train'] = make_json(train,'council_train',cache_dir,save)
	output['dev'] = make_json(dev,'council_dev',cache_dir,save)
	output['test'] = make_json(test,'council_test',cache_dir,save)
	return output
	

def make_vocab_dict(datasets, save=True):
	sentences = []
	for ds in datasets.values():
		for item in ds:
			sentences.append(item['sentence'])
	sentences = ' '.join(sentences)
	vocab = list(set(sentences))
	vocab_dict = {v: k for k, v in enumerate(sorted(vocab))}
	vocab_dict['[UNK]'] = len(vocab_dict)
	vocab_dict['[PAD]'] = len(vocab_dict)
	vocab_dict['|'] = vocab_dict[' ']
	del vocab_dict[' ']
	if save:
		with open(vocab_filename, 'w') as fout:
			json.dump(vocab_dict,fout)
	return vocab_dict
		
	


	
