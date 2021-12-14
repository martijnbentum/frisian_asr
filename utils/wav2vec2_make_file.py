from datasets import load_dataset
import json
import librosa 
from texts.models import Text
council_name = 'frisian council transcripts'
fame_name = 'frisian radio broadcasts'

cache_dir = '../wav2vec2data/'
council_wav_dir = '/vol/tensusers3/Frisiansubtitling/COUNCIL/wav/'
fame_wav_dir = '/vol/tensusers/mbentum/FRISIAN_ASR/corpus/fame/wav/'

def load_audio(filename, sampling_rate=None):
	np_array, sampling_rate = librosa.load(filename,sr=sampling_rate)
	return np_array, sampling_rate

def load_audio_section(start_time,end_time,filename='',audio=None,
	sampling_rate=None):
	if not audio: audio, sampling_rate = load_audio(filename,sampling_rate)
	return audio[int(start_time*sampling_rate):int(end_time*sampling_rate)]

def filter_texts(qs,minimum_duration,maximum_duration):
	qs = qs.filter(duration__gt = minimum_duration)
	qs = qs.filter(duration__lt = maximum_duration)
	qs = qs.exclude(text_without_tags='')
	return qs

def get_train_dev_test_texts(minimum_duration = 0.2,maximum_duration = 7,
	enforce_duration_contraints = True):
	f = Text.objects.filter
	texts = f(source__name=council_name)|f(source__name=fame_name)
	train = texts.filter(partition='train')
	council= f(source__name=council_name)
	dev = council.filter(partition='dev')
	test = council.filter(partition='test')
	if enforce_duration_contraints:
		train = filter_texts(train,minimum_duration,maximum_duration)
		dev= filter_texts(dev,minimum_duration,maximum_duration)
		test= filter_texts(test,minimum_duration,maximum_duration)
	return train, dev, test

def _array_to_float_list(array):
	return [float(x) for x in array]

def text_to_dict(text, load_audio = False):
	if text.source.name == council_name: directory = council_wav_dir
	if text.source.name == fame_name: directory = fame_wav_dir + 'train/'
	filename = directory + text.wav_filename
	d = {'pk':text.pk}
	d = {'sentence':text.text_without_tags}
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

def make_train_dev_test(minimum_duration = 0.2, maximum_duration = 7,
	enforce_duration_contraints = True, cache_dir = cache_dir, save = True):
	output = {}
	train, dev, test = get_train_dev_test_texts(minimum_duration,
		maximum_duration,enforce_duration_contraints)
	output['train'] = make_json(train,'train',cache_dir,save)
	output['dev'] = make_json(train,'dev',cache_dir,save)
	output['test'] = make_json(train,'test',cache_dir,save)
	return output
	
		
	


	
