'''
loading and cleaning data for wav2vec2.0 training
based on:https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
'''

from dataclasses import dataclass, field
from datasets import load_dataset, Audio
import json
import random
import re
import torch
from transformers import Wav2Vec2Processor
from .wav2vec2_make_file import load_audio_section
from typing import Any, Dict, List, Optional, Union

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\\u200b\$\’\´\`\'0-9]'
letter_diacritics_removal_dict={'ö':'o','ä':'a','à':'a','ü':'u','ù':'u','ó':'o'}
letter_diacritics_removal_dict.update({'è':'e','ï':'i','ë':'e','é':'e'})

vocab_dir= '../wav2vec2data/'
cache_dir = '../wav2vec2data/'
vocab_filename = vocab_dir + 'vocab.json'

def load_common_voice_frisian(remove_extra_columns = True, clean_text = True,
	cache_dir = cache_dir, split= 'train', sampling_rate = 16000):
	'''
	loads common voice frisian data.
	remove... 	removes not needed columns
	clean... 	normalizes the text
	cache... 	specifies the cache_dir, if empty common voice data is loaded
	splits 		splits data in training validation data, by default groups
				everything together. If you want to split in training
				and test pass: train,test
	'''
	if split == 'train,test':split={'train':'train+validation','test':'test'}
	elif split == 'train,test,validation':split={'train':'train',
		'validation':'validation','test':'test'}
	else:split = {'train':'train+validation+test'}
	d = {}
	for key,value in split.items():
		d[key]= load_dataset('common_voice','fy-NL',
			split=value,cache_dir=cache_dir)
		if remove_extra_columns:
			d[key] = d[key].remove_columns(["accent", "age","client_id",
				"down_votes", "gender", "locale", "segment", "up_votes"])
		if clean_text:d[key] = clean_dataset_transcriptions(d[key])
		d[key] = resample_audio(d[key], sampling_rate= sampling_rate)
	return d

def _load_audio_council(item):
	st,et = item['start_time'],item['end_time']
	filename = item['audiofilename']
	item['audio'] = {}
	item['audio']['array'] = load_audio_section(st,et,filename,sampling_rate=16000)
	item['audio']['sampling_rate'] = 16000
	return item

def load_council(clean_text = False, cache_dir = cache_dir,split='train,dev,test',
	load_audio = True):
	''' 
	load council
	council dataset texts are already cleaned in w2v2 make_file.
	'''
	d = {}
	for x in split.split(','):
		data_file = cache_dir+'council_'+x+'.json'
		print(cache_dir+'council_'+x+'.json')
		d[x] = load_dataset('json',data_files=cache_dir+'council_'+x+'.json',
			field='data',
			cache_dir=cache_dir)
		if load_audio:
			d[x] = d[x].map(_load_audio_council)
		if clean_text:d[x] = clean_dataset_transcriptions(d[x])
	for key in d.keys():
		d[key] = d[key]['train']
	return d
	

def resample_audio(dataset,sampling_rate=16000):
	return dataset.cast_column('audio',Audio(sampling_rate=sampling_rate))
		
def show_sample_dataset(dataset, return_sample = False, verbose=True):
	sample = random.sample(list(dataset.to_pandas().sentence),10)
	if verbose:print('\n'.join(sample))
	if return_sample or not verbose:return sample

def remove_special_characters(item):
	s = item['sentence']
	s = re.sub(chars_to_remove_regex,'',s).lower()
	for key, value in letter_diacritics_removal_dict.items():
		s = re.sub(key,value,s)
	item['sentence']
	return item

def clean_dataset_transcriptions(dataset):
	dataset = dataset.map(remove_special_characters)
	return dataset




@dataclass
class DataCollatorCTCWithPadding:
	'''
	Data collator that will dynamically pad the inputs received.

	processor 	:class:`~transformers.Wav2Vec2Processor`
				The processor used for proccessing the data.
	padding 	:obj:`bool`, :obj:`str` or 
				:class:`~transformers.tokenization_utils_base.PaddingStrategy`, 
				`optional`, defaults to :obj:`True`:

				Select a strategy to pad the returned sequences (according to 
				the model's padding side and padding index)
				among:

				* :obj:`True` or :obj:`'longest'`: 
				Pad to the longest sequence in the batch (or no padding 
				if only a single sequence if provided).

				* :obj:`'max_length'`: Pad to a maximum length specified 
				with the argument :obj:`max_length` or to the
				maximum acceptable input length for the model if that 
				argument is not provided.

				* :obj:`False` or :obj:`'do_not_pad'` (default): No padding 
				(i.e., can output a batch with sequences of
				different lengths).
	'''

	processor: Wav2Vec2Processor
	padding: Union[bool, str] = True


	def __call__(self, features: List[Dict[str, Union[List[int], 
		torch.Tensor]]]) -> Dict[str, torch.Tensor]:
		'''
		split inputs and labels since they have to be of different lenghts 
		and need different padding methods
		'''
		input_features = [{"input_values": f["input_values"]} for f in features]
		label_features = [{"input_ids": f["labels"]} for f in features]

		batch = self.processor.pad(input_features,padding=self.padding,
			return_tensors="pt")
	
		with self.processor.as_target_processor():
			labels_batch = self.processor.pad(label_features,
				padding=self.padding,return_tensors="pt")
	
		# replace padding with -100 to ignore loss correctly
		labels = labels_batch["input_ids"].masked_fill(
			labels_batch.attention_mask.ne(1), -100)

		batch["labels"] = labels

		return batch
