# code taken from https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
# code is slightly altered from notebook style to function style

#before running this code on ponyland first do:
# export CUDA_VISIBLE_DEVICES=0
#
# export CUDA_VISIBLE_DEVICES=1
# depending on which device you want to run the code


from datasets import load_dataset, load_metric, Audio
from datasets import ClassLabel
import random
import pandas as pd
import re
import json
import numpy as np

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer

from .wav2vec2_data import DataCollatorCTCWithPadding

cache_dir = '/vol/tensusers/mbentum/huggingface_cache/'
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'
recognizer_dir = '../wav2vec2_turkish/'


def load_datasets(do_remove_special_characters = True,
	do_replace_hatted_characters = True,resample = True):
	common_voice_train = load_dataset("common_voice", "tr", 
		split="train+validation",cache_dir = cache_dir)
	common_voice_test = load_dataset("common_voice", "tr", 
		split="test", cache_dir = cache_dir)
	c_names = 'accent,age,client_id,down_votes,gender'
	c_names += ',locale,segment,up_votes'
	c_names = c_names.split(',')
	common_voice_train = common_voice_train.remove_columns(c_names)
	common_voice_test = common_voice_test.remove_columns(c_names)
	if do_remove_special_characters:
		common_voice_train = common_voice_train.map(remove_special_characters)
		common_voice_test = common_voice_test.map(remove_special_characters)
	if do_remove_special_characters:
		common_voice_train = common_voice_train.map(replace_hatted_characters)
		common_voice_test = common_voice_test.map(replace_hatted_characters)
	if resample:
		common_voice_train = common_voice_train.cast_column('audio',
			Audio(sampling_rate=16_000))
		common_voice_test= common_voice_test.cast_column('audio',
			Audio(sampling_rate=16_000))
	return common_voice_train, common_voice_test

def show_random_elements(dataset, num_examples=10):
	assert num_examples <= len(dataset)
	picks = []
	for _ in range(num_examples):
		pick = random.randint(0,len(dataset)-1)
		while pick in picks:
			pick = random.randint(0, len(dataset)-1)
		picks.append(pick)
	df = pd.DataFrame(dataset[picks])
	return df

def remove_special_characters(batch):
	t = re.sub(chars_to_remove_regex,'',batch['sentence'])
	batch['sentence'] = t.lower()
	return batch

def replace_hatted_characters(batch):
	batch['sentence'] = re.sub('[â]', 'a', batch["sentence"])
	batch['sentence'] = re.sub('[î]', 'i', batch["sentence"])
	batch['sentence'] = re.sub('[ô]', 'o', batch["sentence"])
	batch['sentence'] = re.sub('[û]', 'u', batch["sentence"])
	return batch

def extract_all_chars(batch):
	all_text = ' '.join(batch['sentence'])
	vocab = list(set(all_text))
	return {'vocab':[vocab], 'all_text':[all_text]}

def make_vocab(datasets = [], save = False):
	if not datasets: train, test = load_datasets()
	vocab_list = []
	for dataset in datasets:
		v = dataset.map(extract_all_chars,batched = True, batch_size=-1,
			keep_in_memory=True, remove_columns=dataset.column_names)
		for char in v['vocab'][0]:
			if char not in vocab_list: vocab_list.append(char)
	vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
	vocab_dict['|'] = vocab_dict[' ']
	del vocab_dict[' ']
	vocab_dict['[UNK]'] = len(vocab_dict)
	vocab_dict['[PAD]'] = len(vocab_dict)
	with open(recognizer_dir + 'vocab.json','w') as vocab_file:
		json.dump(vocab_dict,vocab_file)
	return vocab_dict

def load_tokenizer():
	tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(recognizer_dir, 
		unk_token='[UNK]',pad_token='[PAD]',word_delemiter_token='|',
		cache_dir = cache_dir)
	return tokenizer

def load_feature_extractor():
	feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, 
		sampling_rate=16000, padding_value=0.0, do_normalize=True, 
		return_attention_mask=True)
	return feature_extractor

def load_processor():
	tokenizer = load_tokenizer()
	feature_extractor = load_feature_extractor()
	processor = Wav2Vec2Processor(feature_extractor=feature_extractor, 
		tokenizer=tokenizer)
	return processor

def prepare_dataset(batch):
	audio = batch['audio']
	batch['input_values'] = processor(audio['array'],
		sampling_rate = audio['sampling_rate']).input_values[0]
	with processor.as_target_processor():
		batch['labels'] = processor(batch['sentence']).input_ids
	return batch

def prepare_datasets(train= [],test= []):
	if not train or not test: train, test = load_datasets()
	train = train.map(prepare_dataset,remove_columns=train.column_names)
	test = test.map(prepare_dataset,remove_columns=test.column_names)
	return train, test

def compute_metrics(pred):
	pred_logits = pred.predictions
	pred_ids = np.argmax(pred_logits, axis=-1)
	
	pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
	pred_str = processor.batch_decode(pred_ids)
	# we do not want to group tokens when computing the metrics
	label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
	
	wer = wer_metric.compute(predictions=pred_str, references=label_str)
	
	return {'wer':wer}

def load_model():
	model = Wav2Vec2ForCTC.from_pretrained(
		'facebook/wav2vec2-xls-r-300m',
		attention_dropout=0.0,
		hidden_dropout=0.0,
		feat_proj_dropout=0.0,
		mask_time_prob=0.05,
		layerdrop=0.0,
		ctc_loss_reduction='mean',
		pad_token_id=processor.tokenizer.pad_token_id,
		vocab_size=len(processor.tokenizer),
		cache_dir=cache_dir,
	)
	model.freeze_feature_extractor()
	return model

def load_training_arguments():
	training_arguments = TrainingArguments(
		output_dir=recognizer_dir,
		group_by_length=True,
		per_device_train_batch_size=16,
		gradient_accumulation_steps=2,
		evaluation_strategy='steps',
		num_train_epochs=30,
		gradient_checkpointing=True,
		fp16=True,
		save_steps=400,
		eval_steps=400,
		logging_steps=400,
		learning_rate=3e-4,
		warmup_steps=500,
		save_total_limit=2,
		push_to_hub=False
	)
	return training_arguments

processor = load_processor()
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric('wer')	
training_arguments = load_training_arguments()


def load_trainer():
	model = load_model()
	train,test = prepare_datasets()
	trainer = Trainer(
		model=model,
		data_collator=data_collator,
		args=training_arguments,
		compute_metrics= compute_metrics,
		train_dataset=train,
		eval_dataset=test,
		tokenizer=processor.feature_extractor,
	)
	return trainer







