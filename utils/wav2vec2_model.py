from datasets import load_metric
from .wav2vec2_data import cache_dir, vocab_dir, load_common_voice_frisian
from .wav2vec2_data import DataCollatorCTCWithPadding, load_council
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
from datetime import datetime
import os

processor = None
wer_metric = load_metric('wer')

def load_tokenizer(vocab_dir = vocab_dir,cache_dir = cache_dir):
	tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(vocab_dir,
		cache_dir = cache_dir, unk_token='[UNK]',pad_token='[PAD]')
	return tokenizer
			
def load_feature_extractor():
	feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, 
		sampling_rate=16000, padding_value=0.0, do_normalize=True, 
		return_attention_mask=True)
	return feature_extractor

def load_processor(vocab_dir= vocab_dir, cache_dir = cache_dir, force = False):
	global processor
	if processor: return processor
	tokenizer = load_tokenizer(vocab_dir,cache_dir)
	feature_extractor = load_feature_extractor()
	processor = Wav2Vec2Processor(feature_extractor=feature_extractor, 
		tokenizer=tokenizer)
	return processor

def preprocess_item(item):
	audio = item['audio']

	item['input_values'] = processor(audio['array'], 
		sampling_rate = audio['sampling_rate']).input_values[0]
	item['input_length'] = len(item['input_values'])

	with processor.as_target_processor():
		item['labels'] = processor(item['sentence']).input_ids
	return item

def preprocess_datasets(datasets,maximum_length = None, sampling_rate = 16000):
	d = datasets
	for key in d.keys():
		column_names = d[key].column_names
		d[key] = d[key].map(preprocess_item, remove_columns= column_names)
		if maximum_length:
			maximum = maximum_length * sampling_rate
			d[key] = d[key].filter(lambda x: x < maximum,
				input_columns=['input_length'])
	return d
	
def preprocess_common_voice_frisian(split= 'train,test', maximum_length=7,
	sampling_rate = 16000):
	processor = load_processor()
	d = load_common_voice_frisian(split = split)
	d = preprocess_datasets(d,maximum_length=maximum_length,
		sampling_rate=sampling_rate)
	return d

def preprocess_council():
	processor = load_processor()
	d = load_council()
	d = preprocess_datasets(d)
	return d

def load_data_collator(): 
	processor = load_processor()
	return DataCollatorCTCWithPadding(processor = processor,padding = True)
		
def compute_metrics(pred):
	processor = load_processor()
	pred_logits = pred.predictions
	pred_ids = np.argmax(pred_logits, axis = -1)
	pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
	pred_str = processor.batch_decode(pred_ids)
	# we do not want to group tokens when computing the metricso
	label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
	wer = wer_metric.compute(predictions=pred_str, references=label_str)
	save_preds_references(pred_str,label_str,wer)
	return {"wer": wer}

def save_preds_references(preds,references,wer):
	wer = str(int(wer * 100))
	d = datetime.now().strftime("%d_%m_%Y_%H_%M")
	filename = cache_dir + 'log_dev_wer_' + wer + '-'+d
	output = []
	for pred, ref in zip(preds,references):
		output.append(pred + '\t' + ref)
	with open(filename,'w') as fout:
		fout.write('\n'.join(output))

def load_model():
	processor = load_processor()
	model = Wav2Vec2ForCTC.from_pretrained(
		"facebook/wav2vec2-xls-r-300m", 
		attention_dropout=0.0,
		hidden_dropout=0.0,
		feat_proj_dropout=0.0,
		mask_time_prob=0.05,
		layerdrop=0.0,
		ctc_loss_reduction="mean", 
		pad_token_id=processor.tokenizer.pad_token_id,
		vocab_size=len(processor.tokenizer),
		cache_dir = cache_dir
	)
	model.freeze_feature_extractor()
	return model

def load_training_arguments(experiment_name= vocab_dir):
	if not os.path.isdir(experiment_name):os.mkdir(experiment_name)
	training_args = TrainingArguments(
		output_dir=experiment_name,
		group_by_length=True,
		per_device_train_batch_size=30,
		gradient_accumulation_steps=2,
		evaluation_strategy="steps",
		num_train_epochs=100,
		gradient_checkpointing=True,
		fp16=True,
		save_steps=1000,
		eval_steps=1000,
		logging_steps=50,
		learning_rate=3e-4,
		warmup_steps=500,
		save_total_limit=6,
		push_to_hub=False,
	)
	return training_args

def load_trainer(experiment_name,model = None, training_args = None, datasets = None,
	train = 'train',evaluate='dev'):
	print('set processor')
	processor = load_processor()
	print('make data collator')
	data_collator = load_data_collator()
	if not model: 
		print('load model')
		model = load_model()
	if not training_args: 
		print('load training arguements')
		training_args = load_training_arguments(experiment_name)
	if not datasets: 
		print('load datasets')
		datasets= preprocess_council()
	print('defining the trainer')
	trainer = Trainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		compute_metrics=compute_metrics,
		train_dataset=datasets[train],
		eval_dataset=datasets[evaluate],
		tokenizer=processor.feature_extractor,
	)
	return trainer

def do_council_training(experiment_name):
	trainer = load_trainer(experiment_name = vocab_dir + experiment_name)
	trainer.train()
	return trainer
