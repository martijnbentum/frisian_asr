# https://huggingface.co/blog/wav2vec2-with-ngram
# based on the above tutorial, wav2vec2 with ngram decoding
# transformer based decoding can give a futher improvement

from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ProcessorWithLM
from transformers import Wav2Vec2ForCTC
import torch

lm_filename = '/vol/tensusers/mbentum/FRISIAN_ASR/LM/FAME_council_mix_50.arpa'
path = '../wav2vec2_test/checkpoint-final/'

'''
the path directory should contain the following:
config.json 						(unknown)
preprocessor_config.json 			(load the tokenizer /processor)
pytorch_model.bin 					(model weights)
vocab.json 							(needed to load processor)
-----------
'''

def load_pretrained_processor(recognizer_dir = ''):
	if not recognizer_dir: recognizer_dir = path
	processor = Wav2Vec2Processor.from_pretrained(recognizer_dir)
	return processor

def load_and_sort_vocab(processor = None):
	if not processor:
		print('no processor provided, using default at:',path)
		processor = load_pretrained_processor(path)
	vocab = processor.tokenizer.get_vocab()
	sorted_vocab = sorted(vocab.items(), key = lambda item: item[1])
	sorted_vocab = {k.lower():v for k, v in sorted_vocab}
	return sorted_vocab

def make_ctc_decoder(sorted_vocab = None):
	if not sorted_vocab:
		print('no vocab provided, using default based on processor form:',path)
		processor = load_pretrained_processor(path)
		sorted_vocab = load_and_sort_vocab(processor)
	labels = list(sorted_vocab.keys())
	decoder = build_ctcdecoder(labels=labels,kenlm_model_path=lm_filename)
	return decoder
		
def make_processor_with_lm(recognizer_dir = '', save = True):
	if not recognizer_dir: 
		print('no directory provided, using default at:',path)
		recognizer_dir = path
	processor = load_pretrained_processor(recognizer_dir)
	sorted_vocab = load_and_sort_vocab(processor)
	decoder = make_ctc_decoder(sorted_vocab)
	processor_with_lm = Wav2Vec2ProcessorWithLM(
		feature_extractor = processor.feature_extractor,
		tokenizer = processor.tokenizer,
		decoder = decoder
	)
	if save:
		processor_with_lm.save_pretrained(recognizer_dir)
	return processor_with_lm

def load_processor_with_lm(recognizer_dir = ''):
	if not recognizer_dir: recognizer_dir = path
	return Wav2Vec2ProcessorWithLM.from_pretrained(recognizer_dir)
	

def load_model(recognizer_dir = ''):
	if not recognizer_dir: recognizer_dir = path
	# should .to('cuda') be added below?
	model = Wav2Vec2ForCTC.from_pretrained(recognizer_dir)
	return model




		
	
