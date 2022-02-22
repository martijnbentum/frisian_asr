import torch
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ProcessorWithLM

from utils import wav2vec2_lm as wlm
from utils import wav2vec2_model as wm

from jiwer import wer
import os
import pickle
import random

data = '../wav2vec2data/'
path = "../wav2vec2data/council_delim_fix/current_council/"
path_lm = "../wav2vec2data/council_delim_fix/current_council_lm/"

class Decoder:
	def __init__(self, recognizer_dir = '', use_lm = False, make_lm = False,
		use_cuda = True, load_model= True):
		if not recognizer_dir: recognizer_dir = path
		if not recognizer_dir.endswith('/'): recognizer_dir += '/'
		self.recognizer_dir = recognizer_dir
		self.logits_dir = recognizer_dir + 'logits/'
		self.use_lm = use_lm
		self.make_lm = make_lm
		self.use_cuda = use_cuda
		self.load_model = load_model
		self.handle_lm()
		self.load()
		
	def handle_lm(self):
		if not self.use_lm: return
		self.lm_dir = self.recognizer_dir + 'language_model/'
		if self.make_lm: 
			self.processor = wlm.make_processor_with_lm(self.recognizer_dir)
			return
		if self.use_lm and not os.path.isdir(self.lm_dir):
			m=self.recognizer_dir+' does not contain language model directory'
			m += 'set make_lm to true or specify another recognizer directory'
			raise ValueError(m)
		else: self.processor = wlm.load_processor_with_lm(self.recognizer_dir)

	def load(self):
		if not hasattr(self, 'processor'):
			self.processor=Wav2Vec2Processor.from_pretrained(self.recognizer_dir)
		if self.load_model:
			self.model = Wav2Vec2ForCTC.from_pretrained(self.recognizer_dir)
			if self.use_cuda:
				self.model = self.model.to("cuda")

	def _audio2inputs(self, audio):
		return load_inputs(audio,self.processor)

	def _inputs2logits(self,inputs):
		return inputs2logits(inputs, self.model, self.use_cuda)
	
	def _logits2labels(self,logits):
		return logits2labels(logits)

	def lm_logits2text(self,logits):
		if 'cuda' in logits.__repr__(): logits = logits.cpu()
		return self.processor.batch_decode(logits.detach().numpy(),
			num_processes=1).text

	def audio2logits(self,audio, filename = ''):
		inputs = self._audio2inputs(audio)
		logits = self._inputs2logits(inputs)
		if filename:
			if not os.path.isdir(self.logits_dir): os.mkdir(self.logits_dir)
			with open(self.logits_dir + filename, 'wb') as fout:
				pickle.dump(logits, fout)
		return logits

	def audio2text(self,audio):
		logits = self.audio2logits(audio)
		if self.use_lm: return self.lm_logits2text(logits)
		labels = self._logits2labels(logits)
		return self.processor.decode(labels)

def get_preprocessed_council_dev_test():
	d = wm.preprocess_council()
	dev = d['dev']
	test = d['test']
	return dev, test

def make_council_dev_test_ground_truth_sentences(directory = data, 
		overwrite = False):
	if not directory.endswith('/'): directory += '/'
	d = wm.load_council()
	dev_gt= [x['sentence'] for x in d['dev']]
	test_gt= [x['sentence'] for x in d['test']]
	filename_dev = directory + 'dev_ground_truth.txt'
	filename_test= directory + 'test_ground_truth.txt'
	if not os.path.isfile(filename_dev) or overwrite:
		with open(filename_dev, 'w') as fout:
			fout.write('\n'.join(dev_gt))
	if not os.path.isfile(filename_test) or overwrite:
		with open(filename_test, 'w') as fout:
			fout.write('\n'.join(test_gt))
	return dev_gt, test_gt

def load_council_ground_truth(directory = data):
	dev_gt = open(data + 'dev_ground_truth.txt').read().split('\n')
	test_gt = open(data + 'test_ground_truth.txt').read().split('\n')
	return dev_gt, test_gt

def council_dev_test_to_text_no_lm(recognizer_dir = path, save = True):
	dev, test = get_preprocessed_council_dev_test()
	dev_output, test_output = [], []
	decoder = Decoder(recognizer_dir, use_lm = False)
	for x in dev:
		dev_output.append(decoder.audio2text(x['input_values']))
	for x in test:
		test_output.append(decoder.audio2text(x['input_values']))
	if save:
		filename_dev = decoder.recognizer_dir + 'dev_pred.txt'
		filename_test= decoder.recognizer_dir + 'test_pred.txt'
		with open(filename_dev, 'w') as fout:
			fout.write('\n'.join(dev_output))
		with open(filename_test, 'w') as fout:
			fout.write('\n'.join(test_output))
	return dev_output, test_output


def _council_dev_test_to_logits_with_lm(recognizer_dir = path_lm):
	'''might be OBSOLETE, trying to step lm recognition to side step 
	memory alloc problems, because lm decode uses multiprocessing
	duplicating everything, current option is to use 1 process

	creates the logits for all dev and test items
	'''
	dev, test = get_preprocessed_council_dev_test()
	dev_logits, test_logits= [], []
	decoder = Decoder(recognizer_dir, use_lm = True, make_lm = False)
	for i,x in enumerate(dev):
		print('dev',i)
		logits=decoder.audio2logits(x['input_values'],filename='dev-'+str(i))
	for i,x in enumerate(test):
		print('test',i)
		logits=decoder.audio2logits(x['input_values'],filename='test-'+str(i))
	return dev_logits, test_logits
	
def council_dev_test_to_text_with_lm(recognizer_dir = path_lm, save = True):
	dev, test = get_preprocessed_council_dev_test()
	dev_output, test_output = [], []
	decoder = Decoder(recognizer_dir, use_lm = True)
	for x in dev:
		dev_output.append(decoder.audio2text(x['input_values'])[0])
	for x in test:
		test_output.append(decoder.audio2text(x['input_values'])[0])
	if save:
		filename_dev = decoder.recognizer_dir + 'lm_dev_pred.txt'
		filename_test= decoder.recognizer_dir + 'lm_test_pred.txt'
		with open(filename_dev, 'w') as fout:
			fout.write('\n'.join(dev_output))
		with open(filename_test, 'w') as fout:
			fout.write('\n'.join(test_output))
	return dev_output, test_output

def load_council_prediction_no_lm(recognizer_dir = path):
	dev_pred = open(recognizer_dir + 'dev_pred.txt').read().split('\n')
	test_pred = open(recognizer_dir + 'test_pred.txt').read().split('\n')
	return dev_pred, test_pred
def load_council_prediction_with_lm(recognizer_dir = path_lm):
	dev_pred = open(recognizer_dir + 'lm_dev_pred.txt').read().split('\n')
	test_pred = open(recognizer_dir + 'lm_test_pred.txt').read().split('\n')
	return dev_pred, test_pred

def compute_wer_council():
	dev_gt, test_gt = load_council_ground_truth()
	dev_pred, test_pred = load_council_prediction_no_lm()
	dev_pred_lm, test_pred_lm = load_council_prediction_with_lm()
	print('WER, dev')
	print(wer(dev_gt,dev_pred), 'no lm')
	print(wer(dev_gt,dev_pred_lm), 'with lm')
	print('-----')
	print('WER, test:')
	print(wer(test_gt,test_pred), 'no lm')
	print(wer(test_gt,test_pred_lm), 'with lm')

def random_sample_council_gt_pred_comparison(n = 10):
	dev_gt, test_gt = load_council_ground_truth()
	dev_pred, test_pred = load_council_prediction_no_lm()
	dev_pred_lm, test_pred_lm = load_council_prediction_with_lm()
	dev_indices = random.sample(range(len(dev_gt)),n)
	test_indices = random.sample(range(len(test_gt)),n)
	print('dev:')
	for i in dev_indices:
		print('gt: ',dev_gt[i])
		print('nlm:',dev_pred[i])
		print('lm: ',dev_pred_lm[i])
		print('---')
	print('test:')
	for i in test_indices:
		print('gt: ',test_gt[i])
		print('nlm:',test_pred[i])
		print('lm: ',test_pred_lm[i])
		print('---')
	


def load_inputs(audio,processor):
	return processor(audio,return_tensors='pt',sampling_rate=16_000)

def inputs2logits(inputs, model, cuda = True):
	if cuda:
		return model(inputs.input_values.to('cuda')).logits
	return model(inputs.input_values).logits

def logits2labels(logits):
	return torch.argmax(logits, dim=-1)[0]

		
