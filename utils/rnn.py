from __future__ import division
import argparse
import sys

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainerx

from . import language_modelling as lm
from . import split_text as st
from collections import Counter

lm_dir = lm.lm_dir


def text2integer(text,vocab,unk = '<unk>',eos = True, sos = False):
	if eos and sos:
		text = '<s> ' + text.replace('\n',' </s> <s> ') + '</s>'
	elif eos:text = text.replace('\n',' </s> ') + '</s>'
	elif sos:text = '<s> ' + text.replace('\n',' <s> ') 
	text = np.array(list(map(vocab.get,text.split(' '))))
	return np.where(text == None, vocab[unk], text)
	
	'''
	for sentence in text.split('\n')
		s = ['<s>']
		s.extend(list(map(vocab.get,sentence.split(' ')))
		s.append
	'''

	
	
#https://docs.chainer.org/en/stable/examples/ptb.html
#https://github.com/chainer/chainer/blob/v7.7.0/examples/ptb/train_ptb.py

# Definition of a recurrent net for language modeling
class RNNForLM(chainer.Chain):
	def __init__(self, n_vocab, n_units):
		super(RNNForLM, self).__init__()
		with self.init_scope():
			self.embed = L.EmbedID(n_vocab, n_units)
			self.l1 = L.LSTM(n_units, n_units)
			self.l2 = L.LSTM(n_units, n_units)
			self.l3 = L.Linear(n_units, n_vocab)

		for param in self.params():
			#? initialize values to start training from ?
			param.array[...] = np.random.uniform(-0.1, 0.1, param.shape)

	def reset_state(self):
		# ?
		self.l1.reset_state()
		self.l2.reset_state()

	def forward(self, x):
		# ? training step ?
		h0 = self.embed(x)
		h1 = self.l1(F.dropout(h0))
		h2 = self.l2(F.dropout(h1))
		y = self.l3(F.dropout(h2))
		return y


# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.
class ParallelSequentialIterator(chainer.dataset.Iterator):

	def __init__(self, dataset, batch_size, repeat=True):
		super(ParallelSequentialIterator, self).__init__()
		self.dataset = dataset
		self.batch_size = batch_size  # batch size
		self.repeat = repeat
		length = len(dataset)
		# Offsets maintain the position of each sequence in the mini-batch.
		self.offsets = [i * length // batch_size for i in range(batch_size)]
		self.reset()

	def reset(self):
		# Number of completed sweeps over the dataset. In this case, it is
		# incremented if every word is visited at least once after the last
		# increment.
		self.epoch = 0
		# True if the epoch is incremented at the last iteration.
		self.is_new_epoch = False
		# NOTE: this is not a count of parameter updates. It is just a count of
		# calls of ``__next__``.
		self.iteration = 0
		# use -1 instead of None internally
		self._previous_epoch_detail = -1.

	def __next__(self):
		# This iterator returns a list representing a mini-batch. Each item
		# indicates a different position in the original sequence. Each item is
		# represented by a pair of two word IDs. The first word is at the
		# "current" position, while the second word at the next position.
		# At each iteration, the iteration count is incremented, which pushes
		# forward the "current" position.
		length = len(self.dataset)
		if not self.repeat and self.iteration * self.batch_size >= length:
			# If not self.repeat, this iterator stops at the end of the first
			# epoch (i.e., when all words are visited once).
			raise StopIteration
		cur_words = self.get_words()
		self._previous_epoch_detail = self.epoch_detail
		self.iteration += 1
		next_words = self.get_words()

		epoch = self.iteration * self.batch_size // length
		self.is_new_epoch = self.epoch < epoch
		if self.is_new_epoch:
			self.epoch = epoch

		return list(zip(cur_words, next_words))


	@property
	def epoch_detail(self):
		# Floating point version of epoch.
		return self.iteration * self.batch_size / len(self.dataset)

	@property
	def previous_epoch_detail(self):
		if self._previous_epoch_detail < 0:
			return None
		return self._previous_epoch_detail

	def get_words(self):
		# It returns a list of current words.
		return [self.dataset[(offset + self.iteration) % len(self.dataset)]
			for offset in self.offsets]

	def serialize(self, serializer):
		# It is important to serialize the state to be recovered on resume.
		self.iteration = serializer('iteration', self.iteration)
		self.epoch = serializer('epoch', self.epoch)
		try:
			self._previous_epoch_detail = serializer(
				'previous_epoch_detail', self._previous_epoch_detail)
		except KeyError:
			# guess previous_epoch_detail for older version
			self._previous_epoch_detail = self.epoch + \
				(self.current_position - self.batch_size) / len(self.dataset)
			if self.epoch_detail > 0:
				self._previous_epoch_detail = max(
					self._previous_epoch_detail, 0.)
			else:
				self._previous_epoch_detail = -1.


class BPTTUpdater(training.updaters.StandardUpdater):

	def __init__(self, train_iter, optimizer, bprop_len, device):
		super(BPTTUpdater, self).__init__(
		train_iter, optimizer, device=device)
		self.bprop_len = bprop_len

	# The core part of the update routine can be customized by overriding.
	def update_core(self):
		loss = 0
		# When we pass one iterator and optimizer to StandardUpdater.__init__,
		# they are automatically named 'main'.
		train_iter = self.get_iterator('main')
		optimizer = self.get_optimizer('main')

		# Progress the dataset iterator for bprop_len words at each iteration.
		for i in range(self.bprop_len):
			# Get the next batch (a list of tuples of two word IDs)
			batch = train_iter.__next__()

			# Concatenate the word IDs to matrices and send them to the device
			# self.converter does this job
			# (it is chainer.dataset.concat_examples by default)
			x, t = self.converter(batch, self.device)

			# Compute the loss at this time step and accumulate it
			loss += optimizer.target(x, t)

		optimizer.target.cleargrads()  # Clear the parameter gradients
		loss.backward()  # Backprop
		loss.unchain_backward()  # Truncate the graph
		optimizer.update()  # Update the parameters


def compute_perplexity(result):
	result['perplexity'] = np.exp(result['main/loss'])
	if 'validation/main/loss' in result:
		result['val_perplexity'] = np.exp(result['validation/main/loss'])


class Train():
	def __init__(self,data,batch_size=20,bproplen = 35, epoch = 39,device = -1,
		grad_clip = 5, out = lm.lm_dir + 'rnn/',resume = '',test=False,unit = 650, 
		model_filename = 'model.npz'):
		''' class to train rnn model
		batch_size  		number of examples in a mini batch
		bproplen 			number or words in each minibatch
		epoch 				number of sweeps over the dataset
		device 				device specifier. Either chainerX device specifier
							or an integer. If non-negative integer, CuPy array
							with specified device id are used. If negative integer
							NumPy arrays are used
		gradclip 			gradient norm threshold to clip
		out 				directory to output the result
		resume 				resume training from snap shot
		unit 				number of LSTM units in each layer
		model_filename 		model filename to serialize
		'''
		self.data = data
		self.batch_size = batch_size
		self.bproplen = bproplen
		self.epoch = epoch
		self.device = chainer.get_device(device)
		self.device.use()
		self.grad_clip = grad_clip
		self.out = out
		self.resume = resume
		self.test = test
		self.unit = unit
		self.model_filename = model_filename
		self.nvocab = self.data.nvocab if self.data.nvocab else max(self.data.train) + 1

	def _set_data(self):
		self.train_iter = ParallelSequentialIterator(self.data.train_integer, self.batch_size)
		if self.data.dev:
			self.dev_iter = ParallelSequentialIterator(self.data.dev_integer,1, repeat= False)
		if self.data.test:
			self.test_iter = ParallelSequentialIterator(self.data.test_integer,1, repeat= False)


	def _set_model(self):
		self.rnn= RNNForLM(self.nvocab,self.unit)
		self.model = L.Classifier(self.rnn)
		self.model.compute_accuracy = False # only compute perplexity
		self.model.to_device(self.device)
		self.optimizer = chainer.optimizers.SGD(lr=1.0)
		self.optimizer.setup(self.model)
		self.optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(self.grad_clip))
		#setup trainer
		self.updater = BPTTUpdater(self.train_iter,self.optimizer, self.bproplen,self.device)
		self.trainer = training.Trainer(self.updater,(self.epoch,'epoch'),out=self.out)

		self.eval_model = self.model.copy() #model with shared params and distinct states
		self.eval_rnn = self.eval_model.predictor
		self.trainer.extend(extensions.Evaluator(
			self.dev_iter, self.eval_model, device = self.device,
			#reset the RNN state at the beginning of each evaluation
			eval_hook=lambda _: self.eval_rnn.reset_state()))

		self.interval = 10 if self.test else 500
		self.trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
			trigger=(self.interval,'iteration')))
		self.trainer.extend(extensions.PrintReport(
			['epoch', 'iteration', 'perplexity', 'val_perplexity']),
			trigger=(self.interval,'iteration'))
		self.trainer.extend(extensions.ProgressBar(
			update_interval=1 if self.test else 10))
		self.trainer.extend(extensions.snapshot())
		self.trainer.extend(extensions.snapshot_object(
			self.model, 'model_iter_' + str(self.updater.iteration)))
		''' reload model
		if args.resume is not None:
			chainer.serializers.load_npz(args.resume, trainer)
		'''
		
	def train(self):
		if not hasattr(self,'train_iter'): self._set_data()
		if not hasattr(self,'trainer'):self._set_model()
		self.trainer.run()
		self.trained = True

	def eval(self, eval_data = None):
		if not hasattr(self,'trained'): self.train()
		print('test')
		self.eval_rnn.reset_state()
		self.evaluator = extensions.Evaluator(self.test_iter, self.eval_model,device=self.device)
		self.result = self.evaluator()
		print('test perplexity:', np.exp(float(result['main/loss'])))

	def save(self):
		# Serialize the final model
		chainer.serializers.save_npz(args.model, model)

def train_on_council():
	train = open(lm.lm_dir + 'council_notes_manual_transcriptions_train').read()
	dev = open(lm.lm_dir + 'manual_transcriptions_dev').read()
	test = open(lm.lm_dir + 'manual_transcriptions_test').read()
	# d = rnn.Data(filename = lm.lm_dir + 'council_notes_cleaned_labelled')
	d = Data(train = train, dev = dev, test =test)
	t = Train(data = d,device = 0,epoch = 10, model_filename='council')
	return t
	

class Data():
	def __init__(self,filename = '', text= None,train=None,dev=None,test=None,vocab = None,
		output_filename = '',input_filename = ''):
		self.filename = filename
		self.text = text
		self.train = train
		self.dev = dev
		self.test = test
		self.vocab = vocab if vocab else load_integer_vocab()
		self.nvocab = len(self.vocab) if self.vocab  else 0
		self.output_filename = output_filename
		self.save = True if output_filename else False	
		self.ok = True
		if input_filename: self._load_data()
		else: self._set_info()
		self._get_unk()
		self._count_data()

	def __repr__(self):
		m = 'data object for rnn training | status ok: ' + str(self.ok) + '\n'
		if self.ok: m += 'data available for: '
		if self.vocab:
			if self.train:m += 'train '
			if self.dev:m += 'dev '
			if self.test:m += 'test'
			if self.nunk_counted: m += '\nnumber of unks: ' + str(self.nunk)
			if self.nwords_counted: m += ' number of words: ' + str(self.nwords)
			if self.nunk_counted and self.nwords_counted: 
				m += '\nperc unks: ' + str(round(self.nunk/self.nwords,2))
		return m
			

	def _set_info(self):
		if self.filename: self.text = open(self.filename).read()
		if self.text:
			self.train, self.dev, self.test = st.split_text(self.text, 
				output_filename = self.output_filename, save = self.save)
		if not self.train and not self.dev and not self.text: self.ok = False
		if self.vocab: self._to_integer()

	def _to_integer(self):
		if self.train: self.train_integer = text2integer(self.train,self.vocab)
		if self.dev: self.dev_integer = text2integer(self.dev,self.vocab)
		if self.test: self.test_integer = text2integer(self.test,self.vocab)
		
	def _get_unk(self):
		if not self.vocab and not '<unk>' in self.vocab: 
			self.nunk_counted = False
			return
		n = self.vocab['<unk>']
		self.nunk_train = len(np.where(self.train_integer == n)[0])
		self.nunk_dev = len(np.where(self.dev_integer == n)[0])
		self.nunk_test= len(np.where(self.test_integer == n)[0])
		self.nunk = self.nunk_train + self.nunk_dev + self.nunk_test
		self.nunk_counted = True

	def _count_data(self):
		if self.vocab:
			if self.train: self.nwords_train = len(self.train_integer)
			if self.dev: self.nwords_dev= len(self.dev_integer)
			if self.test: self.nwords_test= len(self.test_integer)
			self.nwords = self.nwords_train + self.nwords_dev + self.nwords_test
			self.nwords_counted = True
		else: self.nwords_counted = False

	@property
	def unk_words(self):
		if not hasattr(self,'_unk_words') and self.text and self.vocab:
			unk_words = [word for word in self.text.replace('\n',' ').split(' ') 
				if word not in self.vocab]
			self._unk_words = Counter(unk_words).most_common()
		if hasattr(self,'_unk_words'):return self._unk_words
			
		

''' JUNK

def load_train(train_text=lm_dir + 'council_notes_train',vocab = None, unk = '<unk>',
	eos = True, sos = False):
	if not vocab: vocab = load_integer_vocab()
	nvocab = len(vocab)
	if sos: vocab['<s>'] = nvocab +1
	nvocab = len(vocab)
	if eos:vocab['</s>'] = nvocab +1
	return text2integer(open(train_text).read(),vocab,unk,eos,sos)

'''

def _make_integer_vocab(save = False,sos = False,eos = True):
	v = open(lm_dir+ 'vocab_council').read().split('\n')
	i = 0
	vocab= []
	for word in v:
		vocab.append(word + '\t' + str(i))
		i += 1
	nvocab = len(vocab)
	if sos: vocab['<s>'] = nvocab +1
	nvocab = len(vocab)
	if eos:vocab['</s>'] = nvocab +1
	if not save: return 'did not save the vocab'
	name = 'vocab_council_integer'
	with open(lm_dir+ name,'w') as fout:
		fout.write('\n'.join(vocab))
	return 'saved vocab: ' + lm_dir + name

def load_integer_vocab():
	t = open(lm_dir + 'vocab_council_integer').read().split('\n')
	t = [line.split('\t') for line in t]
	return dict([[line[0],int(line[1])] for line in t])
							
