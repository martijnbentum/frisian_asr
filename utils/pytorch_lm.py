#Pytorch code taken from wsj/s5/steps/pytorch/train.py
#default parameters from wsj/s5/local/pytorchnn/run_nnlm.sh see below for github link
#https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/local/pytorchnn/run_nnlm.sh

#the kaldi script is based on a pytoch example script:
#https://github.com/pytorch/examples/tree/master/word_language_model


'''
data format
create a vocabulary that maps each word to integer
create a train, dev and test set
map each word to the integer according to the vocabulary

load a tensor with the integers for train (and a tensor for dev, etc)
devide the tensor in n batches (e.g. 32) each batch contains 1/32 of the tensor length
(see batchify)
for training get a seq_len number of words (i.e. integers) from each batch in data and 
the next word for each word in each batch in target (see get_batch)
data has shape [seq_len,batch_size] and target has shape [seq_len * batch_size]
the first item in target equals the second item in the first batch in data:
data[1,0] == target[0]
data[1,1] == target[1]
data[2,0] == target[32] (with a batch_size of 32)
'''

import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

from . import pytorch_data
from . import pytorch_model as model
save_dir = '../LM/rnn/pytorch/'
oov = '<unk>'
eval_batch_size = 20


class Train:
	def __init__(self,model_type = 'Transformer',embedding_dim=768,hidden_dim=768,
		nlayers=8,nhead = 8,learning_rate = 0.1,seq_len=100,dropout=0.2,clip=1.0,batch_size = 32,
		epoch = 64, model_filename = 'default', tied = True, cuda = True, oov = '<unk>',
		log_interval = 100, optimizer_name = 'SGD', random_seed = 1111, corpus = None,
		device_number = 0):
		'''Train a pytorch LM
		model_type  		can be rnn_tanh, rnn_relu, lstm, gru or Transformer
							default transformer taken from wsj kaldi project (as other defaults)
							(pytorch default LSTM)
		embedding_dim 		size of word embeddings (pytorch default 200)
		hidden_dim 			number of hidden units per layer (pytorch default 200)
		nlayers 			number layers (pytorch default 2)
		nhead 				the number of heads in the encoder/decoder of the transformer model
		learning_rate 		initial learning rate
		seq_len 			sequence length limit (pytorch default = 35) (?number of words?)
		clip 				gradient clipping (pytorch default = 0.25)
		tied 				tie the word embeddings and softmax weights
		optimizer_name		type of optimizer (! not adam?) DOES NOT DO ANYTHING YET
		log_interval 		report interval
		model_filename 		filename of saved model
		epoch 				upper epoch limit (pytorch default = 20)
		random_seed 		seed number for reproducability
		corpus 				object to train dev and test the model with
		'''
	
		self.model_type = model_type
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.nlayers = nlayers
		self.nhead = nhead
		self.learning_rate = learning_rate
		self.seq_len = seq_len
		self.dropout = dropout
		self.clip = clip
		self.optimizer_name = optimizer_name # does not do anything right now
		self.batch_size = batch_size
		self.epoch = epoch
		self.model_filename = model_filename
		self.log_interval = log_interval
		self.tied = tied
		self.cuda = cuda
		self.oov = oov
		self.random_seed = random_seed
		self.device_number = device_number
		self._set_corpus(corpus)

		random.seed(self.random_seed)
		torch.manual_seed(self.random_seed)
		# self.device = torch.device('cuda' if self.cuda else 'cpu')
		self.device = device_number

		if self.model_type == 'Transformer':self._set_transformer()
		else: self._set_rnn()

		self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9,	
			weight_decay=1e-5)
		self.criterion = nn.CrossEntropyLoss()

		self.current_loss = []
		self.dev_loss = []
		self.elapsed_time = []
		self.epoch_counter = 0

		self.nbatches = len(self.train_data)//self.seq_len 


	def _set_corpus(self, corpus):
		if not corpus: self.corpus = pytorch_data.Corpus()
		else: self.corpus = corpus
		self.train_data = batchify(self.corpus.train, self.batch_size,device= self.device_number)
		self.dev_data = batchify(self.corpus.valid, eval_batch_size, device=self.device_number)
		self.test_data = batchify(self.corpus.test, eval_batch_size, device=self.device_number)
		self.nvocab = len(self.corpus.dictionary)

	def _set_transformer(self):
		self.model = model.TransformerModel(self.nvocab, self.embedding_dim, self.nhead,
			self.hidden_dim, self.nlayers, self.dropout, 'gelu', self.tied).to(self.device)

	def _set_rnn(self):
		self.model = model.RNNModel(self.model_type, self.nvocab, self.embedding_dim,
			self.hidden_dim, self.nlayers, self.dropout, self.tied).to(self.device)


	def _train_epoch(self):
		'''train the model on all data in the training data set once.'''
		self.model.train()
		self.total_loss = 0.
		self.start_time = time.time()
		if self.model_type != 'Transformer':
			self.hidden = model.init_hidden(self.batch_size)
		for batch, i in enumerate(range(0, self.train_data.size(0), self.seq_len)):
			data, targets = get_batch(self.train_data, i)
			self.optimizer.zero_grad()
			if self.model_type == 'Transformer': output = self.model(data)
			else:
				# Starting each batch, the hidden state is detached from how it was
				# previously produced. Otherwise, the model would try
				# backpropagating all the way to start of the dataset.
				hidden = repackage_hidden(hidden)
				output, hidden = model(data, hidden)
			self.loss = self.criterion(output.view(-1, self.nvocab), targets)
			self.loss.backward()
		
			# 'clip_grad_norm' helps prevent the exploding gradient problem.
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
			self.optimizer.step()

			self.total_loss += self.loss.item()
			if batch % self.log_interval == 0 and batch > 0:
				self.current_loss.append( self.total_loss / self.log_interval )
				self.elapsed_time.append( time.time() - self.start_time )
				self._message_train(batch)
			self.total_loss = 0.
			self.start_time = time.time()


	def train(self):
		self.counter = 0
		self.best_dev_loss = None
		print('start training')
		for epoch in range(1,self.epoch + 1):
			self.epoch_counter += 1
			print('starting epoch:',epoch)
			self.epoch_start_time = time.time()
			self._train_epoch()
			self.dev_loss.append(self.evaluate(self.dev_data))
			self._message_epoch()
			#if dev_loss is best loss or (none save) save current model
			if not self.best_dev_loss or self.dev_loss[-1] < self.best_dev_loss:
				with open(save_dir + self.model_filename, 'wb') as f:
					torch.save(self.model.state_dict(), f)
					self.best_dev_loss = self.dev_loss[-1]
			else:
			# if dev loss is not an improvement lower learning rate
				self.learning_rate /= 2.
				self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate,
					momentum = 0.9, weight_decay = 1e-5)
				self.counter += 1
			if self.counter == 8: 
				print('no improvement to loss for the eight time, stopping early')
				print('finished',epoch,'of training')
				break
		self.test()
				

	def load_model(self,filename= None):
		if not filename: filename = save_dir + self.model_filename
		if '/' not in filename: filename = save_dir + self.model_filename
		print('loading model with name:',filename)
		with open(filename, 'rb') as f:
			self.model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))
			if self.model_type in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
				self.model.rnn.flatten_parameters()


	def test(self, load_saved_model = True):
		if load_saved_model: self.load_model()
		self.test_loss = self.evaluate(self.test_data)
		print('test results, test loss:',self.test_loss,'test ppl:',math.exp(self.test_loss))


	def evaluate(self,source):
		# Turn on evaluation mode which disables dropout.
		self.model.eval()
		self.eval_loss = 0.
		if self.model_type != 'Transformer':
			self.hidden = model.init_hidden(eval_batch_size)
		# Speed up evaluation with torch.no_grad()
		with torch.no_grad():
			for i in range(0,source.size(0) -1, self.seq_len):
				data, targets = get_batch(source,i)
				if self.model_type == 'Transformer': output = self.model(data)
				else:
					hidden = repackage_hidden(hidden)
					output, hidden = model(data, hidden)
				loss = self.criterion(output.view(-1,self.nvocab), targets)
				self.eval_loss += len(data) * loss.item()
		return self.eval_loss / (len(source) -1 )
				

	def _message_train(self, batch):
		m = 'training\n'
		m += 'epoch'.ljust(20) + str(self.epoch_counter) +'\n'
		m += 'batch'.ljust(20) + str(batch) + '/' +  str(self.nbatches) + '\n'
		m += 'learning rate'.ljust(20) + str(self.learning_rate) + '\n'
		m += 'ms/batch'.ljust(20) + str(self.elapsed_time[-1] *1000 / self.log_interval) + '\n'
		m += 'loss'.ljust(20) + str(self.current_loss[-1]) + '\n'
		m += 'ppl'.ljust(20) + str(math.exp(self.current_loss[-1])) + '\n'
		print(m)

	def _message_epoch(self):
		m = 'finished epoch: '+ str(self.epoch) +'\n'
		m += 'learning rate'.ljust(20) + str(self.learning_rate) + '\n'
		m += 'loss'.ljust(20) + str(self.dev_loss[-1]) +'\n'
		m += 'ppl'.ljust(20) + str(math.exp(self.dev_loss[-1])) +'\n'
		print(m)

		
		
				
				
		
			
			
		

def batchify(data,batch_size, random_start_idx=False, device = 1):
	# Work out how cleanly we can divide the dataset into batch_size parts.
	# number of words in each data
	nbatch = data.size(0) // batch_size
	# Shuffle data
	if random_start_idx: start_idx = random.randint(0, data.size(0) % batch_size - 1)
	else: start_idx = 0
	# Trim off any extra elements that wouldn't cleanly fit (remainders).
	# 1 - (batch_size-1) words gets chopped of at the end of the data
	data = data.narrow(0, start_idx, nbatch * batch_size)
	# Evenly divide the data across the batch_size batches
	data = data.view(batch_size, -1).t().contiguous()
	#data is a torch tensor of shape [nbatch, batch_size] (whereby nbatch is the number of
	#words in a batch and batch_size is the number of batches
	return data.to(device)
	
# Divide the source data into chunks of length seq_len. ?number of words in a sequence?
def get_batch(source, i, seq_len = 100):
	'''gets seq_len number of words of each batch in source in data and the next word
	of each word in target.
	source 		torch tensor object of [x, batch_size], whereby x 1/batch_size of the
				original text length
	i 			the index in each batch, extract seq_len words starting from this index
	seq_len 	n words to extract from each batch
	'''

	seq_len = min(seq_len, len(source) - 1 - i)
	data = source[i: i + seq_len]
	target = source[i + 1: i + 1 + seq_len].view(-1)
	return data, target
		
		
	
def repackage_hidden(h):
	'''Wraps hidden states in new Tensors, to detach them from their history.'''
	if isinstance(h, torch.Tensor): 
		return h.detach()
	return type(repackage_hidden(v) for v in h)
		

