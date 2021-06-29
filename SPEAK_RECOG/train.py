import torch
from torch import nn, optim
from jcopdl.callback import Callback, set_config
from . import files, dataset
from . import speakers as spk
from .model import Encoder
from jcopdl.optim import RangerLARS
from tqdm.auto import tqdm
import pickle
import progressbar as pb

def load_pickled_speakers():
	fin = open('speakers.pickle','rb')
	return pickle.load(fin)


class Trainer():
	def __init__(self,speakers = None,tables=None, config = None, min_dur = 2, 
		sr = 16000,n_data=9000, batch_size = 32, device = 0,outdir='default',gender = False):
		self.gender = gender
		self.outdir = outdir
		self.n_data = n_data
		self.batch_size = batch_size
		self.sr = sr
		self.min_dur = min_dur
		self._load_speakers(tables, speakers)
		self._set_train()
		self.test_set = dataset.VCTKTripletDataset(self.speakers,n_data=n_data,
			sr=sr,min_dur=min_dur,gender = self.gender)
		self.testloader = dataset.VCTKTripletDataloader(self.test_set,self.batch_size)
		self.device = torch.device("cuda:"+str(device) if torch.cuda.is_available() else "cpu")
		if not config: self._set_config()
		else:self.config = config
		self._prepare_training()

	def _load_speakers(self, tables, speakers):
		if not tables and not speakers: self.tables = files.make_tables()
		else: self.tables = tables
		if not speakers: self.speakers = spk.make_speakers(tables,self.min_dur,self.sr)
		else:self.speakers = speakers

	def _set_config(self):
		self.config = set_config({
			"ndim":256,
			"margin":1,
			"sr":self.sr,
			"n_mfcc":self.train_set.n_mfcc,
			"min_dur":self.min_dur
		})

	def _set_train(self):
		if hasattr(self,'train_set'):
			delattr(self,'train_set')
			delattr(self,'trainloader')
		self.train_set = dataset.VCTKTripletDataset(self.speakers,n_data=self.n_data, 
			sr=self.sr,min_dur=self.min_dur, gender = self.gender)
		self.trainloader = dataset.VCTKTripletDataloader(self.train_set,self.batch_size)

	def _prepare_training(self):
		self.model = Encoder(ndim=self.config.ndim, triplet=True).to(self.device)
		self.criterion = nn.TripletMarginLoss(self.config.margin)
		self.callback = Callback(self.model, self.config, outdir=self.outdir, 
			early_stop_patience=15)
		self.optimizer = RangerLARS(self.model.parameters(), lr=0.001)


	def train(self):

		while True:
			print('epoch:',self.callback.ckpt.epoch)
			if self.callback.ckpt.epoch % 15 == 0:
				self._set_train()
			self.model.train()
			cost,i = 0,0
			bar = pb.ProgressBar()
			bar(range(len(self.trainloader)))
			print('training')
			for images, labels in self.trainloader:
				bar.update(i)
				i += 1
				images = images.to(self.device)
				output = self.model(images)
				loss = self.criterion(output[0],output[1],output[2])
				loss.backward()

				self.optimizer.step()
				self.optimizer.zero_grad()

				cost += loss.item()*images.shape[0]
			train_cost = cost/len(self.train_set)

			with torch.no_grad():
				self.model.eval()
				cost,i = 0,0
				bar = pb.ProgressBar()
				bar(range(len(self.trainloader)))
				print('test')
				for images, labels in self.testloader:
					bar.update(i)
					i += 1
					images = images.to(self.device)
					output = self.model(images)
					loss = self.criterion(output[0],output[1],output[2])
					cost += loss.item()*images.shape[0]
				test_cost = cost/len(self.test_set)

			# logging
			self.callback.log(train_cost,test_cost)

			# checkpoint
			self.callback.save_checkpoint()

			# runtime plot
			self.callback.cost_runtime_plotting()

			# early stopping
			if self.callback.early_stopping(self.model, monitor="test_cost"):
				self.callback.plot_cost()
				break
					 
				
		

	
'''
while True:
	if callback.ckpt.epoch % 15 == 0:
		train_set = VCTKTripletDataset("vctk_dataset/wav48/", "vctk_dataset/txt/", n_data=3000)
		trainloader = VCTKTripletDataloader(train_set, batch_size=bs)
	
	model.train()
	cost = 0
	for images, labels in tqdm(trainloader, desc="Train"):
		images = images.to(device)
		
		output = model(images)
		loss = criterion(output[0], output[1], output[2])
		loss.backward()

		optimizer.step()
		optimizer.zero_grad()
		
		cost += loss.item()*images.shape[0]
	train_cost = cost/len(train_set)
	
	with torch.no_grad():
		model.eval()
		cost = 0
		for images, labels in tqdm(testloader, desc="Test"):
			images = images.to(device)
		
			output = model(images)
			loss = criterion(output[0], output[1], output[2])
			
			cost += loss.item()*images.shape[0]
		test_cost = cost/len(test_set)

	# Logging
	callback.log(train_cost, test_cost)

	# Checkpoint
	callback.save_checkpoint()
		
	# Runtime Plotting
	callback.cost_runtime_plotting()
	
	# Early Stopping
	if callback.early_stopping(model, monitor="test_cost"):
		callback.plot_cost()
		break 
'''
