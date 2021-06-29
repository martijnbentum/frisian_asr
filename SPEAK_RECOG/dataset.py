import os
import wave
import torch
import numpy as np

from glob import glob
from tqdm.auto import tqdm

import torchaudio
from torchaudio.transforms import MFCC, Resample
from torch.utils.data import Dataset, DataLoader
from . import speakers as spk


class BaseLoad:
	def __init__(self, sr, n_mfcc=40):
		self.sr = sr
		self.n_mfcc = n_mfcc
		self._mfcc = MFCC(sr, n_mfcc=40, log_mels=True)
		
	def _load(self, line, mfcc=True, wav_name = None):
		if wav_name:
			waveform, ori_sr = torchaudio.load(wav_name)
			waveform = waveform.mean(0, keepdims=True)
		else:
			try:
				waveform, ori_sr = torchaudio.load(line.audio_fn,
					frame_offset=line.start_frame,num_frames=line.nframes)
				waveform = waveform.mean(0, keepdims=True)
			except RuntimeError:
				raise Exception(f"Error loading {line.audio_fn}")
		_resample = Resample(ori_sr, self.sr)
		audio = _resample(waveform)
		# print('audio',audio.shape)

		if mfcc:
			audio = self._mfcc(audio)
		return audio



class VCTKTripletDataset(Dataset, BaseLoad):
	def __init__(self,speakers =None, tables = None, n_data=9000, sr=16000, min_dur=2,gender=False):
		if not speakers: 
			self.speakers = spk.make_speakers(tables = tables,min_duration = min_dur, min_sr=sr)
		else: self.speakers = speakers

		BaseLoad.__init__(self, sr)
		
		self.min_dur = min_dur
		self.sr = sr
		# self.speakers = list(sorted(os.listdir(wav_path)))
		if gender:
			male = [x for x in self.speakers if x.sex=='male'][0]
			female = [x for x in self.speakers if x.sex=='female'][0]
			male.lines = []
			female.lines = []
			self.speaker_to_idx = {'male':male,'female':female}
			for x in self.speakers:
				self.speaker_to_idx[x.sex].lines.extend(x.lines)
				
		self.speaker_to_idx = {v: k for k, v in enumerate(self.speakers)}
		
		#self.data is a list of data triplets created by repeated calss to _random_sample
		#each item in this lists contains
		#wav filename speaker 1 (anchor) a
		#wav filename speaker 1 (positive) p
		#wav filename speaker 2 (negative) n 
		#ya integer id of speaker1 
		#yp id of speaker 1 (same as ya?)
		#yn id of speaker2
		self.data = [self._random_sample() for _ in tqdm(range(n_data), desc="Sample Data")]
		# self._remove_short_audio()
		
	def __getitem__(self, i):
		a, p, n, ya, yp, yn = self.data[i]
		mfcc_a = self._load(a)
		mfcc_p = self._load(p)
		mfcc_n = self._load(n)
		ya = self.speaker_to_idx[ya]
		yp = self.speaker_to_idx[yp]
		yn = self.speaker_to_idx[yn]
		return mfcc_a, mfcc_p, mfcc_n, ya, yp, yn
		
	def __len__(self):
		return len(self.data)
	
	def _random_sample(self):
		'''returns 3 files from 2 speaker
		anchor		audio from speaker 1
		positive	audio from speaker 1
		negative audio from speaker 2
		'''
		speaker_a, speaker_n = np.random.choice(self.speakers, 2, replace=False)
		# a, p = np.random.choice(glob(f"{self.wav_path}/{speaker_a}/*.wav"), 2, replace=False)
		a, p = speaker_a.sample(2)
		n = speaker_n.sample(1)[0]
		return a, p, n, speaker_a, speaker_a, speaker_n
	
	def _remove_short_audio(self):
		#obsolete
		def _dur(fname):
			with wave.open(fname, 'r') as f:
				frames = f.getnframes()
				rate = f.getframerate()
				duration = frames / float(rate)
			return duration
		
		new_data = [data for data in self.data if min(_dur(data[0]), _dur(data[0]), _dur(data[0])) >= self.min_dur]
		n_excluded = len(self.data) - len(new_data)
		
		if n_excluded > 0:
			print(f"Excluding {n_excluded} triplet containing audio shorter than {self.min_dur}s")
		self.data = new_data
	

class VCTKTripletDataloader(DataLoader):
	def __init__(self, dataset, batch_size, shuffle=True, num_workers=9):
		super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate, num_workers=num_workers)
		
	def collate(self, batch):
		a, p, n, ya, yp, yn = zip(*batch)
		X = a + p + n
		y = ya + yp + yn
		
		min_frame = min([i.shape[-1] for i in X])
		X = [i[:, :, :min_frame] for i in X]
		return torch.cat(X).unsqueeze(1), torch.LongTensor(y)
	
	
class VCTKSpeakerDataset(Dataset, BaseLoad):
	def __init__(self, wav_path, txt_path, n_speaker=20, n_each_speaker=10, sr=16000, min_dur=2):
		self.wav_path = wav_path
		self.txt_path = txt_path
		BaseLoad.__init__(self, sr)
		
		self.min_dur = min_dur
		self.speakers = list(sorted(os.listdir(wav_path)))
		self.speaker_to_idx = {v: k for k, v in enumerate(self.speakers)}
		
		random_speakers = np.random.choice(self.speakers, n_speaker, replace=False)
		self.data = [(path, speaker) for speaker in tqdm(random_speakers, desc="Sample Data")
					 for path in np.random.choice(glob(f"{self.wav_path}/{speaker}/*.wav"), n_each_speaker, replace=False)]
		self._remove_short_audio()
		
	def __getitem__(self, i):
		X, y = self.data[i]
		mfcc = self._load(X)
		y = self.speaker_to_idx[y]
		return mfcc, y
		
	def __len__(self):
		return len(self.data)
	
	def _remove_short_audio(self):
		def _dur(fname):
			with wave.open(fname, 'r') as f:
				frames = f.getnframes()
				rate = f.getframerate()
				duration = frames / float(rate)
			return duration
		
		new_data = [data for data in self.data if _dur(data[0]) >= self.min_dur]
		n_excluded = len(self.data) - len(new_data)
		
		if n_excluded > 0:
			print(f"Excluding {n_excluded} triplet containing audio shorter than {self.min_dur}s")
		self.data = new_data	

	
class VCTKSpeakerDataloader(DataLoader):
	def __init__(self, dataset, batch_size, shuffle=True, num_workers=3):
		super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate, num_workers=num_workers)
		
	def collate(self, batch):
		X, y = zip(*batch)
		
		min_frame = min([i.shape[-1] for i in X])
		X = [i[:, :, :min_frame] for i in X]
		return torch.cat(X).unsqueeze(1), torch.LongTensor(y)
