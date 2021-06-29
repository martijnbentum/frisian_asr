import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import onnx

from texts.models import Text
from .model import Encoder
from .dataset import BaseLoad
from .utils import zcr_vad, get_timestamp
from .cluster import OptimizedAgglomerativeClustering

from utils import make_kaldi_files as mkf
from utils import frisian_speaker_classification as fsc

# from openvino.inference_engine import IECore, IENetwork

class BasePredictor(BaseLoad):
	def __init__(self, config_path, max_frame, hop):
		config = torch.load(config_path)
		self.device = torch.device("cuda:"+str(self.device) if torch.cuda.is_available() else "cpu")
		super().__init__(config.sr, config.n_mfcc)
		self.ndim = config.ndim
		self.max_frame = max_frame
		self.hop = hop
		
	@staticmethod
	def _plot_diarization(y, spans, speakers):
		c = y[0].cpu().numpy().copy()
		for (start, end), speaker in zip(spans, speakers):
			c[start:end] = speaker

		plt.figure(figsize=(15, 2))
		plt.plot(y[0], "k-")
		for idx, speaker in enumerate(set(speakers)):
			plt.fill_between(range(len(c)), -1, 1, where=(c==speaker), alpha=0.5, 
				label=f"speaker_{speaker}")
		plt.legend(loc="upper center", ncol=idx+1, bbox_to_anchor=(0.5, -0.25))
		plt.savefig('diarization.png')
		
		
class PyTorchPredictor(BasePredictor):
	def __init__(self, config_path, model_path, max_frame=40, hop=3, device = 0,
		max_cluster=10,gender=False):
		self.device = device
		super().__init__(config_path, max_frame, hop)
		self.gender = gender
		self.max_cluster = max_cluster
		
		weight = torch.load(model_path, map_location="cpu")
		self.model = Encoder(self.ndim).to(self.device)
		self.model.load_state_dict(weight)
		self.model.eval()

	def predict_wav(self,wav):
		path = wav
		y = self._load(path, mfcc=False,wav_name=path)
		nframes = y.shape[-1]
		wav_name = wav.split('/')[-1]
		texts = wav_name2texts(wav_name,verbose = False)
		# print(texts,wav_name)
		ntexts = len(texts)
		if ntexts == 1: return [0],[0],texts
		if ntexts == 0: return False,False,False
		spans,lines = get_frisian_council_spans(texts,frame_length_audio = nframes)
		# print(min([x[1]-x[0] for x in spans]), 'min span', wav)
		embed = [self._encode_segment(y, span,True) for span in spans]
		# print(len(spans),len(lines),len(embed))
		embed = torch.cat(embed).cpu().numpy()
		max_cluster = ntexts if ntexts < 10 else self.max_cluster
		speakers = OptimizedAgglomerativeClustering(max_cluster).fit_predict(embed)
		timestamp = np.array(spans) / self.sr
		return timestamp, speakers, lines

	def predict_meeting(self,meeting):
		wavs = meeting2wav_names(meeting)
		spans,lines,embed = [],[],[]
		for wav in wavs:
			path = mkf.council_wav_dir + wav
			y = self._load(path, mfcc=False,wav_name=path)
			nframes = y.shape[-1]
			texts = wav_name2texts(wav,meeting)
			sp,li = get_frisian_council_spans(texts,frame_length_audio = nframes)
			spans.extend(sp)
			lines.extend(li)
			print(min([x[1]-x[0] for x in sp]), 'min span', wav)
			emb = [self._encode_segment(y, span,True) for span in sp]
			embed.extend(emb)
		print(len(spans),len(lines),len(embed))

		embed = torch.cat(embed).cpu().numpy()
		speakers = OptimizedAgglomerativeClustering(max_cluster= self.max_cluster).fit_predict(embed)
		
		timestamp = np.array(spans) / self.sr
		return timestamp, speakers, lines

	def predict_table(self, table, plot=False):		
		y = self._load(table.audio_fn, mfcc=False,wav_name=table.audio_fn)
		if self.gender: spans,lines = get_spans(table,exlude_unknown=True)
		spans,lines= get_spans(table)
		
		embed = [self._encode_segment(y, span) for span in spans]
		embed = torch.cat(embed).cpu().numpy()
		speakers = OptimizedAgglomerativeClustering(max_cluster= self.max_cluster).fit_predict(embed)
		
		if plot:
			self._plot_diarization(y, spans, speakers)
			
		timestamp = np.array(spans) / self.sr
		return timestamp, speakers, lines
	
	def predict(self, path, plot=False):		
		y = self._load(path, mfcc=False,wav_name=path)
		activity = zcr_vad(y)
		spans = get_timestamp(activity)
		
		embed = [self._encode_segment(y, span) for span in spans]
		embed = torch.cat(embed).cpu().numpy()
		speakers = OptimizedAgglomerativeClustering(max_cluster=self.max_cluster).fit_predict(embed)
		
		if plot:
			self._plot_diarization(y, spans, speakers)
			
		timestamp = np.array(spans) / self.sr
		return timestamp, speakers
	
	def _encode_segment(self, y, span, verbose = False):
		start, end = span
		if verbose: print(start,end, start/16000, end/16000)
		mfcc = self._mfcc(y[:, start:end]).to(self.device)
		mfcc = mfcc.unfold(2, self.max_frame, self.hop).permute(2, 0, 1, 3)
		with torch.no_grad():
			embed = self.model(mfcc).mean(0, keepdims=True)
		return embed
		
	def to_onnx(self, outdir="model/openvino"):
		os.makedirs(outdir, exist_ok=True)
		mfcc = torch.rand(1, 1, self.n_mfcc, self.max_frame).to(self.device)
		onnx.export(self.model, mfcc, f"{outdir}/diarization.onnx", input_names=["input"], output_names=["output"])
		print(f"model is exported as {outdir}/diarization.onnx")	 
		

def get_spans(table, min_nframes=8000, exclude_background_comment = True, exlude_unknown=False):
	spans = []
	lines = []
	for l in table.lines:
		if l.speaker_id in ['BACKGROUND','COMMENT']: continue
		if l.speaker_id == 'UNKNOWN': continue
		start, end = [int(l.start*16000),int(l.end*16000)]
		if (end-start) < min_nframes: continue
		spans.append([start,end])
		lines.append(l)
	return spans,lines


def get_frisian_council_spans(texts, min_nframes=8000, frame_length_audio =None):
	lines,spans = [],[]
	for i,text in enumerate(texts):
		start,end = int(text.start_time*16000), int(text.end_time*16000)
		if start < 0: start = 0
		if frame_length_audio and end > frame_length_audio: end = frame_length_audio
		extra = 0
		if (end - start) < min_nframes:
			half = int((min_nframes- (end-start)) /2) +1
			start -=half
			if start < 0:
				extra = start*-1
				start = 0
			if i == len(texts) -1:
				#last text of audio file
				start -=half
			else: end += half + extra
		if (end-start) < min_nframes:
			if not i == 0:
				start -= (min_nframes - (end-start) + 1)
			else:
				end += (min_nframes - (end-start) + 1)
		if (end-start) < min_nframes: print(text,end-start)
		spans.append([start,end])
		lines.append(text)
	return spans, lines
		
	
def meeting2wav_names(meeting):
	'''extract the set of unique audio files from a meeting.
	a meeting was cut into distinct audio files'''
	return list(set([text.wav_filename for text in meeting]))


def wav_name2texts(wav_name,meeting=None, verbose = True,select = True):
	'''returns the transcripts related to the specific filename.
	without meeting the query is much slower
	'''
	if meeting:return meeting.filter(wav_filename = wav_name)
	if verbose:print('provide meeting to speed up query time')
	texts = Text.objects.filter(wav_filename = wav_name)
	if select: texts = select_texts(texts,verbose)
	return texts 


def select_texts(texts,verbose=True):
	'''filter out texts without duration or any transcription'''
	output,rejected = [],[]
	for text in texts:
		if text.transcription.duration < 0.1 or not text.transcription.line_with_tags:
			rejected.append(text)
		else: output.append(text)
	print('rejected:',len(rejected),'texts','Accepted',len(output),'texts')
	return output

