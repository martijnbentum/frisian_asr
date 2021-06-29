import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report as cr
import pickle


class OptimizedAgglomerativeClustering:
	def __init__(self, max_cluster=10):
		self.kmax = max_cluster
		
	def fit_predict(self, X):
		if self.kmax == 2: best_k = 2
		else:best_k = self._find_best_k(X)
		print('best_k:',best_k,self.kmax)
		membership = self._fit(X, best_k)
		return membership

	def _fit(self, X, n_cluster):
		return AgglomerativeClustering(n_cluster).fit_predict(X)
		
	def _find_best_k(self, X):
		if self.kmax == 2:return 2 
		else:cluster_range = range(2, min(len(X), self.kmax))
		score = [silhouette_score(X, self._fit(X, k)) for k in cluster_range]
		best_k = cluster_range[np.argmax(score)]
		return best_k


def find_mapping(table_lines,speaker_integers, gender = False, sid2gender = None):
	assert len(table_lines) == len(speaker_integers)
	speaker_id_dict = {}
	for i,l in enumerate(table_lines):
		attr = getattr(l,'speaker_id')
		if gender and sid2gender: attr = sid2gender[attr]
		if attr  not in speaker_id_dict.keys():
			speaker_id_dict[attr] = []
		speaker_id_dict[attr].append(speaker_integers[i])
	for k in speaker_id_dict.keys():
		speaker_id_dict[k] = np.bincount(speaker_id_dict[k]).argmax()
	return speaker_id_dict
		

def make_hypothesis_and_ground_truth(table_lines, speaker_integers, gender = False,sid2gender = None):
	if sid2gender == None: sid2gender = load_speaker2gender()
	speaker_id_dict = find_mapping(table_lines, speaker_integers,gender, sid2gender)
	print(speaker_id_dict,list(set(speaker_integers)))
	if gender:
		if 'female' not in speaker_id_dict.keys():
			speaker_id_dict['female'] = 1 if  speaker_id_dict['male'] == 0 else 0
		elif 'male' not in speaker_id_dict.keys():
			speaker_id_dict['male'] = 1 if  speaker_id_dict['female'] == 0 else 0
		if speaker_id_dict['female'] == speaker_id_dict['male']:
			speaker_id_dict['female'] = 0 if speaker_id_dict['male'] == 1 else 1
	int2id = dict([[speaker_id_dict[k],k] for k in speaker_id_dict.keys()])
	for i in list(set(speaker_integers)):
		if i not in int2id.keys():
			int2id[i] = 'unk'
	speaker_hypothesis = [int2id[i] for i in speaker_integers if i in int2id.keys()]
	if gender: ground_truth = [sid2gender[l.speaker_id] for l in table_lines]
	else:ground_truth = [l.speaker_id for l in table_lines]
	sh_int = speaker_integers
	gt_int = [speaker_id_dict[speaker_id] for speaker_id in ground_truth]
	return speaker_hypothesis,ground_truth,sh_int,gt_int

def classification_report(table_lines,speaker_integers):
	sph,spt = make_hypothesis_and_ground_truth(table_lines, speaker_integers)
	print(spt,sph)

def load_speaker2gender():
	fin = open('speakers2gender','rb')
	return pickle.load(fin)
