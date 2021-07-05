import glob
from utils import make_kaldi_files as mkf
from texts.models import Text 
from . import prefix_meeting_names as pmn
pmn = pmn.prefix_meeting_names
from SPEAK_RECOG import predict
import tqdm



def make_pmn_dict():
	'''create a dictionary of all frisian council meetings mapping those to the 
	related transcriptions.
	'''
	t = Text.objects.filter(source__name = 'frisian council transcripts')
	d = {}
	for name in pmn:
		d[name] = t.filter(source__name = 'frisian council transcripts',
			file_id__icontains = name)
	return d


def meeting2speakerids(meeting,p, meeting_name, save = False, subdivide= 300): 
	if subdivide: timestamps,speakers,texts,subdivide_list=p.predict_meeting(meeting,subdivide)
	else:timestamps, speakers, texts= p.predict_meeting(meeting)
	assert len(speakers) == len(texts)
	speaker2text, text2speaker = {},{}
	for i,text in enumerate(texts):
		speaker_num = str(speakers[i])
		speaker_num = '0' + speaker_num if len(speaker_num) == 1 else speaker_num
		subdivide_name = '_' + subdivide_list[i] if subdivide_list else ''
		speaker_id = meeting_name + subdivide_name + '_speaker_'+ speaker_num
		text2speaker[text.pk] = speaker_id
		if speaker_id not in speaker2text.keys(): speaker2text[speaker_id] = [] 
		speaker2text[speaker_id].append(text.pk)
		text.speaker_id = speaker_id
		if save:text.save()
	return text2speaker, speaker2text, texts
			
		


def classify_speakers(model_dir = "cgn_speaker_model_v3/", device=1, save = False, 
	start_index = None,subdivide=300):
	p = predict.PyTorchPredictor(model_dir + "configs.pth", 
		model_dir + "weights_best.pth",device=device)
	pmnd = make_pmn_dict()
	d = {}
	for k in pmnd.keys():
		print(k)
		if start_index:
			if list(pmnd.keys()).index(k) < start_index:
				print('skipping:',k)
				continue
		meeting = pmnd[k]
		text2speaker,speaker2text,texts = meeting2speakerids(meeting,p,k,save, subdivide)
		d[k] = [text2speaker, speaker2text, texts]
	return d


def wav2speakerids(wav_filename,p,save=False):
	timestamps, speakers, texts = p.predict_wav(wav_filename)
	if not texts: return False, False, False # no text associated with wav file
	assert len(speakers) == len(texts)
	speaker2text, text2speaker = {},{}
	for i,text in enumerate(texts):
		speaker_num = str(speakers[i])
		speaker_num = '0' + speaker_num if len(speaker_num) == 1 else speaker_num
		speaker_id = text.wav_filename.split('.')[0] + '_speaker_'+ speaker_num
		text2speaker[text.pk] = speaker_id
		if speaker_id not in speaker2text.keys(): speaker2text[speaker_id] = [] 
		speaker2text[speaker_id].append(text.pk)
		text.speaker_id = speaker_id
		if save:text.save()
	return text2speaker, speaker2text, texts


def classify_speakers_wav(model_dir = "cgn_speaker_model_v3/", device=1, 
	save = False, start_index = None):
	p = predict.PyTorchPredictor(model_dir + "configs.pth", 
		model_dir + "weights_best.pth",device=device)
	pmnd = make_pmn_dict()
	d = {}
	fn = glob.glob(mkf.council_wav_dir +'*.wav')
	for i,f in enumerate(fn):
		print(f,i,len(fn))
		if start_index:
			if fn.index(f) < start_index:
				print('skipping:',f)
				continue
		text2speaker,speaker2text,texts = wav2speakerids(f,p,save)
		if not text2speaker: 
			print('no texts did nothing')
			continue
		d[f] = [text2speaker, speaker2text, texts]
	return d
	

def texts_per_wav():
	fn = glob.glob(mkf.council_wav_dir +'*.wav')
	d = {}
	count = []
	for f in fn:
		wav_name = f.split('/')[-1]
		texts = predict.wav_name2texts(wav_name,meeting=None, verbose = True,select = True)
		d[wav_name] = texts
		count.append(len(texts))
	return d, count




	 
