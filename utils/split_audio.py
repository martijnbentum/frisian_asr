'''
split an audio file into non silence chunks
code assumes mono sound
'''
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse

def load_audio(filename, sample_rate=None):
	np_array, sample_rate = librosa.load(filename,sr=sample_rate)
	return np_array, sample_rate

def make_ref(x):
	return np.std(x) * 5

def duration2nframes(duration,sample_rate):
	return int(sample_rate * duration)

def split_audio(filename, min_duration = .2, top_db = 30, ref = make_ref):
	'''
	returns an array of arrays with start end frames for non silent chuncks.
	min_duration  		the minimal length of silence
	top_db 				the db below the ref to be considered silence
						higher values will accept more material as non silent
	ref 				the reference value to subtract top_db from for silence 
						threshold
	'''
	x, sr = load_audio(filename)
	frame_length = duration2nframes(min_duration,sr)
	o=librosa.effects.split(x, frame_length=frame_length, top_db=top_db, ref=ref)
	return o

def spectogram(filename, boundaries = None, add_boundaries = True, start = None, 
	end = None):
	'''plot spectogram of audio marking boundaries between silence and 
	non silence. 
	boundaries can be provided or if non provided but add boundaries == true 
	they will be computed
	start and end can be used to select a section of audio, should be in seconds
	'''
	plt.figure()
	x, sr = load_audio(filename)
	if start != None: x = x[duration2nframes(start):]
	if end != None: x = x[:duration2nframes(end)]
	if add_boundaries and not boundaries: boundaries = split_audio(filename)
	plt.specgram(x, Fs=sr)
	if add_boundaries: 
		boundaries = boundaries.flatten()
		[plt.vlines(x,0,8000,colors = 'b') for x in boundaries]

def plot(filename, boundaries = None, add_boundaries = True, start = None, 
	end = None):
	'''plot waveform of audio marking non silent audio with color
	boundaries can be provided or if non provided but add boundaries == true 
	they will be computed
	start and end can be used to select a section of audio, should be in seconds
	'''
	plt.figure()
	y, sr = load_audio(filename)
	if start != None: y = y[duration2nframes(start):]
	if end != None: y = y[:duration2nframes(end)]
	if add_boundaries and not boundaries: boundaries = split_audio(filename)
	x = np.array(list(range(y.shape[0]))) / sr
	plt.plot(x,y)
	if add_boundaries: 
		[plt.axvspan(x[0]/sr ,x[1]/sr ,facecolor='b',alpha=0.2) for x in boundaries]

def split2silence_frames(split, audio):
	nframes = audio.shape[0]
	output = np.zeros(split.shape, dtype= split.dtype)
	for i, x in enumerate(split):
		start, end = x
		if i == 0: before = start
		else: 
			prev_start, prev_end = split[i-1]
			before = start - prev_end

		if i < split.shape[0] -1:
			next_start, next_end = split[i+1]
			after = next_start - end
		else: after = nframes - end
		output[i] = np.array([before,after])
	return output

def split2silence_duration(split, audio, sample_rate):
	o = split2silence_frames(split,audio)
	return o / sample_rate

def make_non_silent_chunks(audio_filename, max_extra_silence = 1, split = None,
	minimum_silence_duration = 0.2):
	'''creates an np array of start,end points in seconds of non silent audio.
	max_extra_silence 		the maximum extra silence before or after a chunk
	split 					optional split (output from split_audio) for non default 
							splits
	'''
	audio, sample_rate = load_audio(audio_filename)
	if not split: 
		split= split_audio(filename = audio_filename,
			min_duration = minimum_silence_duration)
	split_duration = split / sample_rate
	silence_duration = split2silence_duration(split, audio, sample_rate)
	output = np.zeros(split_duration.shape ,dtype = split_duration.dtype)
	for i,x in enumerate(split_duration):
		start, end = x
		silence_before, silence_after = silence_duration[i]/2
		if silence_before > max_extra_silence: silence_before = 1.
		if silence_after > max_extra_silence: silence_after = 1.
		output[i] = np.array([round(start - silence_before,2), round(end + silence_after,2)])
	return output

def make_silent_chunks(audio_filename, split = None):
	'''creates an np array of start,end point in secods of silent audio.
	split 		optional split (output from split_audio) for non default splits
	'''
	audio, sample_rate = load_audio(audio_filename)
	if not split: split = split_audio(filename = audio_filename)
	split_duration = split / sample_rate
	output = np.zeros(split_duration.shape)
	for i, x in enumerate(split_duration):
		start, end = x
		if i == 0: 
			output[i] = np.array([0, start])
	return output

def make_single_chunk(audio_filename):
	audio, sample_rate= load_audio(audio_filename)
	chunk = np.zeros([1,2])
	chunk[0] = np.array([0.,audio.shape[0] / sample_rate])
	return chunk

def make_speaker_names(nspeakers):
	n = len(str(nspeakers))
	output = []
	for i in range(nspeakers):
		name = 'speaker_' + '0' * (n - len(str(i+1))) + str(i+1)
		output.append(name)
	return output

def make_kaldi_recources(audio_filename, chunks= None, speaker_names = None,
	make_chunks= make_non_silent_chunks, goal_dir = '', force_save = False,
	minimum_silence_duration= 0.2):
	'''create kaldi files for decoding
	audio_filename 			filename that needs to be decoded
	chunks 					start end point in seconds for non silent chunks in audio
							if false the whole audio will be used as one chunk
	'''
	if chunks == False: chunks = make_single_chunk(audio_filename)
	elif chunks == None: 
		chunks = make_chunks(audio_filename,
			minimum_silence_duration = minimum_silence_duration)
	if not speaker_names: speaker_names = make_speaker_names(len(chunks))
	segments = make_segments(audio_filename,chunks,speaker_names)
	wavscp = make_wavscp(audio_filename)
	utt2spk = make_utt2spk(audio_filename,speaker_names)
	save(segments,'segments',goal_dir,force_save)
	save(wavscp,'wav.scp',goal_dir,force_save)
	save(utt2spk,'utt2spk',goal_dir,force_save)
	print('saved segments wav.scp and utt2spk to ',goal_dir)

def make_segments(audio_filename, chunks, speaker_names):
	output = []
	for chunk, name in zip(chunks, speaker_names):
		audio_filename = audio_filename.split('/')[-1]
		start, end = chunk
		line = name + '-' + '.'.join(audio_filename.split('.')[:-1]) 
		line += ' ' + audio_filename
		line += ' ' + str(start) + ' ' + str(end)
		output.append(line)
	return output

def make_wavscp(audio_filename):
	return audio_filename.split('/')[-1] + ' ' + audio_filename

def make_utt2spk(audio_filename,speaker_names):
	output = []
	for name in speaker_names:
		audio_filename = audio_filename.split('/')[-1]
		line = name + '-' + '.'.join(audio_filename.split('.')[:-1]) + ' ' + name
		output.append(line)
	return output
	
	
def save(t, name, directory, force_save = False):
	if type(t) == list: t = '\n'.join(t)
	filename = directory + name
	if os.path.isfile(filename) and not force_save:
		print(filename,'already exists and no force save, doing nothing')
		return
	print('saving:',filename)
	with open(filename,'w') as fout:
		fout.write(t)
	

		
	
if __name__ == "__main__":
	p= argparse.ArgumentParser(description="create kaldi resources for decoding")
	p.add_argument('fn',metavar="audio filename",type=str,
		help="audio file to decode")
	p.add_argument('-d',metavar="goal dir",type=str,
		help="directory to story kaldi resource files", required = False)
	p.add_argument('--force',action="store_true",
		help="whether to overwrite excisting resource files", required = False)
	p.add_argument('--no_split',action="store_true",
		help="whether to split the audio file on silence", required = False)
	p.add_argument('-minimum_silence_duration',type=float,
		help="sets minimum duration audio chunks, default is 0.2 seconds", 
		required = False)
	args = p.parse_args()
	if not args.fn or not os.path.isfile(args.fn):
		print('please provide a filename to an audio file:',args.fn,'does not exist')
		sys.exit()
	if args.d != None and not os.path.isdir(args.d):
		print('please provide an existing directory:',args.d,'does not exist')
	if args.d == None: args.d = ''
	chunks = False if args.no_split else None
	if  not args.minimum_silence_duration:
		minimum_silence_duration = 0.2
	elif type(args.minimum_silence_duration) != float:
		print('could not process',args.minimum_silence_duration,'should be float')
		minimum_silence_duration = 0.2
	else: minimum_silence_duration = args.minimum_silence_duration
	print('minimum_silence_duration is:',minimum_silence_duration)
	make_kaldi_recources(audio_filename = args.fn, goal_dir = args.d, 
		force_save=args.force, chunks = chunks, 
		minimum_silence_duration = minimum_silence_duration)






