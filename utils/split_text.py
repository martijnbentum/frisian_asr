import random
def split_text(text='',input_filename= '',
	output_filename='',train = True,dev = True, ratio = .1,save=True):
	'''split a cleaned text into sections train, dev, test
	text 				a text to be split into training dev test set
	input_filename 		filename of a text to be split in training dev test set
	output_filename 	base name which will be postpended with train dev test
	train 				whether to create a training set, if not text will be
						split evenly between dev and test
	dev 				whether to create a development set
	ratio 				percentage of material reserved for dev and test
						if both dev and test are created training precentage =
						1 - ratio *2
	'''
	if not text and not input_filename: print('please provide text or input_filename')
	if not text and input_filename: text = open(input_filename).read()
	sentences = text.split('\n')
	n = sum([train,dev])
	names = []
	if train:
		train_ratio = 1 - n *ratio 
		ratios =[train_ratio] + [ratio] * (n)
		names.append('train')
	else: ratios = [0.5,0.5]
	if dev: names.append('dev')
	names.append('test')
	indices = list(range(len(sentences)))
	random.shuffle(indices)
	start,end = 0,0
	output_sentences,output_indices = [],[]
	for i,r in enumerate(ratios):
		end = start + int(r * len(sentences))
		selected_indices = indices[start:end]
		selected_sentences = extract_from_list(sentences,selected_indices)
		if save:
			with open(output_filename + '_' + names[i],'w') as fout:
				fout.write('\n'.join(selected_sentences))
		output_sentences.append('\n'.join(selected_sentences))
		output_indices.append('\n'.join(map(str,selected_indices)))
		start = end
	return output_sentences 
	
		

def extract_from_list(l,indices):
	'''extracts items from list based on a list of indices'''
	return [l[i] for i in indices]


def create_manual_transcriptions():
	t = Text.objects.filter(text_type__name = 'manual transcription')
	o = []
	for x in t:
		o.append(x.text_with_tags)
	with open('../LM/manual_transcriptions','w') as fout:
		fout.write('\n'.join(o))

def create_train_dev_test_manual_transcriptions():
	return split_text(input_filename = '../LM/manual_transcriptions',
		output_filename='../LM/manual_transcriptions',train=True)

def create_train_test_council_notes():
	return split_text(input_filename = '../LM/council_notes_cleaned_labelled',
		output_filename='../LM/council_notes',train=True,dev=False)
