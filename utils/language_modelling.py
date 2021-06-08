import os
'''
ngram -order 3 -unk -ppl council_transcriptions_dev -lm /home/eyilmaz/main2/latest_ASR_exps/fame/input/LM_interp_NL_xxxlarge -mix-lm council.lm -lambda 0.5 -debug 1 > mix_50_out


council model:
ngram-count -order 3 -vocab vocab_council  -limit-vocab -text council_notes_manual_transcriptions_train -lm council.lm -unk -kndiscount -kndiscount1 -kndiscount2 -kndiscount3 -interpolate -interpolate1 -interpolate2 -interpolate3 -gt1min 0 -gt2min 0 -gt3min 0
'''


lm_dir = '/vol/tensusers/mbentum/FRISIAN_ASR/LM/'
lm_fame = '/home/eyilmaz/main2/latest_ASR_exps/fame/input/LM_interp_NL_xxxlarge'
lm_council = lm_dir + 'council.lm'

c0 = 'ngram -order 3 -unk -ppl ../LM/manual_transcriptions_dev.txt -lm '
c1 = ' -mix-lm '
c2 = ' -lambda '
c3 = ' -debug 2 > '

#implicit in srilm, needs to be in vocab of rnn
sos = '<s>'
eos = '</s>'







def test_fame_council(lamb = .7, debug = 1,output_filename = 'default', run = False):
	command = c0 + lm_fame + c1 + lm_council + c2 + str(lamb) 
	command += '-debug ' + str(debug) + ' > ' + lm_dir + output_filename
	print(command)
	if run: os.system(command)

def lambda_test_fame_council():
	'''assumes lamachine is loaded or SRILM is in path'''
	#l is lambda, the weight of the main model
	for l in range(5,50,5):
		command = c0 + lm_fame + c1 + lm_council + c2 + str(l/100) + c3 
		command += lm_dir + 'fame_council_mix_' + str(l)
		print(command)
		os.system(command)
