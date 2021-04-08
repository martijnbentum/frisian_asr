import os
'''
ngram -order 3 -unk -ppl council_transcriptions_dev -lm /home/eyilmaz/main2/latest_ASR_exps/fame/input/LM_interp_NL_xxxlarge -mix-lm council.lm -lambda 0.5 -debug 1 > mix_50_out
'''

c0 = 'ngram -order 3 -unk -ppl ../LM/council_transcriptions_dev -lm '
c1 = ' -mix-lm '
c2 = ' -lambda '
c3 = ' -debug 1 > '


def lambda_test_fame_council():
	'''assumes lamachine is loaded or SRILM is in path'''
	lm_fame = '/home/eyilmaz/main2/latest_ASR_exps/fame/input/LM_interp_NL_xxxlarge'
	lm_council = '/vol/tensusers/mbentum/FRISIAN_ASR/LM/council.lm'
	#l is lambda, the weight of the main model
	for l in range(50,105,5):
		command = c0 + lm_fame + c1 + lm_council + c2 + str(l/100) + c3 
		command += '../LM/fame_council_mix_' + str(l)
		print(command)
		os.system(command)
		
	
