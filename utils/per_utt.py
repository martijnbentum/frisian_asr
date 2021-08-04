import os

directory = '/vol/tensusers/mbentum/FRISIAN_ASR/PER_UTT/'

def make_utterances():
	f_council_tags = directory + 'per_utt_council_tags'
	f_council_no_tags = directory + 'per_utt_council_no_tags'
	f_fame_tags = directory + 'per_utt_fame_tags'
	f_fame_no_tags = directory + 'per_utt_fame_no_tags'
	uct = Utterances(f_council_tags)
	ucnt = Utterances(f_council_no_tags)
	ucnt.set_language(uct.id2language)
	uft = Utterances(f_fame_tags)
	ufnt = Utterances(f_fame_no_tags)
	ufnt.set_language(uft.id2language)
	print(uct)
	print(ucnt)
	print(uft)
	print(ufnt)
	


class Utterances:
	def __init__(self,filename):
		self.filename = filename
		self.text = open(filename).read().split('\n')
		self._process_text()
		self._handle_language()
		self._handle_csid()
		self._make_wer()

	def __repr__(self):
		return self.filename + ' #utterances: ' + str(len(self.utterances))

	def __str__(self):
		n = 20
		m = 'filename:'.ljust(n) +self.filename + '\n'
		m += '#utterances:'.ljust(n) +str(len(self.utterances))+'\n'
		m += '#dutch:'.ljust(n) +str(self.ndutch) +'\n'
		m += '#frisian:'.ljust(n)+str(self.nfrisian)+'\n'
		m += '#mix:'.ljust(n)+str(self.nmix)+'\n'
		m += '#unknown:'.ljust(n)+str(self.nunknown)+'\n'
		m += '-'*50+'\n'
		m += ' ' *20 +'WER'.ljust(15) + 'SER' + '\n'
		m += '-'*50+'\n'
		m += self.show_wer_ser()
		return m

	def _process_text(self):
		self._make_id_dict()
		self._make_utterances()

	def _make_id_dict(self):
		self.id2lines = {}
		for line in self.text:
			if not line:continue
			pk = int(line.split('pk-')[1].split(' ')[0])
			if pk not in self.id2lines.keys():
				self.id2lines[pk] = []
			l = line.split('pk-')[1].split(' ')[2:]
			l = ' '.join([x for x in l if x])
			self.id2lines[pk].append(l)

	def _make_utterances(self):
		self.utterances = []
		for key in self.id2lines.keys():
			ref, hyp, eva, csid = self.id2lines[key]
			self.utterances.append(Utterance(key,ref,hyp,eva,csid))

	def _handle_language(self):
		self.dutch = [x for x in self.utterances if x.language == 'Dutch']
		self.ndutch = len(self.dutch)
		self.frisian= [x for x in self.utterances if x.language == 'Frisian']
		self.nfrisian = len(self.frisian)
		self.mix= [x for x in self.utterances if x.language == 'mix']
		self.nmix= len(self.mix)
		self.unknown = [x for x in self.utterances if x.language == 'unknown']
		self.nunknown = len(self.unknown)

	def _handle_csid(self):
		self.ncorrect, self.ndeletions, self.nsubstitutions = 0,0,0
		self.ninsertions, self.nerrors, self.nwords_ref = 0,0,0
		for u in self.utterances:
			self.ncorrect += u.ncorrect 
			self.ndeletions+= u.ndeletions
			self.nsubstitutions+= u.nsubstitutions
			self.ninsertions += u.ninsertions
			self.nerrors += u.nerrors
			self.nwords_ref += u.nwords_ref

	def compute_wer(self, utterances = None):
		if not utterances: utterances = self.utterances
		errors, nwords_ref = 0,0
		for u in utterances:
			errors += u.nerrors
			nwords_ref += u.nwords_ref
		return round(100 * errors / nwords_ref,2)

	def compute_ser(self,utterances = None):
		if not utterances: utterances = self.utterances
		errors, nutterances = 0, len(utterances)
		for u in utterances:
			if not u.all_correct: errors += 1
		return round(100 * errors / nutterances, 2)

	def _make_wer(self):
		for x in 'all,dutch,frisian,mix'.split(','):
			if x == 'all': utterances = self.utterances
			else: utterances = getattr(self,x)
			wer = self.compute_wer(utterances)
			setattr(self,'wer_'+x,wer)

	def _make_ser(self):
		for x in 'all,dutch,frisian,mix'.split(','):
			if x == 'all': utterances = self.utterances
			else: utterances = getattr(self,x)
			wer = self.compute_ser(utterances)
			setattr(self,'ser_'+x,wer)

	@property
	def id2language(self):
		if hasattr(self,'_id2language'):return self._id2language
		self._id2language = {}
		for u in self.utterances:
			self._id2language[u.ide] = u.language
		return self._id2language

	def set_language(self,id2language):
		for u in self.utterances:
			u.language = id2language[u.ide]
		self._handle_language()
		self._make_wer()
		self._make_ser()

	def show_wer_ser(self):
		m = ''
		for x in 'all,dutch,frisian,mix'.split(','):
			attr_wer = 'wer_'+x
			attr_ser = 'ser_'+x
			if not hasattr(self,attr_wer):self._make_wer()
			if not hasattr(self,attr_ser):self._make_ser()
			m += x.ljust(20) + str(getattr(self,attr_wer)).ljust(15)
			m += str(getattr(self,attr_ser)) + '\n'
		return m
			
			
			
			



class Utterance:
	def __init__(self,ide, ref, hyp, evaluation, csid):
		self.ide = ide
		self.ref= ref
		self.hyp= hyp
		self.evaluation= _make_evaluation(evaluation)
		self.csid = csid.split(' ')
		self.words = []
		self._handle_language()
		self._handle_csid()
		self._make_words()
		self._compute_wer()


	def __repr__(self):
		correct = 'all' if self.all_correct else str(self.ncorrect)
		return self.ref + ' | # words correct: ' + correct

	def _handle_language(self):
		fr,nl= False, False
		for word in self.ref.split(' '):
			if word.endswith('-nl'): nl = True
			elif word.endswith('-fr'): fr = True
		if nl and fr:self.language = 'mix'
		elif nl:self.language = 'Dutch'
		elif fr:self.language = 'Frisian'
		else: self.language = 'unknown'

	def _handle_csid(self):
		self.ncorrect = int(self.csid[0])
		self.nsubstitutions = int(self.csid[1])
		self.ninsertions = int(self.csid[2])
		self.ndeletions = int(self.csid[3])
		x = self.nsubstitutions == self.ninsertions == self.ndeletions == 0
		self.all_correct = True if x else False
	

	def _make_words(self):
		words = zip(self.ref.split(' '),self.hyp.split(' '), self.evaluation)
		for ref_word, hyp_word, evaluation in words:
			self.words.append(Word(ref_word,hyp_word,evaluation,self.ide))

	def _compute_wer(self):
		self.nwords_ref = len([w for w in self.ref.split(' ') if w != '***'])
		self.nerrors = self.nsubstitutions + self.ninsertions +self.ndeletions
		if self.nwords_ref == 0: self.wer = 0.0
		else: self.wer = round(100 * self.nerrors / self.nwords_ref, 2)


class Word:
	def __init__(self,ref_word, hyp_word, evaluation,sentence_ide):
		self.ref_word = ref_word
		self.hyp_word = hyp_word
		self.evaluation= evaluation
		self.sentence_ide = sentence_ide
		self._handle_language()
		
	def _handle_language(self):
		self.frisian2dutch= False
		self.dutch2frisian= False
		if self.ref_word.endswith('-fr'):self.ref_language = 'Frisian'
		elif self.ref_word.endswith('-nl'):self.ref_language = 'Dutch'
		else: self.ref_language = 'unknown'
		if self.hyp_word.endswith('-fr'):self.hyp_language = 'Frisian'
		elif self.hyp_word.endswith('-nl'):self.hyp_language = 'Dutch'
		else: self.hyp_language = 'unknown'
		self.lanugage_match = self.ref_language == self.hyp_language
		if not self.lanugage_match:
			if self.ref_language == 'Frisian' and self.hyp_language == 'Dutch':
				self.frisian2dutch = True
			if self.ref_language == 'Dutch' and self.hyp_language == 'Frisian':
				self.dutch2frisian= True

	

def _make_evaluation(evaluation):
	o = []
	for x in evaluation.split(' '):
		if x in ['C','I','S','D']: o.append(x)
	return o
