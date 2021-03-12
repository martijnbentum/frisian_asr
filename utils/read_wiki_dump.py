import lxml
from lxml import etree
import re
import progressbar as pb
from texts.models import Text, Language, TextType, Source

#https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/fywiktionary/20210201/
wikidump = '/vol/tensusers/mbentum/FRISIAN_ASR/fywiki-20210201-pages-articles.xml'
source = Source.objects.get(name= 'wikipedia')
text_type = TextType.objects.get(name = 'wikipedia article')
frisian = Language.objects.get(name='Frisian')

def read_into_database(xml = None,texts = None, start = 0):
	if xml == None and texts == None:xml = get_xml(wikidump)
	if texts == None: texts = get_texts(xml)
	o = []
	bar = pb.ProgressBar()
	bar(range(len(texts[start:])))
	for i,t in enumerate(texts[start:]):
		bar.update(i)
		if not t.text: continue
		o.append(handle_text_xml(t))
	return xml, texts

def handle_text_xml(t):
	clean_text = clean(t.text)
	title = text_xml2title(t)
	file_id = text_xml2id(t)
	t = Text.objects.filter(file_id = file_id, title = title)
	if t: print('already found wiki entry in database')
	else: 
		t = Text(clean_text = clean_text, title = title, file_id = file_id,
			main_language = frisian, source = source, text_type = text_type)
		t.save()
		t.all_languages.add(frisian)
	return t


def get_xml(filename):
	t = open(filename).read()
	xml = etree.fromstring(t)
	return xml

def get_texts(xml):
	return xml.findall('.//{http://www.mediawiki.org/xml/export-0.10/}text')

def text_xml2title(x):
	try: return x.getparent().getparent().getchildren()[0].text
	except: return ''

def text_xml2id(x):
	try: return x.getparent().getparent().getchildren()[2].text
	except: return ''

def remove_section(t, start = '{{', end='}}'):
	starts, ends = [],[]
	o = ''
	last_start = 0
	sindex,eindex = -1,0
	while True:
		sindex = t.find(start, sindex+1)
		eindex = t.find(end, eindex+1)
		# print(o,sindex,eindex)
		if 0 <= sindex < eindex:
			starts.append(sindex)
			ends.append(eindex)
			o += t[last_start:sindex]
			last_start = eindex + len(end)
		else:
			o += t[last_start:]
			break
		# print(o,sindex,eindex)
	return o


def show_rejected_items(rejected,rejection_type, join_char='\n'):
	#print('---- rejecting lines based on these stop words: ----')
	#print('\t'.join(stop_words))
	print('removed the following', rejection_type, ':', len(rejected))
	print('='*90)
	print(join_char.join(list(set(rejected))))
	print('---- end of rejected',rejection_type,' ---- ')


def remove_lines(t,stop_words='cet,cest'.split(','),show_rejected = False,return_rejected=False,rejection_type='lines'):
	lines = t.split('\n')
	output ,rejected = [],[]
	for line in lines:
		if line == '':continue
		words = line.split(' ')
		ok = True
		for word in words:
			if word.strip() in stop_words: 
				ok = False
				break
		if ok: output.append(line)
		else:rejected.append(line)
	if show_rejected and rejected: show_rejected_items(rejected,rejection_type)
	t = '\n'.join(output)
	if return_rejected: return t, rejected
	return t

def remove_short_lines(t,show_rejected):
	lines = t.split('\n')
	output ,rejected = [],[]
	for line in lines:
		ok = True
		if len(line) < 4: ok =False
		if ok: output.append(line)
		else:rejected.append(line)
	if show_rejected and rejected: show_rejected_items(rejected,'to short lines')
	t = '\n'.join(output)
	return t
	

def filter_english(t,show_rejected=False):
	stop_words='you,this,the,about,here,there,up,down,now,your,she'
	stop_words += ',please,contact,local,if,apologies,today,across,encourage,campaign,articles'
	stop_words += ',write,talk,page,pages,conflicts,project,volenteers,interface,redirect'
	stop_words += ',photos,invite,development,community,framework,grow,many,pathways'
	stop_words += ',style,redirect,ceremony,award,wikidata,changes,recent,watchlist'
	stop_words += ',description,maintenance,functions,contest,function,include,included'
	stop_words += ',available,submission,request,image,images,videos,video,image,images'
	stop_words = stop_words.split(',')
	t= remove_lines(t,stop_words=stop_words, show_rejected=show_rejected,rejection_type='english lines')
	return t

def remove_words(t,extra=[],show_rejected=False,return_rejected=False):
	words = t.split(' ')
	rejected = []
	output = []
	extra.extend( 'http,http:,htm,#redirect,www,#'.split(',') )
	for line in t.split('\n'):
		words = line.split(' ')
		temp = []
		for word in words:
			bad = False
			if word.strip() in extra: bad = True
			if '/' in word: bad = True
			if '_' in word: bad = True
			if '|' in word: bad = True
			if '$' in word: bad = True
			if '<' in word: bad = True
			if '>' in word: bad = True
			if bad:rejected.append(word.replace('\n',''))
			else:temp.append(word)
		output.append(' '.join(temp))
	if show_rejected and rejected: show_rejected_items(rejected,'words',' ')
	t = '\n'.join(output)
	return t
			


def remove_abbrev(t,show_rejected=False):
	rejected,output = [],[]
	for line in t.split('\n'):
		words = line.split(' ')
		temp = []
		for word in words:
			bad = False
			w = word.strip('. ')
			if len(w) >2 and '.' in w[1:-1]: bad= True
			if bad:rejected.append('_'+w+'_')
			else:temp.append(word)
		output.append(' '.join(temp))
	if show_rejected and rejected: show_rejected_items(rejected,'abbreviation',' &&& ')
	t = '\n'.join(output)
	return t
		

def remove_sections(t):
	for start,end in zip(['__','{{','{','<ref','<div','<font','[','<'],['__','}}','}','</ref>','</div>','</font>',']','>']):
		t = remove_section(t,start=start,end=end)
	return t


def remove_tags(t):
	t = re.sub('(\d\.)','0',t)
	# t = re.sub('(\d+)','<nûmer>',t)
	tags = "<br>,''','',== , ==,=,+,*,!,&nbsp,km²,&,-,;,:,(,), ',' ,–,†,},{,±".split(",")
	for tag in tags:
		if '=' in tag: rep = '.'
		else: rep = ' '
		t = t.replace(tag, rep)
		# print([tag],'--->',t,'*'*100)
	t = t.replace(',',' ')
	t = t.replace('"',' ')
	t = re.sub(' +',' ',t)

	# t = re.sub('(\D\.[a-zA-Z])','',t)
	return t

def handle_end_of_line(t):
	t = re.sub(' +',' ',t)
	t = re.sub('(\.)','\n',t)
	t = re.sub('\n +','\n',t)
	t = re.sub(' +\n','\n',t)
	t = re.sub('\n+','\n',t)
	return t

def clean(t,show_rejected=False):
	t = t.lower()
	t = remove_sections(t)
	t = remove_tags(t)
	t = remove_abbrev(t,show_rejected=show_rejected)
	t = handle_end_of_line(t)
	t = remove_lines(t)
	t = filter_english(t,show_rejected=show_rejected)
	t = remove_words(t,show_rejected=show_rejected)
	t = remove_short_lines(t,show_rejected=show_rejected)
	return t

