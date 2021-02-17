from texts.models import Text, Language, TextType, Source

languages = 'Dutch,Frysian'.split(',')
abbreviations= 'nl,fr'.split(',')

def make_languages(languages=languages,abbreviations=abbreviations, remove_old = False):
	if remove_old: Language.objects.all().delete()
	for l, a in zip(languages, abbreviations):
		language = Language(name=l,abbreviation=a)
		try:language.save()
		except:print('could not save:',l)

texttypes = 'manual transcription,automatic transcription,council notes'
texttypes += ',wikipedia article,book'
texttypes = texttypes.split(',')
def make_texttypes(texttypes=texttypes, remove_old = False):
	if remove_old: TextType.objects.all().delete()
	for tt in texttypes:
		texttype = TextType(name = tt)
		try:texttype.save()
		except:print('could not save:',tt)


sources = 'frysian radio broadcasts,frysian council notes,frysian council transcripts'
sources += 'wikipedia,gutenberg,magazine,newspaper,book,cgn,cow'
sources = sources.split(',')

def make_sources(sources = sources,remove_old=False):
	if remove_old: Source.objects.all().delete()
	for s in sources:
		source = Source(name = s)
		try: source.save()
		except:print('could not save:',s)

		

		
	
