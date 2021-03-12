from texts.models import Text
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def get_texts(language= 'Dutch'):
	t =Text.objects.filter(text_type__name = 'manual transcription', 
		multiple_languages = False, main_language__name=language)
	return t

def get_transcriptions(language= 'Dutch'):
	t = get_texts(language)
	output = []
	for x in t:
		transcription = x.transcription
		if not transcription or type(transcription) == str: continue
		if not transcription.code_switched: output.append(transcription)
	return output

def get_dutch_frisian_transcriptions():
	dutch = get_transcriptions()
	frisian = get_transcriptions('Frisian')
	return dutch, frisian

def make_list(transcriptions):
	output = []
	for x in transcriptions:
		temp = x.text_without_tags
		if temp: output.append(temp)
	return output

def train(dutch = None, frisian=None):
	if None in [dutch,frisian]:
		dutch, frisian = get_dutch_frisian_transcriptions()
	dl, fl = make_list(dutch), make_list(frisian)
	X = np.array(dl + fl)
	y = np.array(['dutch']*len(dl) + ['frisian']*len(fl))
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	cnt = CountVectorizer(analyzer = 'char',ngram_range=(2,2))
	pipeline = Pipeline([
		('vectorizer',cnt),
		('model',MultinomialNB())
	])
	pipeline.fit(X_train,y_train)
	y_pred = pipeline.predict(X_test)
	cm = confusion_matrix(y_test,y_pred)
	print(classification_report(y_test,y_pred))
	return dutch, frisian, X, y, cnt, pipeline, cm

