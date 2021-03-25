import Levenshtein
from Levenshtein import distance, ratio


def compare(string1,string2):
	'''
	compares two strings on char similarity with levenshtein ratio
	returns a value between 0 and 1, 0 no overlap, 1 complete overlap
	'''
	return ratio(string1,string2)

