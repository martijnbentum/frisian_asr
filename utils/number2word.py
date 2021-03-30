import sys


def frisian_numbers():
	return open('../frisian_numbers').read().split('\n')

def dutch_numbers():
	return open('../dutch_numbers').read().split('\n')

def number_dict(language = 'frisian'):
	if language == 'frisian':t = frisian_numbers()
	elif language == 'dutch':t=dutch_numbers()
	else: 
		print(language, 'unknown using default frisian language')
		t = frisian_numbers()
	return dict([[int(line.split(' ')[0]),line.split(' ')[1]] for line in t if line])

class Number2word:
	'''converts a number into the word representing to number.
	by using this class the number_dict does not have to be reloaded.
	you can also use handle_number
	'''

	def __init__(self):
		self.frisian_number_dict = number_dict('frisian')
		self.dutch_number_dict = number_dict(language = 'dutch')

	def toword(self,number,language = 'frisian'):
		if language.lower() == 'frisian':nd = self.frisian_number_dict 
		if language.lower() == 'dutch':nd = self.dutch_number_dict
		return handle_number(number,nd)
	

def handle_number(number, nd = None):
	'''converts a number into the word representing to number.
	upto (not including) a billion
	'''
	if type(number) == float or '.' in str(number) or ',' in str(number):
		return _handle_float(number,nd)
	number = convert_number_to_int(number)
	len_number = len(str(number))
	if len_number == 1: return _handle_single_digit(number,nd)
	if len_number == 2: return _handle_two_digit(number,nd)
	if len_number == 3: return _handle_three_digit(number,nd)
	if len_number == 4: return _handle_four_digit(number,nd)
	if len_number == 5: return _handle_five_digit(number,nd)
	if len_number == 6: return _handle_six_digit(number,nd)
	if len_number == 7: return _handle_seven_digit(number,nd)
	if len_number == 8: return _handle_eight_digit(number,nd)
	if len_number == 9: return _handle_nine_digit(number,nd)
	print('handles number upto length 9, return default value 42 ',_handle_two_digit('42'))
	return handle_two_digit('42',nd)

def _handle_float(number,nd = None):
	number = str(number)
	number = number.replace(',','.')
	assert number.count('.') == 1
	before_decimal, after_decimal = number.split('.')
	bdint = convert_number_to_int(before_decimal) 
	adint = convert_number_to_int(after_decimal) 
	if bdint == 0 and adint == 5: 
		if nd[2] == 'twa':return 'healwei'
		else: return 'half'
	elif bdint ==1 and adint == 5: 
		if nd[2] == 'twa': return 'oardel'
		else: return 'anderhalf'
	elif 0 < bdint <10 and adint == 5: 
		if nd[2] == 'twa' :return handle_number(before_decimal) + 'eninheal'
		else: return handle_number(before_decimal) + 'eneenhalf'
	before_decimal_word = handle_number(before_decimal)
	after_decimal_word =  _handle_number_after_decimal(after_decimal)
	return before_decimal_word + 'komma' + after_decimal_word


def _handle_number_after_decimal(after_decimal):
	if len(after_decimal) < 3 and after_decimal[0] != '0': return handle_number(after_decimal)
	output =''
	for digit in after_decimal:
		output += handle_number(digit)
	return output

def _handle_single_digit(number, nd= None):
	if not len(str(number)) == 1: return handle_number(number)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	return nd[number]

def _handle_two_digit(number, nd = None):
	if not len(str(number)) == 2: return handle_number(number)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	str_number = str(number)
	first_digit = _handle_two_digit(str_number[0] +'0',nd)
	last_digit = _handle_single_digit(str_number[-1],nd)
	return last_digit + 'en' + first_digit

def _handle_three_digit(number, nd = None):
	if not len(str(number)) == 3: return handle_number(number)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	str_number = str(number)
	last_digits = _handle_two_digit(str_number[-2:],nd)
	first_digit = _handle_three_digit(str_number[0] + '00',nd)
	return first_digit + last_digits

def _handle_four_digit(number, nd = None):
	if not len(str(number)) == 4: return handle_number(number)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	str_number = str(number)
	if str_number[1] != '0':
		first_digits = _handle_two_digit(str_number[:2],nd)
		last_digits = _handle_two_digit(str_number[-2:],nd)
		return first_digits + _handle_three_digit('100') + last_digits
	first_digit = _handle_four_digit(str_number[0] +'000',nd)
	last_digits = _handle_two_digit(str_number[-2:],nd)
	return first_digit + last_digits
		
def _handle_five_digit(number, nd = None):
	if not len(str(number)) == 5: return handle_number(number)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	str_number = str(number)
	first_digits = _handle_two_digit(str_number[:2],nd)
	last_digits = _handle_three_digit(str_number[2:],nd)
	return first_digits + _handle_four_digit('1000',nd) + last_digits

def _handle_six_digit(number, nd = None):
	if not len(str(number)) == 6: return handle_number(number)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	str_number = str(number)
	first_digits = _handle_three_digit(str_number[:3],nd)
	last_digits = _handle_three_digit(str_number[3:],nd)
	return first_digits + _handle_four_digit('1000',nd) + last_digits

def _handle_seven_digit(number, nd = None):
	if not len(str(number)) == 7: return handle_number(number)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	str_number = str(number)
	first_digit = _handle_single_digit(str_number[0],nd)
	last_digits = _handle_six_digit(str_number[1:],nd)
	return first_digit + _handle_seven_digit(10**6,nd) + last_digits

def _handle_eight_digit(number, nd = None):
	if not len(str(number)) == 8: return handle_number(number)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	str_number = str(number)
	first_digits = _handle_two_digit(str_number[:2],nd)
	last_digits = _handle_six_digit(str_number[2:],nd)
	return first_digits + _handle_seven_digit(10**6,nd) + last_digits

def _handle_nine_digit(number, nd = None):
	if not len(str(number)) == 9: return handle_number(number)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	str_number = str(number)
	first_digits = _handle_three_digit(str_number[:3],nd)
	last_digits = _handle_six_digit(str_number[3:],nd)
	return first_digits + _handle_seven_digit(10**6,nd) + last_digits

def convert_number_to_int(number, hard_fail = True):
	try:return int(number)
	except: 
		if hard_fail: 
			raise ValueError(sys.exc_info()+ ' ' +str(number)+ ' could not convert number to int') 
		print(sys.exc_info(),number, 'could not convert number returning empty string')
		return ''
	
	
	
