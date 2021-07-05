'''
Copyright 2021, Martijn Bentum, Humanities Lab, Radboud University Nijmegen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''


import sys
'''whether to fail if a number cannot be mapped to int'''
hard_fail_global = None

def frisian_numbers():
	return open('../NUMBERS/frisian_numbers').read().split('\n')

def dutch_numbers():
	return open('../NUMBERS/dutch_numbers').read().split('\n')

def number_dict(language = 'frisian'):
	'''a number dict contains a digit and a word column; map digits to words.'''
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

	def __init__(self,spaces = False,hard_fail = False):
		'''spaces 		whether to separate number word e.g. vier en twintig / vierentwintig
		hard_fail 		whether to raise an error if a number can be converted to int
						mainly for debugging, if false will return empty if error occurs
		'''
		global hard_fail_global
		hard_fail_global = hard_fail
		self.spaces = spaces
		self.frisian_number_dict = number_dict('frisian')
		self.dutch_number_dict = number_dict(language = 'dutch')

	def toword(self,number,language = 'frisian', spaces = None):
		'''map digit number to word number'''
		if spaces != None and type(spaces) == bool: self.spaces = spaces
		if language.lower() == 'frisian':nd = self.frisian_number_dict 
		if language.lower() == 'dutch':nd = self.dutch_number_dict
		return handle_number(number,nd,spaces)
	

def handle_number(number, nd = None,spaces =False):
	'''converts a number into the word representing to number.
	upto (not including) a billion
	'''
	str_number = str(number)
	minus, number = _handle_minus(str_number)
	if type(number) == float or '.' in str(number) or ',' in str(number):
		return _handle_float(number,nd,spaces,minus)
	number = convert_number_to_int(number)
	len_number = len(str_number)
	if len_number == 1: return minus +_handle_single_digit(number,nd,spaces)
	if len_number == 2: return minus +_handle_two_digit(number,nd,spaces)
	if len_number == 3: return minus +_handle_three_digit(number,nd,spaces)
	if len_number == 4: return minus +_handle_four_digit(number,nd,spaces)
	if len_number == 5: return minus +_handle_five_digit(number,nd,spaces)
	if len_number == 6: return minus +_handle_six_digit(number,nd,spaces)
	if len_number == 7: return minus +_handle_seven_digit(number,nd,spaces)
	if len_number == 8: return minus +_handle_eight_digit(number,nd,spaces)
	if len_number == 9: return minus +_handle_nine_digit(number,nd,spaces)
	print('handles number upto length 9, return default value 42 ',_handle_two_digit('42',nd,spaces))
	return _handle_two_digit('42',nd,spaces)

def _handle_minus(str_number):
	'''checks whether the number start with a minus sign and prepends 
	the number word with min if it does.'''
	if len(str_number) == 0: return '',''
	if str_number[0] == '-':
		if len(str_number) == 1: return '',''
		str_number = str_number[1:]
		minus = 'min '
	else: minus = ''
	return minus , str_number

def _handle_float(number,nd = None,spaces = False, minus = ''):
	'''handles float numbers.'''
	sep = ' ' if spaces else ''
	number = str(number)
	number = number.replace(',','.')
	if not number.count('.') == 1: return handle_number(number.replace('.',''),nd,spaces)
	before_decimal, after_decimal = number.split('.')
	if before_decimal == '': before_decimal = 0
	if after_decimal == '': after_decimal = 0
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
	before_decimal_word = handle_number(before_decimal,nd,spaces)
	after_decimal_word =  _handle_number_after_decimal(after_decimal,nd,spaces)
	return minus + before_decimal_word + ' komma ' + after_decimal_word


def _handle_number_after_decimal(after_decimal,nd,spaces):
	'''handles number after the decimal
	if there are more than two digits they are spelled out one by one
	e.g. 3.781 three comma seven eight one
	'''
	sep = ' ' if spaces else ''
	if len(after_decimal) < 3 and after_decimal[0] != '0': return handle_number(after_decimal,nd,spaces)
	output = []
	for digit in after_decimal:
		output.append( handle_number(digit) )
	return sep.join(output)

# helper function to handle digits of different lengths

def _handle_single_digit(number, nd= None,spaces = False):
	if not len(str(number)) == 1: return handle_number(number,nd,spaces)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	return nd[number]

def _handle_two_digit(number, nd = None,spaces = False):
	if not len(str(number)) == 2: return handle_number(number,nd,spaces)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	str_number = str(number)
	first_digit = _handle_two_digit(str_number[0] +'0',nd,spaces)
	last_digit = _handle_single_digit(str_number[-1],nd,spaces)
	if spaces: return last_digit + ' en ' + first_digit
	return last_digit + 'en' + first_digit

def _handle_three_digit(number, nd = None,spaces = False):
	if not len(str(number)) == 3: return handle_number(number,nd,spaces)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	sep = ' ' if spaces else ''
	str_number = str(number)
	if str_number[-2:] != '00':
		last_digits = _handle_two_digit(str_number[-2:],nd,spaces)
	else: last_digits = ''
	if str_number[0] != '1': 
		first_digit = _handle_single_digit(str_number[0],nd,spaces) + sep
	else: first_digit = ''
	first_digit += _handle_three_digit('100',nd,spaces)
	return first_digit + sep + last_digits

def _handle_four_digit(number, nd = None,spaces = False):
	if not len(str(number)) == 4: return handle_number(number,nd,spaces)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	sep = ' ' if spaces else ''
	str_number = str(number)
	if str_number[1] != '0':
		first_digits = _handle_two_digit(str_number[:2],nd,spaces)
		last_digits = _handle_two_digit(str_number[-2:],nd,spaces)
		return first_digits + sep + _handle_three_digit('100',nd,spaces)+ sep + last_digits
	if str_number[0] != '1': 
		first_digit = _handle_single_digit(str_number[0],nd,spaces) 
		first_digit += sep
	else: first_digit = ''
	first_digit += _handle_four_digit('1000',nd,spaces)
	last_digits = _handle_two_digit(str_number[-2:],nd,spaces)
	return first_digit + sep + last_digits
		
def _handle_five_digit(number, nd = None,spaces=False):
	if not len(str(number)) == 5: return handle_number(number,nd,spaces)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	sep = ' ' if spaces else ''
	str_number = str(number)
	first_digits = _handle_two_digit(str_number[:2],nd,spaces)
	last_digits = _handle_three_digit(str_number[2:],nd,spaces)
	return first_digits + sep + _handle_four_digit('1000',nd,spaces) + sep + last_digits

def _handle_six_digit(number, nd = None,spaces=False):
	if not len(str(number)) == 6: return handle_number(number,nd,spaces)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	sep = ' ' if spaces else ''
	str_number = str(number)
	first_digits = _handle_three_digit(str_number[:3],nd,spaces)
	last_digits = _handle_three_digit(str_number[3:],nd,spaces)
	return first_digits + sep + _handle_four_digit('1000',nd) + sep + last_digits

def _handle_seven_digit(number, nd = None,spaces = False):
	if not len(str(number)) == 7: return handle_number(number,nd,spaces)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	sep = ' ' if spaces else ''
	str_number = str(number)
	first_digit = _handle_single_digit(str_number[0],nd,spaces)
	last_digits = _handle_six_digit(str_number[1:],nd,spaces)
	return first_digit + sep + _handle_seven_digit(10**6,nd,spaces) + sep + last_digits

def _handle_eight_digit(number, nd = None,spaces = False):
	if not len(str(number)) == 8: return handle_number(number,nd,spaces)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	sep = ' ' if spaces else ''
	str_number = str(number)
	first_digits = _handle_two_digit(str_number[:2],nd,spaces)
	last_digits = _handle_six_digit(str_number[2:],nd,spaces)
	return first_digits + sep + _handle_seven_digit(10**6,nd,spaces) + sep + last_digits

def _handle_nine_digit(number, nd = None,spaces = False):
	if not len(str(number)) == 9: return handle_number(number,nd,spaces)
	if not nd: nd= number_dict()
	number = convert_number_to_int(number)
	if number in nd.keys(): return nd[number]
	sep = ' ' if spaces else ''
	str_number = str(number)
	first_digits = _handle_three_digit(str_number[:3],nd,spaces)
	last_digits = _handle_six_digit(str_number[3:],nd,spaces)
	return first_digits + sep + _handle_seven_digit(10**6,nd,spaces) + sep + last_digits

def convert_number_to_int(number, hard_fail = False):
	if hard_fail_global != None: hard_fail = hard_fail_global
	try:return int(number)
	except: 
		if hard_fail: 
			print(sys.exc_info(),number)
			raise ValueError('could not convert number to int') 
		print(sys.exc_info(),number, 'could not convert number returning empty string')
		return ''
	
	
	
