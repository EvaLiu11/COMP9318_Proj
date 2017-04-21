# Phenoms
vowels= ['AA','AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH'
, 'IY', 'OW', 'OY', 'UH', 'UW']
consonants= ['P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 
'N', 'NG', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

def read_data(file_path):
	with open(file_path) as f:
		lines = f.read().splitlines()
	return lines

def stress_map(pronunciation):
	pn_list = pronunciation.split()
	return [1 if '1' in num else 0 for num in pn_list]

class word(object):
	"""docstring for word"""
	def __init__(self, word_string):
		super(word, self).__init__()
		self.word = word_string.split(':')[0]
		self.pronunciation = word_string.split(':')[1]
		self.stress_map = stress_map(self.pronunciation)

for line in read_data('asset/training_data.txt'):
	x = word(line)
	print(x.stress_map)