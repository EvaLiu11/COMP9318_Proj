# Phenoms
vowels= ['AA','AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH'
, 'IY', 'OW', 'OY', 'UH', 'UW']
consonants= ['P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M',
'N', 'NG', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
vector_map = vowels + consonants

def read_data(file_path):
	with open(file_path) as f:
		lines = f.read().splitlines()
	return lines

# Maps the location of the stress, 1 if stress at position
# 0 otherwise
def stress_map(pronunciation,stress='1'):
	return [1 if stress in num else 0 for num in pronunciation]

# Maps the the location of phenom, 1 in phenom_list
# 0 otherwise
def phenom_map(pronunciation,phenom_list):
	return [1 if phenom in phenom_list else 0 for phenom in pronunciation]


class word(object):
	"""docstring for word"""
	def __init__(self, word_string):
		super(word, self).__init__()
		self.word = word_string.split(':')[0]
		self.pronunciation = word_string.split(':')[1]
		self.pn_list = self.pronunciation.split()
		self.primary_stress_map = stress_map(self.pn_list)
		self.secondary_stress_map = stress_map(self.pn_list, stress='2')
		self.vowel_map = phenom_map(self.pn_list,vowels)
		self.consonant_map = phenom_map(self.pn_list,consonants)
		self.vector_map = phenom_map(vector_map,filter(str.isalpha,self.pn_list))

for line in read_data('asset/training_data.txt'):
	x = word(line)
	print(x.word)
	print(x.pronunciation)
	print(x.vector_map)
