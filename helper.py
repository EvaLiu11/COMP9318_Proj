# Phenoms
vowels= ['AA','AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH'
, 'IY', 'OW', 'OY', 'UH', 'UW']
consonants= ['P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 
'N', 'NG', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']


def read_data(file_path):
	with open(file_path) as f:
		lines = f.read().splitlines()
	return lines



