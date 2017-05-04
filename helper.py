import nltk
import pandas as pd
import pandas2arff as pd2a
import string

# Phonemes
vowels= ('AA','AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH'
, 'IY', 'OW', 'OY', 'UH', 'UW')

consonants= ('P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M','N',
 				'NG', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH')
suffixes=('inal','tion','sion','osis','oon','sce','que','ette','eer','ee','aire','able','ible','acy','cy','ade','age','al','al','ial','ical','an','ance','ence',
		'ancy','ency','ant','ent','ant','ent','ient','ar','ary','ard','art','ate','ate','ate','ation','cade','drome','ed','ed','en','en','ence','ency','er','ier',
		'er','or','er','or','ery','es','ese','ies','es','ies','ess','est','iest','fold','ful','ful','fy','ia','ian','iatry','ic','ic','ice','ify','ile',
		'ing','ion','ish','ism','ist','ite','ity','ive','ive','ative','itive','ize','less','ly','ment','ness','or','ory','ous','eous','ose','ious','ship','ster',
		'ure','ward','wise','ize','phy','ogy')

prefixes=('ac','ad','af','ag','al','an','ap','as','at','an','ab','abs','acer','acid','acri','act','ag','acu','aer','aero','ag','agi',
			'ig','act','agri','agro','alb','albo','ali','allo','alter','alt','am','ami','amor','ambi','ambul','ana','ano','andr','andro','ang',
			'anim','ann','annu','enni','ante','anthrop','anti','ant','anti','antico','apo','ap','aph','aqu','arch','aster','astr','auc','aug',
			'aut','aud','audi','aur','aus','aug','auc','aut','auto','bar','be','belli','bene','bi','bine','bibl','bibli','biblio','bio','bi',
			'brev','cad','cap','cas','ceiv','cept','capt','cid','cip','cad','cas','calor','capit','capt','carn','cat','cata','cath','caus','caut'
			,'cause','cuse','cus','ceas','ced','cede','ceed','cess','cent','centr','centri','chrom','chron','cide','cis','cise','circum','cit',
			'civ','clam','claim','clin','clud','clus claus','co','cog','col','coll','con','com','cor','cogn','gnos','com','con','contr','contra',
			'counter','cord','cor','cardi','corp','cort','cosm','cour','cur','curr','curs','crat','cracy','cre','cresc','cret','crease','crea',
			'cred','cresc','cret','crease','cru','crit','cur','curs','cura','cycl','cyclo','de','dec','deca','dec','dign','dei','div','dem','demo',
			'dent','dont','derm','di','dy','dia','dic','dict','dit','dis','dif','dit','doc','doct','domin','don','dorm','dox','duc','duct','dura',
			'dynam','dys','ec','eco','ecto','en','em','end','epi','equi','erg','ev','et','ex','exter','extra','extro','fa','fess','fac','fact',
			'fec','fect','fic','fas','fea','fall','fals','femto','fer','fic','feign','fain','fit','feat','fid','fid','fide','feder','fig','fila',
			'fili','fin','fix','flex','flect','flict','flu','fluc','fluv','flux','for','fore','forc','fort','form','fract','frag',
			'frai','fuge','fuse','gam','gastr','gastro','gen','gen','geo','germ','gest','giga','gin','gloss','glot','glu','glo','gor','grad','gress',
			'gree','graph','gram','graf','grat','grav','greg','hale','heal','helio','hema','hemo','her','here','hes','hetero','hex','ses','sex','homo',
			'hum','human','hydr','hydra','hydro','hyper','hypn','an','ics','ignis','in','im','in','im','il','ir','infra','inter','intra','intro','ty',
			'jac','ject','join','junct','judice','jug','junct','just','juven','labor','lau','lav','lot','lut','lect','leg','lig','leg','levi','lex',
			'leag','leg','liber','liver','lide','liter','loc','loco','log','logo','ology','loqu','locut','luc','lum','lun','lus','lust','lude','macr',
			'macer','magn','main','mal','man','manu','mand','mania','mar','mari','mer','matri','medi','mega','mem','ment','meso','meta','meter','metr',
			'micro','migra','mill','kilo','milli','min','mis','mit','miss','mob','mov','mot','mon','mono','mor','mort','morph','multi','nano','nasc',
			'nat','gnant','nai','nat','nasc','neo','neur','nom','nom','nym','nomen','nomin','non','non','nov','nox','noc','numer','numisma','ob','oc',
			'of','op','oct','oligo','omni','onym','oper','ortho','over','pac','pair','pare','paleo','pan','para','pat','pass','path','pater','patr',
			'path','pathy','ped','pod','pedo','pel','puls','pend','pens','pond','per','peri','phage','phan','phas','phen','fan','phant','fant','phe',
			'phil','phlegma','phobia','phobos','phon','phot','photo','pico','pict','plac','plais','pli','ply','plore','plu','plur','plus','pneuma',
			'pneumon','pod','poli','poly','pon','pos','pound','pop','port','portion','post','pot','pre','pur','prehendere','prin','prim','prime',
			'pro','proto','psych','punct','pute','quat','quad','quint','penta','quip','quir','quis','quest','quer','re','reg','recti','retro','ri','ridi',
			'risi','rog','roga','rupt','sacr','sanc','secr','salv','salu','sanct','sat','satis','sci','scio','scientia','scope','scrib','script','se',
			'sect','sec','sed','sess','sid','semi','sen','scen','sent','sens','sept','sequ','secu','sue','serv','sign','signi','simil','simul','sist','sta',
			'stit','soci','sol','solus','solv','solu','solut','somn','soph','spec','spect','spi','spic','sper','sphere','spir','stand','stant','stab',
			'stat','stan','sti','sta','st','stead','strain','strict','string','stige','stru','struct','stroy','stry','sub','suc','suf','sup','sur','sus',
			'sume','sump','super','supra','syn','sym','tact','tang','tag','tig','ting','tain','ten','tent','tin','tect','teg','tele','tem','tempo','ten',
			'tin','tain','tend','tent','tens','tera','term','terr','terra','test','the','theo','therm','thesis','thet','tire','tom','tor','tors','tort'
			,'tox','tract','tra','trai','treat','trans','tri','trib','tribute','turbo','typ','ultima','umber','umbraticum','un','uni','vac','vade','vale',
			'vali','valu','veh','vect','ven','vent','ver','veri','verb','verv','vert','vers','vi','vic','vicis','vict','vinc','vid','vis','viv','vita','vivi'
			,'voc','voke','vol','volcan','volv','volt','vol','vor','with','zo')

suffixes_set = {suffix.upper() for suffix in suffixes}
prefixes_set = {prefix.upper() for prefix in prefixes}

vector_map = vowels + consonants

def read_data(file_path):
	with open(file_path) as f:
		lines = f.read().splitlines()
	return lines

# Filter numbers from string
def filter_stress(string):
	return ''.join([i for i in string if not i.isdigit()]).split()

# Maps the location of the stress, 1 if stress at position
# 0 otherwise
def stress_map(pronunciation,stress='1'):
	return [1 if stress in num else 0 for num in pronunciation]

# Maps the the location of phenom, 1 in phenom_list
# 0 otherwise
# When being used to map phenoms to vector_map filters stresses filters stresses
# first, this step is superfluous for for vowel and consonant map
def phenom_map(pronunciation, phenom_list=None):
	return [1 if phenom in phenom_list else 0 for phenom in pronunciation]

def get_pos_tag(word):
	return nltk.pos_tag([word])[0][1]

def get_stress_position(stress_map,stress=1):
	return str(stress_map.index(stress) + 1)

# Check if prefix exists
def check_prefix(word):
	for letter_idx in range(len(word) + 1):
		if word[:letter_idx] in prefixes_set:
			return 1
	return 0

# Check if suffix exists
def check_suffix(word):
	word_length = len(word)
	for letter_idx in range(word_length + 1):
		if word[abs(letter_idx-word_length) :] in suffixes_set:
			return 1
	return 0

'''
Object to hold each word
word : Word
pronunciation: String of phonemes
pn_list: List of pronunciation phonemes
primary_stress_map: binary vecotr with position of primary stress
primary_stress_idx: Index of primary stress
secondary_stress_map: binary vector with position of secondary stress
vowel_map: binary vector with position of vowels
consonant_map: binary vector with position of consonants
vector_map: Binary vector for vowel and constant existance
type_tag: Pos_Tag for the word from nltk
first_letter_index: Alphabetic index of first letter
phenom_length: Number of phonemes
prefix: 1 if prefix exists 0 otherwise
suffix: 1 if suffix exists 0 otherwise
'''
class word(object):
	"""docstring for word"""
	def __init__(self, word_string):
		super(word, self).__init__()
		self.word = word_string.split(':')[0]
		self.pronunciation = word_string.split(':')[1]
		self.pn_list = self.pronunciation.split()
		self.primary_stress_map = stress_map(self.pn_list)
		self.primary_stress_idx = get_stress_position(self.primary_stress_map)
		self.secondary_stress_map = stress_map(self.pn_list, stress='2')
		self.vowel_map = phenom_map(self.pn_list,vowels)
		self.consonant_map = phenom_map(self.pn_list,consonants)
		self.vector_map = phenom_map(vector_map,self.pn_list)
		self.type_tag = get_pos_tag(self.word)
		self.first_letter_index = string.ascii_lowercase.index(self.word[0].lower()) + 1
		self.phoneme_length = len(self.pn_list)
		self.prefix = check_prefix(self.word)
		self.suffix = check_suffix(self.word)

def get_words(file_path):
	words = pd.read_csv(file_path,sep=':',names=['word','pronunciation'])
	words['pn_list'] = words.pronunciation.apply(str.split)
	words['destressed_pn_list'] = words.pronunciation.apply(filter_stress)
	words['primary_stress_map'] = words.pn_list.apply(stress_map)
	words['secondary_stress_map'] = words.pn_list.apply(stress_map,stress='2')
	words['primary_stress_idx'] = words.primary_stress_map.apply(get_stress_position)
	words['vowel_map'] = words.destressed_pn_list.apply(phenom_map,vowels)
	words['consonant_map'] = words.destressed_pn_list.apply(phenom_map, consonants)

	return words

if __name__ == '__main__':
	data_loc = 'asset/training_data.txt'
	words = get_words(data_loc)
	words.head()

	words_df = pd.DataFrame([word.__dict__ for word in words])
	columns = ['word','type_tag','first_letter_index','phoneme_length','suffix','prefix','primary_stress_idx']
	pd2a.pandas2arff(words_df[columns],'weka/word_typetag.arff','word_typetag')
