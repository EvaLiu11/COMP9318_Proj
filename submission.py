## import modules here
import pandas as pd
import numpy as np
import string
from collections import deque,Counter
from sklearn.utils.extmath import cartesian
import re
import nltk
import pickle

from sklearn.neighbors import KNeighborsClassifier

################# helper data ##############

# Vowel Phonemes
vowels = ('AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH'
          , 'IY', 'OW', 'OY', 'UH', 'UW')

# Consonants Phonemes
consonants = ('P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N',
              'NG', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH')

suffixes = (
'inal', 'tion', 'sion', 'osis', 'oon', 'sce', 'que', 'ette', 'eer', 'ee', 'aire', 'able', 'ible', 'acy', 'cy', 'ade',
'age', 'al', 'al', 'ial', 'ical', 'an', 'ance', 'ence',
'ancy', 'ency', 'ant', 'ent', 'ant', 'ent', 'ient', 'ar', 'ary', 'ard', 'art', 'ate', 'ate', 'ate', 'ation', 'cade',
'drome', 'ed', 'ed', 'en', 'en', 'ence', 'ency', 'er', 'ier',
'er', 'or', 'er', 'or', 'ery', 'es', 'ese', 'ies', 'es', 'ies', 'ess', 'est', 'iest', 'fold', 'ful', 'ful', 'fy', 'ia',
'ian', 'iatry', 'ic', 'ic', 'ice', 'ify', 'ile',
'ing', 'ion', 'ish', 'ism', 'ist', 'ite', 'ity', 'ive', 'ive', 'ative', 'itive', 'ize', 'less', 'ly', 'ment', 'ness',
'or', 'ory', 'ous', 'eous', 'ose', 'ious', 'ship', 'ster',
'ure', 'ward', 'wise', 'ize', 'phy', 'ogy')

prefixes = (
'ac', 'ad', 'af', 'ag', 'al', 'an', 'ap', 'as', 'at', 'an', 'ab', 'abs', 'acer', 'acid', 'acri', 'act', 'ag', 'acu',
'aer', 'aero', 'ag', 'agi',
'ig', 'act', 'agri', 'agro', 'alb', 'albo', 'ali', 'allo', 'alter', 'alt', 'am', 'ami', 'amor', 'ambi', 'ambul', 'ana',
'ano', 'andr', 'andro', 'ang',
'anim', 'ann', 'annu', 'enni', 'ante', 'anthrop', 'anti', 'ant', 'anti', 'antico', 'apo', 'ap', 'aph', 'aqu', 'arch',
'aster', 'astr', 'auc', 'aug',
'aut', 'aud', 'audi', 'aur', 'aus', 'aug', 'auc', 'aut', 'auto', 'bar', 'be', 'belli', 'bene', 'bi', 'bine', 'bibl',
'bibli', 'biblio', 'bio', 'bi',
'brev', 'cad', 'cap', 'cas', 'ceiv', 'cept', 'capt', 'cid', 'cip', 'cad', 'cas', 'calor', 'capit', 'capt', 'carn',
'cat', 'cata', 'cath', 'caus', 'caut'
, 'cause', 'cuse', 'cus', 'ceas', 'ced', 'cede', 'ceed', 'cess', 'cent', 'centr', 'centri', 'chrom', 'chron', 'cide',
'cis', 'cise', 'circum', 'cit',
'civ', 'clam', 'claim', 'clin', 'clud', 'clus claus', 'co', 'cog', 'col', 'coll', 'con', 'com', 'cor', 'cogn', 'gnos',
'com', 'con', 'contr', 'contra',
'counter', 'cord', 'cor', 'cardi', 'corp', 'cort', 'cosm', 'cour', 'cur', 'curr', 'curs', 'crat', 'cracy', 'cre',
'cresc', 'cret', 'crease', 'crea',
'cred', 'cresc', 'cret', 'crease', 'cru', 'crit', 'cur', 'curs', 'cura', 'cycl', 'cyclo', 'de', 'dec', 'deca', 'dec',
'dign', 'dei', 'div', 'dem', 'demo',
'dent', 'dont', 'derm', 'di', 'dy', 'dia', 'dic', 'dict', 'dit', 'dis', 'dif', 'dit', 'doc', 'doct', 'domin', 'don',
'dorm', 'dox', 'duc', 'duct', 'dura',
'dynam', 'dys', 'ec', 'eco', 'ecto', 'en', 'em', 'end', 'epi', 'equi', 'erg', 'ev', 'et', 'ex', 'exter', 'extra',
'extro', 'fa', 'fess', 'fac', 'fact',
'fec', 'fect', 'fic', 'fas', 'fea', 'fall', 'fals', 'femto', 'fer', 'fic', 'feign', 'fain', 'fit', 'feat', 'fid', 'fid',
'fide', 'feder', 'fig', 'fila',
'fili', 'fin', 'fix', 'flex', 'flect', 'flict', 'flu', 'fluc', 'fluv', 'flux', 'for', 'fore', 'forc', 'fort', 'form',
'fract', 'frag',
'frai', 'fuge', 'fuse', 'gam', 'gastr', 'gastro', 'gen', 'gen', 'geo', 'germ', 'gest', 'giga', 'gin', 'gloss', 'glot',
'glu', 'glo', 'gor', 'grad', 'gress',
'gree', 'graph', 'gram', 'graf', 'grat', 'grav', 'greg', 'hale', 'heal', 'helio', 'hema', 'hemo', 'her', 'here', 'hes',
'hetero', 'hex', 'ses', 'sex', 'homo',
'hum', 'human', 'hydr', 'hydra', 'hydro', 'hyper', 'hypn', 'an', 'ics', 'ignis', 'in', 'im', 'in', 'im', 'il', 'ir',
'infra', 'inter', 'intra', 'intro', 'ty',
'jac', 'ject', 'join', 'junct', 'judice', 'jug', 'junct', 'just', 'juven', 'labor', 'lau', 'lav', 'lot', 'lut', 'lect',
'leg', 'lig', 'leg', 'levi', 'lex',
'leag', 'leg', 'liber', 'liver', 'lide', 'liter', 'loc', 'loco', 'log', 'logo', 'ology', 'loqu', 'locut', 'luc', 'lum',
'lun', 'lus', 'lust', 'lude', 'macr',
'macer', 'magn', 'main', 'mal', 'man', 'manu', 'mand', 'mania', 'mar', 'mari', 'mer', 'matri', 'medi', 'mega', 'mem',
'ment', 'meso', 'meta', 'meter', 'metr',
'micro', 'migra', 'mill', 'kilo', 'milli', 'min', 'mis', 'mit', 'miss', 'mob', 'mov', 'mot', 'mon', 'mono', 'mor',
'mort', 'morph', 'multi', 'nano', 'nasc',
'nat', 'gnant', 'nai', 'nat', 'nasc', 'neo', 'neur', 'nom', 'nom', 'nym', 'nomen', 'nomin', 'non', 'non', 'nov', 'nox',
'noc', 'numer', 'numisma', 'ob', 'oc',
'of', 'op', 'oct', 'oligo', 'omni', 'onym', 'oper', 'ortho', 'over', 'pac', 'pair', 'pare', 'paleo', 'pan', 'para',
'pat', 'pass', 'path', 'pater', 'patr',
'path', 'pathy', 'ped', 'pod', 'pedo', 'pel', 'puls', 'pend', 'pens', 'pond', 'per', 'peri', 'phage', 'phan', 'phas',
'phen', 'fan', 'phant', 'fant', 'phe',
'phil', 'phlegma', 'phobia', 'phobos', 'phon', 'phot', 'photo', 'pico', 'pict', 'plac', 'plais', 'pli', 'ply', 'plore',
'plu', 'plur', 'plus', 'pneuma',
'pneumon', 'pod', 'poli', 'poly', 'pon', 'pos', 'pound', 'pop', 'port', 'portion', 'post', 'pot', 'pre', 'pur',
'prehendere', 'prin', 'prim', 'prime',
'pro', 'proto', 'psych', 'punct', 'pute', 'quat', 'quad', 'quint', 'penta', 'quip', 'quir', 'quis', 'quest', 'quer',
're', 'reg', 'recti', 'retro', 'ri', 'ridi',
'risi', 'rog', 'roga', 'rupt', 'sacr', 'sanc', 'secr', 'salv', 'salu', 'sanct', 'sat', 'satis', 'sci', 'scio',
'scientia', 'scope', 'scrib', 'script', 'se',
'sect', 'sec', 'sed', 'sess', 'sid', 'semi', 'sen', 'scen', 'sent', 'sens', 'sept', 'sequ', 'secu', 'sue', 'serv',
'sign', 'signi', 'simil', 'simul', 'sist', 'sta',
'stit', 'soci', 'sol', 'solus', 'solv', 'solu', 'solut', 'somn', 'soph', 'spec', 'spect', 'spi', 'spic', 'sper',
'sphere', 'spir', 'stand', 'stant', 'stab',
'stat', 'stan', 'sti', 'sta', 'st', 'stead', 'strain', 'strict', 'string', 'stige', 'stru', 'struct', 'stroy', 'stry',
'sub', 'suc', 'suf', 'sup', 'sur', 'sus',
'sume', 'sump', 'super', 'supra', 'syn', 'sym', 'tact', 'tang', 'tag', 'tig', 'ting', 'tain', 'ten', 'tent', 'tin',
'tect', 'teg', 'tele', 'tem', 'tempo', 'ten',
'tin', 'tain', 'tend', 'tent', 'tens', 'tera', 'term', 'terr', 'terra', 'test', 'the', 'theo', 'therm', 'thesis',
'thet', 'tire', 'tom', 'tor', 'tors', 'tort'
, 'tox', 'tract', 'tra', 'trai', 'treat', 'trans', 'tri', 'trib', 'tribute', 'turbo', 'typ', 'ultima', 'umber',
'umbraticum', 'un', 'uni', 'vac', 'vade', 'vale',
'vali', 'valu', 'veh', 'vect', 'ven', 'vent', 'ver', 'veri', 'verb', 'verv', 'vert', 'vers', 'vi', 'vic', 'vicis',
'vict', 'vinc', 'vid', 'vis', 'viv', 'vita', 'vivi'
, 'voc', 'voke', 'vol', 'volcan', 'volv', 'volt', 'vol', 'vor', 'with', 'zo')

suffixes_set = {suffix.upper() for suffix in suffixes}
prefixes_set = {prefix.upper() for prefix in prefixes}
vector_map = vowels + consonants

################# helper functions #########


def read_data(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines


# Convert list to tuple
def as_tuple(list_to_convert):
    return tuple(list_to_convert)


# Filter stress from string

def filter_stress(string_to_be_filtered, to_filter=None):
    if type(string_to_be_filtered) in [list, tuple]:
        string_to_be_filtered = ' '.join(string_to_be_filtered)
    return tuple(re.sub(to_filter,'',string_to_be_filtered).split())


# Filter non-important stresses
def filter_non_primary_stress(pronunciation):
    pronunciation = pronunciation.replace('0', '')
    return pronunciation.replace('2', '')


# Maps the location of the stress, 1 if stress at position
# 0 otherwise
def stress_map(pronunciation, stress='1'):
    return [1 if stress in num else 0 for num in pronunciation]


# Maps the the location of phenom, 1 in phenom_list
# 0 otherwise
def phoneme_map(pronunciation, phoneme_list):
    return [1 if phoneme in phoneme_list else 0 for phoneme in pronunciation]


# Map existence of one iterable in another
def iterable_map(list_to_map, iterable):
    return [1 if iter_item in list_to_map else 0 for iter_item in iterable]


# Get nltk pos_tag
def get_pos_tag(word):
    return nltk.pos_tag([word])[0][1]


# Returning string as a classification
def get_stress_position(stress_map_list, stress=1):
    return str(stress_map_list.index(stress) + 1)


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
        if word[abs(letter_idx - word_length):] in suffixes_set:
            return 1
    return 0


# Get ascii index of first letter
def get_first_letter_idx(word):
    return string.ascii_lowercase.index(word[0].lower()) + 1


# Return the stressed vowel
def get_stressed_vowel(pn_list):
    for vowel in pn_list:
        if '1' in vowel:
            return filter_stress(vowel,to_filter='1')[0]


# Return all possible consecutive tuples length n from list
def sub_string(pronunciation_list, n):
    return tuple(zip(*(pronunciation_list[i:] for i in range(n))))


# Build a dict of all possible sequences of phonemes
def get_sequences(phoneme_series):
    sequences = {}
    max_length = max(phoneme_series.str.len())
    for i in range(2, max_length + 1):
        for pn_list in phoneme_series:
            # Next iteration if pn_list is shorter then the sequence length be built
            if len(pn_list) < i:
                continue
            word_sequences = sub_string(pn_list, i)
            for seq in word_sequences:
                sequences[seq] = sequences.get(seq, 0) + 1
    return sequences


def in_list(pn_list, sequence):
    if pn_list in sequence:
        return 1
    return 0


# Return 1 if sequence has a primary stress in it
def is_primary(sequence):
    for phoneme in sequence:
        if '1' in phoneme:
            return True
    return False


def line_split(line):
    line = line.split(':')
    return line[0], line[1]


'''
Dataframe to hold list of words
word : Word
pronunciation: String of phonemes
pn_list: List of pronunciation phonemes
primary_stress_map: binary vector with position of primary stress
primary_stress_idx: Index of primary stress
secondary_stress_map: binary vector with position of secondary stress
vowel_map: binary vector with position of vowels
consonant_map: binary vector with position of consonants
vector_map: Binary vector for vowel and constant existence
type_tag: Pos_Tag for the word from nltk
first_letter_index: Alphabetic index of first letter
phenom_length: Number of phonemes
prefix: 1 if prefix exists 0 otherwise
suffix: 1 if suffix exists 0 otherwise
'''


def get_words(datafile):
    lines = [line_split(line) for line in datafile]
    words = pd.DataFrame(data=lines, columns=('word', 'pronunciation'))
    words['pn_list'] = words.pronunciation.apply(str.split)
    words['destressed_pn_list'] = words.pronunciation.apply(filter_stress, args=('[012]',))
    words['primary_stress_map'] = words.pn_list.apply(stress_map)
    words['secondary_stress_map'] = words.pn_list.apply(stress_map, stress='2')
    words['vowel_map'] = words.destressed_pn_list.apply(phoneme_map, args=(vowels,))
    words['consonant_map'] = words.destressed_pn_list.apply(phoneme_map, args=(consonants,))
    words['vector_map'] = words.destressed_pn_list.apply(iterable_map, args=(vector_map,))
    words['vowel_count'] = words.vowel_map.apply(np.sum)
    words['consonant_count'] = words.consonant_map.apply(np.sum)
    # words['type_tag'] = words.word.apply(get_pos_tag)
    words['1st_letter_idx'] = words.word.apply(get_first_letter_idx)
    words['phoneme_length'] = words.pn_list.str.len()
    # words['prefix'] = words.word.apply(check_prefix)
    # words['suffix'] = words.word.apply(check_suffix)
    # words['prefix_suffix_vector'] = words.
    # words['primary_stress_idx'] = words.primary_stress_map.apply(get_stress_position)
    words['stressed_vowel'] = words.pn_list.apply(get_stressed_vowel)
    words['ngrams'] = words.pn_list.apply(get_all_ngrams)

    # Unpack vector map into single columns
    # unpacked_vector_map = pd.DataFrame.from_records(words.vector_map.tolist(),columns=vector_map)
    # words = pd.concat([words, unpacked_vector_map],axis=1)
    return words


################# ngrams ###################

# Return all ngrams of particular length
def get_ngram_possibilities(pronunciation_list, length):
    return tuple(zip(*(pronunciation_list[i:] for i in range(length))))


# Develop deque of all possible ngrams
def get_all_ngrams(pn_list):
    ngrams = deque()
    for i in range(2,len(pn_list) + 1):
        ngrams.extend(get_ngram_possibilities(pn_list,i))
    return ngrams

# Check if ngram in list
def in_list(pn_list, ngram):
    if pn_list in ngram:
        return 1
    return 0


# Check if ngram has primary stress
def is_primary(ngram):
    for phoneme in ngram:
        if '1' in phoneme:
            return True
    return False


# Check if there is a smaller ngram in set
def has_ngram(ngram, ngram_set):
    # Do not check sequences of length 2 or the final as they will obviously be in the set
    for i in range(2, len(ngram)):
        subsequence = ngram[0:i]
        if subsequence in ngram_set:
            return True
    return False


# Return true if ngram in family
def in_family(family, ngram):
    return family == ngram[0:len(family)]

# Apply function over index
def apply_index(row, func):
    return func(row.name)

# Use string matching to check if ngram in ngram_base
def has_subngram(ngram, ngram_base):
    ngram = str(ngram).strip('()')
    ngram_base = str(ngram_base).strip('()')
    return ngram in ngram_base


# Add series to data frame which include the smallest ngram within a larger ngram
def collapse_ngrams(ngram_lists):
    ngrams = deque()
    # Pack all possibilities into deque
    for ngram_list in ngram_lists:
        ngrams.extend(ngram_list)

    ngrams_df = pd.DataFrame.from_dict(Counter(ngrams),orient='index')
    ngrams_df = ngrams_df.rename(columns={0:'ngram_counts'})
    ngrams_df['is_primary'] = ngrams_df.index.map(is_primary)
    return ngrams_df.query('is_primary == True').sort_values(by='ngram_counts', ascending=False)

# Return dict key = ngram families, count = Total times ngram is developed for set of words
def get_ngram_counts(ngram_families,possible_ngrams):
    ngram_counts = {family:0 for family in ngram_families}
    for possible in possible_ngrams:
        if possible in ngram_counts:
            ngram_counts[possible] += 1
    return ngram_counts

################# training #################

def train(data, classifier_file):  # do not change the heading of the function
    words = get_words(data)
    return words


################# testing #################

def test(data, classifier_file):  # do not change the heading of the function
    pass  # **replace** this line with your code
