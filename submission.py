## import modules here
import pandas as pd
import numpy as np
import string
from collections import deque,Counter
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score,classification_report
import re
import nltk
import pickle


################# training #################

def train(data, classifier_file):  # do not change the heading of the function
    words = word_data(data)
    mb_clf = classifier(MultinomialNB)
    
    train_X = words.df.ngram_counts
    train_Y = words.df.classification
    
    mb_clf.train(train_X, train_Y)
    save_Pickle(mb_clf,classifier_file)    
    
    return


################# testing #################

def test(data, classifier_file,sample=None,DEBUG=None):  # do not change the heading of the function
    clf = get_Pickle(classifier_file)
    test_words = word_data(data)
    
    if sample:
        test_words.df = test_words.df.sample(sample)
    
    test_words.set_predicted_classes(clf.predict_classifications(test_words.df.ngram_counts))
    pred = test_words.df.predicted_primary_index.tolist()
    
    if DEBUG:
        print(classification_report(test_words.df.primary_stress_index,pred))
        print(f1_score(test_words.df.primary_stress_index, pred, average='macro'))
        
    
    return pred
    
    

################# helper data ##############

# Vowel Phonemes
vowels = ('AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH'
          , 'IY', 'OW', 'OY', 'UH', 'UW')

# Consonants Phonemes
consonants = ('P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N',
              'NG', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH')


# Classification Map
classifications = { '10'  : 0,
                    '100' : 0,
                    '1000': 0,
                    '01'  : 3,
                    '001' : 3,
                    '0001': 3,
                    '010' : 1,
                    '0100': 1,
                    '0010': 2
                    }

vector_map = vowels + consonants

################# classes ##################

'''

word_data       = Class to hold word data and perform all requisite pre-processing

    Attributes
lines           = List of word and stressed phonemes
df              = dataframe to hold and process word data
pn_list         = list of phonemes
vowel_map       = 2-4 bit string depicting location of primary stress
classifications = Group Index of stressed vowel, 0 is 1st, 3 is last irrespective of vowel count/word length
                  1 and 2 are then 2nd and 3rd respecively.
ngrams          = All possible ngrams of pn_list
ngrams_counts   = Dict object of ngrams 


'''
class word_data(object):
    def __init__(self,data):
        self.lines = [line_split(line) for line in data]
        self.df = pd.DataFrame(data=self.lines, columns=('word', 'pronunciation'))
        self.df['pn_list'] = self.df.pronunciation.apply(str.split)
        self.df['destressed_pn_list'] = self.df.pronunciation.apply(filter_stress, args=('[012]',))
        self.df['vowel_map'] = self.df.destressed_pn_list.apply(phoneme_map, args=(vowels,))
        self.df['vowel_map_string'] = self.df.vowel_map.apply(to_string)
        self.df['stress_map'] = self.df.pn_list.apply(get_stress_map)
        self.df['classification'] = self.df.stress_map.apply(get_classification)
        self.df['primary_stress_index'] = self.df.apply(get_classsification_index,args=('classification',) ,axis=1)
        self.df['ngrams'] = self.df.pn_list.apply(get_all_ngrams)
        self.df['ngram_counts'] = self.df.ngrams.apply(Counter)
        self.df['destressed_ngrams'] = self.df.destressed_pn_list.apply(get_all_ngrams)
        self.df['destressed_ngram_count'] = self.df.destressed_ngrams.apply(Counter)
        
    
    def set_predicted_classes(self,classes):
        self.df['predicted_classes'] = classes
        self.df['predicted_primary_index'] = self.df.apply(get_classsification_index,args=('predicted_classes',), axis=1)
        
    
'''
classifier      = Class to hold classifier and training/testing/prediction methods

    Attributes
clf             = Passed in Classifier
encoder         = LabelEncoder for classes
vectorizer      = DictVectorizer (Sparse Matrix) to hold features
train_X         = Vectorized training features
train_Y         = Label Encoded training classifications
test_X          = 


    Methods
train           = Encode Features and Classifications, Train Classifier
test

'''

class classifier(object):
    def __init__(self,classifier,*args,**kwargs):
        self.clf = classifier()
        self.encoder = LabelEncoder()
        self.vectorizer = DictVectorizer(dtype=int, sparse=True)
    
    def train(self,X,Y):
        self.train_X = self.vectorizer.fit_transform(X.tolist())
        self.train_Y = self.encoder.fit_transform(Y)
        self.clf.fit(self.train_X,self.train_Y)

    
    def _encode_test_features(self,X):
        return self.vectorizer.transform(X.tolist())

    
    def predict_classifications(self,X):
        predicted_Y = self.clf.predict(self._encode_test_features(X))
        return predicted_Y
    
        
        
################# helper functions #########

# Pickler
def save_Pickle(obj, file):
    with open(file,'wb') as f:
        pickle.dump(obj,f)
    f.close()
    
def get_Pickle(file):
    with open(file,'rb') as f:
        obj = pickle.load(f)
    f.close()
    return obj
    


# Return all ngrams of particular length
def get_ngram_possibilities(pronunciation_list, length):
    return tuple(zip(*(pronunciation_list[i:] for i in range(length))))


# Develop deque of all possible ngrams
def get_all_ngrams(pn_list,restrict_length = None):
    ngrams = set()
    if not restrict_length:
        restrict_length = len(pn_list)
    for i in range(2,restrict_length + 1):
        ngrams.update(get_ngram_possibilities(pn_list,i))
    return ngrams


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


# Return classification for pn_list
def get_stress_map(pn_list):
    vowels = str()
    for pn in pn_list:
        if pn in consonants:
            continue
        elif '1' in pn:
            vowels += '1'
        elif '0' in pn or '2' in pn:
            vowels += '0'
    return vowels

def get_classification(vowel_map):
    return classifications[vowel_map]

# Return the index of the stressed vowel based on classification
def get_classsification_index(df,classification_column):
    vowel_idx = [idx.start() for idx in re.finditer('1',df.vowel_map_string)]
    if df[classification_column] > len(vowel_idx) - 1:
        return vowel_idx[-1]
    if df[classification_column] < 3:
        return vowel_idx[df[classification_column]]
    else:
        return vowel_idx[-1]

def to_string(list_to_convert):
    return ''.join([str(x) for x in list_to_convert])


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
    
    words['pn_list'] = words.pronunciation.apply(str.split)
    words['destressed_pn_list'] = words.pronunciation.apply(filter_stress, args=('[012]',))
    words['primary_stress_map'] = words.pn_list.apply(stress_map)
    words['primary_stress_index'] = words.primary_stress_map.apply(list.index, args=(1,))
    words['secondary_stress_map'] = words.pn_list.apply(stress_map, stress='2')
    words['vowel_map'] = words.destressed_pn_list.apply(phoneme_map, args=(vowels,))
    words['vowel_map_string'] = words.vowel_map.apply(to_string)
    words['consonant_map'] = words.destressed_pn_list.apply(phoneme_map, args=(consonants,))
    words['vector_map'] = words.destressed_pn_list.apply(iterable_map, args=(vector_map,))
    words['vowel_count'] = words.vowel_map.apply(np.sum)
    words['classification'] = words.pn_list.apply(get_classification)
    words['consonant_count'] = words.consonant_map.apply(np.sum)
    words['primary_stress_index'] = words.primary_stress_map.apply(list.index, args=(1,))
    words['classification_index'] = words.apply(get_classsification_index, axis=1)
    words['secondary_stress_map'] = words.pn_list.apply(stress_map, stress='2')
    #words['type_tag'] = words.word.apply(get_pos_tag)
    words['1st_letter_idx'] = words.word.apply(get_first_letter_idx)
    words['phoneme_length'] = words.pn_list.str.len()
    # words['prefix'] = words.word.apply(check_prefix)
    # words['suffix'] = words.word.apply(check_suffix)
    # words['prefix_suffix_vector'] = words.
    # words['primary_stress_idx'] = words.primary_stress_map.apply(get_stress_position)
    words['stressed_vowel'] = words.pn_list.apply(get_stressed_vowel)
    words['ngrams'] = words.pn_list.apply(get_all_ngrams)
    words['ngram_counts'] = words.ngrams.apply(Counter)

    # Unpack vector map into single columns
    # unpacked_vector_map = pd.DataFrame.from_records(words.vector_map.tolist(),columns=vector_map)
    # words = pd.concat([words, unpacked_vector_map],axis=1)
    return words

