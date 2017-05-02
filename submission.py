## import modules here
import helper as h
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier



################# training #################

def train(data, classifier_file):# do not change the heading of the function
    pass # **replace** this line with your code

################# testing #################

def test(data, classifier_file):# do not change the heading of the function
    pass # **replace** this line with your code

if __name__ == '__main__':
    data_loc = 'asset/training_data.txt'
    words = h.get_words(data_loc)

    word_vectors = [word.vector_map for word in words]
    stress_pos = [word.primary_stress_map.index(1) for word in words]
    df = pd.DataFrame(word_vectors,columns=h.vector_map)
    #df['Stress_Pos'] = stress_pos

    neigh = KNeighborsClassifier(n_neighbors=3,weights='uniform',p=2)
    neigh.fit(df, stress_pos)
    print(neigh.score(df, stress_pos))
