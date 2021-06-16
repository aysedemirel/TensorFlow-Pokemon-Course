import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

pokemon_data = pd.read_csv('pokemon_alopez247.csv')

#print(pokemon_data.columns)
pokemon_data = pokemon_data[['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style']]
#print(pokemon_data.columns)

# model should be numerical
pokemon_data['isLegendary'] = pokemon_data['isLegendary'].astype(int) # convert boolean to int
#print(pokemon_data.columns)

def dummy_creation(pokemon_data, dummy_categories):
    print(dummy_categories)
    for i in dummy_categories:
        df_dummy = pd.get_dummies(pokemon_data[i])
        print("------------")
        pokemon_data = pd.concat([pokemon_data,df_dummy],axis=1)
        print( pokemon_data)
        pokemon_data = pokemon_data.drop(i, axis=1)
        print( pokemon_data)
    return(pokemon_data)

pokemon_data = dummy_creation(pokemon_data, ['Egg_Group_1', 'Body_Style', 'Color','Type_1', 'Type_2'])
#print(pokemon_data.columns)

def train_test_splitter(DataFrame, column):
    df_train = DataFrame.loc[pokemon_data[column] != 1]
    df_test = DataFrame.loc[pokemon_data[column] == 1]

    df_train = df_train.drop(column, axis=1)
    df_test = df_test.drop(column, axis=1)

    return(df_train, df_test)

df_train, df_test = train_test_splitter(pokemon_data, 'Generation')


def label_delineator(df_train, df_test, label):
    
    train_data = df_train.drop(label, axis=1).values
    train_labels = df_train[label].values
    test_data = df_test.drop(label,axis=1).values
    test_labels = df_test[label].values
    return(train_data, train_labels, test_data, test_labels)

train_data, train_labels, test_data, test_labels = label_delineator(df_train, df_test, 'isLegendary')


def data_normalizer(train_data, test_data):
    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    return(train_data, test_data)

train_data, test_data = data_normalizer(train_data, test_data)

length = train_data.shape[1]

model = keras.Sequential()
model.add(keras.layers.Dense(500, activation='relu', input_shape=[length,]))
model.add(keras.layers.Dense(2, activation='softmax'))

#print(length)
