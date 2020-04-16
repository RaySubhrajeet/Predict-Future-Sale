#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:40:38 2020

@author: devendraswami
"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, concatenate
from keras.layers import Dense, Dropout
from keras import regularizers
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
import numpy as np
import pandas as pd
# import googletrans
import sys
import requests
import data_preprocess
from sklearn.model_selection import train_test_split
pd.set_option("display.max_columns", None)

NA_THRESHOLD = 9
PERIOD = 12
MIN_TRAIN = 12
MAX_TRAIN = 32
MIN_VALIDATION = 33
MAX_VALIDATION = 33
CLIP_TRAIN = 50
CLIP_VAL = 20
CLIP_TEST = 20
TEST_FILENAME = "submission_4.csv"

train = data_preprocess.train_data(PERIOD, MIN_TRAIN, MAX_TRAIN)    # label
val = data_preprocess.train_data(PERIOD, MIN_VALIDATION, MAX_VALIDATION)

# Drop rows with more than NA_THRESHOLD null values from training
train.dropna(thresh=train.shape[1]-NA_THRESHOLD, axis=0, inplace=True)
train.fillna(0, inplace=True)

# # Divide data into x and y
# # Split the data into training and validation sets
# train, val = train_test_split(train_df, test_size=0.2, random_state=101)


def build_x(df, clip):
    """
    Parameters
    ------------
    df: dataframe containing SHOP ITEM VIEW for each row
    clip: value for item_cnt_mnth to be max clipped at

    Returns
    ---------
    x: numpy array with size (samples,timesteps,features)
    const_cols: number of const_cols that are replictaed through timestamps
    """

    # Constant columns to be replictaed through timestamps
    cols = ['isMovie', 'isMusic', 'isBook', 'isGame', 'isGift', 'isAccessory', 'isProgram',
            'isPaymentCard', 'isService', 'isDelivery', 'isBatteries']
    const_cols = len(cols)
    for i in range(PERIOD):
        cols.append('item_cnt_mnth'+str(i))
        cols.append('item_price'+str(i))
    temp = df.filter(items=cols)

    x = temp.values
    stack_list = []
    for _ in range(PERIOD):
        stack_list.append(x)
    x = np.stack(stack_list, axis=1)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # Clipped item cnt/mnth
            x[i, j, const_cols] = min(x[i, j, const_cols+(2*j)], clip)
            x[i, j, const_cols+1] = x[i, j, (const_cols+1)+(2*j)]
    x = x[:, :, : const_cols+2]
    return x, const_cols


# Prepare data to ingest in LSTM (samples, timestep, features) and labels
y_train = train["label"].values.reshape(-1, 1)
y_train = np.clip(y_train, 0, CLIP_TRAIN)
y_val = val["label"].values.reshape(-1, 1)
y_val = np.clip(y_val, 0, CLIP_VAL)

x_train, const_cols = build_x(train, CLIP_TRAIN)
print("Shape of training data is", x_train.shape)
x_val, const_cols = build_x(val, CLIP_VAL)
print("Shape of validation data is", x_val.shape)


# NORMALIZATION (MU SIGMA)
mean_c = np.mean(x_train[:, :, const_cols])
mean_p = np.mean(x_train[:, :, const_cols+1])
std_c = np.std(x_train[:, :, const_cols])
std_p = np.std(x_train[:, :, const_cols+1])

x_train[:, :, const_cols] = (x_train[:, :, const_cols]-mean_c)/std_c
x_train[:, :, const_cols+1] = (x_train[:, :, const_cols+1]-mean_p)/std_p
x_val[:, :, const_cols] = (x_val[:, :, const_cols]-mean_c)/std_c
x_val[:, :, const_cols+1] = (x_val[:, :, const_cols+1]-mean_p)/std_p

y_train = (y_train-mean_c)/std_c
y_val = (y_val-mean_c)/std_c


# Model Development & Hyperparameter Tuning
main_input = Input(shape=(x_train.shape[1], x_train.shape[2]))
encoded_1 = LSTM(10, return_sequences=True,
                 activation='tanh', dropout=0.2)(main_input)
encoded_2 = LSTM(10, activation='tanh', dropout=0.1)(encoded_1)
output = Dense(1, activation='linear',
               kernel_regularizer=regularizers.l2(0.01))(encoded_2)
model = Model(inputs=[main_input], outputs=[output])
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit([x_train], [y_train],
                    validation_data=([x_val], [y_val]),
                    epochs=10, batch_size=512)

# plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()


# Prediction on validation set
yhat = model.predict(x_val)
inv_yhat = mean_c + (std_c*yhat)
inv_y = mean_c + (std_c*y_val)
print('The validation r squared value is: ', r2_score(inv_y, inv_yhat))

# Prediction on test data
test_df = data_preprocess.test_data(PERIOD, 33)      # ID
x_test, const_cols = build_x(test_df, CLIP_VAL)
x_test[:, :, const_cols] = (x_test[:, :, const_cols]-mean_c)/std_c
x_test[:, :, const_cols+1] = (x_test[:, :, const_cols+1]-mean_p)/std_p

ypred = model.predict(x_test)
inv_ypred = mean_c + (std_c*ypred)

test_df["item_cnt_month"] = np.clip(inv_ypred, 0, CLIP_TEST)
all_test_id = pd.read_csv("test.csv")
master_test = pd.merge(all_test_id, test_df, how="left", on="ID")

submit_test = master_test.filter(items=["ID", "item_cnt_month"])
submit_test.fillna(0, inplace=True)
submit_test.to_csv(TEST_FILENAME, sep=',', index=False, header=True)
