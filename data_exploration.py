# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 18:49:30 2018 by Devendra Swami
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.layers import Input, LSTM, RepeatVector, concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras import backend as K
import os
os.environ["PATH"] += os.pathsep + r'C:\ProgramData\Anaconda3\Lib\site-packages\graphviz-2.38\release\bin'

#______________________________________________________________________________
"""
SOME CUSTOM FUNCTIONS USED LATER
"""
# fn to reverse column names which doesn't start with months
def reverse_col_name(name):
    mnths = ['121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138']
    if(name.split('_')[0] in mnths): # Column start with month, don't reverse 
        return name
    else:
        return '_'.join(name.split('_')[::-1])

# Fn to compute mean, std of dataframe 
def mean_std(df):
    mean = df.values.mean()
    std = df.values.std()
    return mean,std

# Fn to drop a particular variable occurring somewhere in the column titles
def drop_columns(df,string):
    drop_cols = []
    for value in df.columns.values.tolist():
        if string.lower() in value.lower():
            drop_cols.append(value)
    return df.drop(columns=drop_cols)
#______________________________________________________________________________    
"""
Data Preparation Step
"""   
 
df= pd.read_csv(r'C:\Users\Axis_Inside\Music\burgundy_testing\data.csv') # READING DATA

values_fill={}   # Filling NA in all columns except MDAB/CDAB as 0
for key in df.columns[32:]:
    values_fill[key] = 0
df = df.fillna(value=values_fill)

df = df.dropna(axis=0, how='any') # Removing entries containing NA in MDAB/CDAB

# Drop dynamic and txn tags level variable 
num_dy_var = 10
num_txn_tags = 38
df = drop_columns(df,'CDAB')
num_dy_var = 9 # since CDAB is removed

# Segment filter based on MDAB sum for 1 year
df['z_mdab_sum_z'] = df['MDAB_121']+df['MDAB_122']+df['MDAB_123']+df['MDAB_124']+df['MDAB_125']+df['MDAB_126']+df['MDAB_127']+df['MDAB_128']+df['MDAB_129']+df['MDAB_130']+df['MDAB_131']+df['MDAB_132']
df['z_mdab_pred_sum_z'] = df['MDAB_133']+df['MDAB_134']+df['MDAB_135']
#df = df[(df['z_mdab_sum_z']<12000000) & (df['z_mdab_sum_z']>2400000)]    # 2-10 lakh MDAB
#df = df[(df['z_mdab_sum_z']>2400000)]
# Dictionary of modified column names so that they can be sorted by month/time step
column_names = {}
for key in df.columns:
    column_names[key]=reverse_col_name(key.upper())
df = df.rename(index=str, columns=column_names) # Replacing with modified column names
df = df [sorted(df.columns.tolist())]           # Sorting month-wise

## Outlier Removal
lower_bound = df['133_MDAB'].describe(percentiles=[0.05,0.95]).loc['5%']
upper_bound = df['133_MDAB'].describe(percentiles=[0.05,0.95]).loc['95%']
df = df[(df['133_MDAB']>lower_bound) & (df['133_MDAB']<upper_bound)]

# 48 dynamic variables for APR'17 to JULY'18 makes first 768 variables [i=767]  {[768]:'ACC_OF_NO', [769]:'AGE', [770]:'CONSTITUTION', [771]:'C_NAME_CAT_REGION',  [772]:'C_NAME_CENTRE',[773]:'C_NAME_CIRCLE',[774]:'C_PRODUCT',[775]:'C_SEGMENT', [776]:'C_STATUS_MARITAL',[777]:'GENDER',[778]:'ID',[779]:'MOB',[780]:'NEW_V2_CAT_OCCUPATION', [781]: 'Z_SUM_MDAB_Z', [782]: 'Z_SUM_PRED_MDAB_Z'} 
df = df.drop(columns=df.columns[(num_dy_var+num_txn_tags)*16:((num_dy_var+num_txn_tags)*16)+(2*num_txn_tags)]) # unwanted txn col for aug-sep'18
keep_columns = (df.columns[:(num_dy_var+num_txn_tags)*12].tolist()) + (df.columns[(num_dy_var+num_txn_tags)*16:].tolist())
keep_columns.append('133_MDAB')
keep_columns.append('134_MDAB')
keep_columns.append('135_MDAB')
drop_cols = list(set(df.columns.tolist()) - set(keep_columns))
df = df.drop(columns=drop_cols)

# CREATING TEST TRAIN SUBSETS
X = df
y = df['133_MDAB']
X_tr, X_te, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

# Outlier Removal (Training only, comment if outlier already removed above)
#lower_bound = X_tr['133_MDAB'].describe(percentiles=[0.05,0.95]).loc['5%']
#upper_bound = X_tr['133_MDAB'].describe(percentiles=[0.05,0.95]).loc['95%']
#y_train = y_train[(X_tr['133_MDAB']>lower_bound) & (X_tr['133_MDAB']<upper_bound)]
#X_tr = X_tr[(X_tr['133_MDAB']>lower_bound) & (X_tr['133_MDAB']<upper_bound)]
X_train = X_tr.drop(columns = ['C_NAME_CENTRE','C_PRODUCT','C_SEGMENT','ID','Z_SUM_MDAB_Z','Z_SUM_PRED_MDAB_Z','133_MDAB','134_MDAB','135_MDAB'])
X_test = X_te.drop(columns = ['C_NAME_CENTRE','C_PRODUCT', 'C_SEGMENT','ID','Z_SUM_MDAB_Z','Z_SUM_PRED_MDAB_Z','133_MDAB','134_MDAB','135_MDAB'])

# Separating static and dynamic variables
X_train_static = X_train[X_train.columns[(num_dy_var+num_txn_tags)*12:]]
X_train_dynamic = X_train[X_train.columns[:(num_dy_var+num_txn_tags)*12]]
X_test_static = X_test[X_test.columns[(num_dy_var+num_txn_tags)*12:]]
X_test_dynamic = X_test[X_test.columns[:(num_dy_var+num_txn_tags)*12]]
#______________________________________________________________________________
"""
Data Pre-processing Step
"""
###  Data Preprocessing (STATIC)- 10 categories--------------------------------

#'ACC_OF_NO', 'AGE', 'CONSTITUTION','C_NAME_CAT_REGION', 'C_NAME_CIRCLE', 
#'C_STATUS_MARITAL', 'GENDER', 'MOB', 'NEW_V2_CAT_OCCUPATION'
X_train_static = X_train_static.values                              # df to numpy array
X_test_static = X_test_static.values

# text categories to integers
encoder = [None for i in range(10)] 
for i in [2,3,4,5,6,8]:
    encoder[i] = LabelEncoder()
    X_train_static[:,i] = encoder[i].fit_transform(X_train_static[:,i])
    X_test_static[:,i] = encoder[i].transform(X_test_static[:,i])
X_train_static = X_train_static.astype('float32')
X_test_static = X_test_static.astype('float32')

# Dummy Variables
onehotencoder = OneHotEncoder(categorical_features=[2,3,4,5,6,8])
X_train_static = onehotencoder.fit_transform(X_train_static).toarray()
X_test_static = onehotencoder.transform(X_test_static).toarray()

# normalize features
scaler = StandardScaler()
X_train_static = scaler.fit_transform(X_train_static)
X_test_static = scaler.transform(X_test_static)

## Data Preprocessing (Dynamic & Target)---------------------------------------
# Computing mean, std. dev. vector & doing Mu-sigma normalization 
mean = []
std = []
timesteps = 12
for i in range(num_dy_var+num_txn_tags):
    col = []
    for j in range(timesteps):
        col.append(i+(j*(num_dy_var+num_txn_tags)))
    mean_std_value = (mean_std(X_train_dynamic[X_train_dynamic.columns[col]]))
    if "MDAB" in X_train_dynamic.columns[col][0]: # Normalization for y
        y_mean = mean_std_value[0]
        y_std = mean_std_value[1]        
    mean.append(mean_std_value[0])
    std.append(mean_std_value[1])

X_mean = mean+mean+mean+mean+mean+mean+mean+mean+mean+mean+mean+mean # replicate across mnths
X_std = std+std+std+std+std+std+std+std+std+std+std+std

X_train_dynamic = X_train_dynamic.values
X_test_dynamic = X_test_dynamic.values
y_train = y_train.values
y_test = y_test.values
X_train_dynamic = (X_train_dynamic - X_mean)/X_std
X_test_dynamic = (X_test_dynamic - X_mean)/X_std
y_train = (y_train - y_mean)/y_std
y_test = (y_test - y_mean)/y_std

# reshape dynamic input & output to be 3D [samples, timesteps, features]
# reshape static input to be 2D [samples, features] - already the same
X_train_dynamic = X_train_dynamic.reshape(X_train_dynamic.shape[0],timesteps,int(X_train_dynamic.shape[1]/timesteps))
X_test_dynamic = X_test_dynamic.reshape(X_test_dynamic.shape[0],timesteps,int(X_test_dynamic.shape[1]/timesteps))
#pred_timestep = 1
#y_train = y_train.reshape(y_train.shape[0],pred_timestep,1)
#y_test = y_test.reshape(y_test.shape[0],pred_timestep,1)
y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)
#______________________________________________________________________________

# Model Development
main_input = Input(shape=(X_train_dynamic.shape[1], X_train_dynamic.shape[2]))
encoded1 = LSTM(10,return_sequences=True,activation='tanh',dropout=0.2)(main_input)
encoded = LSTM(10,activation='tanh',dropout=0.1)(encoded1)
auxiliary_input = Input(shape=(X_train_static.shape[1],))
hidden_auxiliary = Dense(25, activation='tanh',kernel_regularizer=regularizers.l2(0.01))(auxiliary_input)
output_auxiliary = Dense(10, activation='tanh',kernel_regularizer=regularizers.l2(0.01))(hidden_auxiliary)
#output_auxiliary = Dropout(0.1, seed=42)(output_auxiliary)
x = concatenate([encoded, output_auxiliary])
#decoded = RepeatVector(y_train.shape[1])(x)
#main_output = LSTM(1, return_sequences=True,activation='linear')(decoded)   # return_sequences=True gives Many to Many relationship
hidden_main = Dense(7, activation='tanh',kernel_regularizer=regularizers.l2(0.01))(x)
output_main = Dense(1, activation='linear',kernel_regularizer=regularizers.l2(0.01))(x)
model = Model(inputs=[main_input,auxiliary_input], outputs=[output_main])
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit([X_train_dynamic, X_train_static], [y_train],
                    validation_data=([X_test_dynamic, X_test_static], y_test), 
                    epochs=30, batch_size=128)
#history = model.fit([X_train_dynamic], [y_train],
#                    validation_data=([X_test_dynamic], [y_test]), 
#                    epochs=20, batch_size=128)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
#### ACCURACY METRIC (make a prediction)
yhat = model.predict([X_test_dynamic, X_test_static])
#yhat = model.predict([X_test_dynamic])
yhat = yhat.reshape(yhat.shape[0],yhat.shape[1])
y_test = y_test.reshape(y_test.shape[0],y_test.shape[1])
### invert scaling for forecast
inv_yhat = y_mean + (y_std*yhat)
inv_y = y_mean + (y_std*y_test)
##
#test_acc = np.sum(np.abs((inv_yhat - inv_y)/inv_y)<0.1)/(inv_y.shape[0])
### calculate RMSE
#rmse = np.sqrt(mean_squared_error(inv_y.flatten(), inv_yhat.flatten()))
#print('Test RMSE: %.3f' % rmse)
# Calculate r square
print ('The test r squared value is: ',r2_score(inv_y,inv_yhat))
#-----------------------------------------------------------------------------
# TRAINING Accuracy
yhat_train = model.predict([X_train_dynamic, X_train_static])
yhat_train = yhat_train.reshape(yhat_train.shape[0],yhat_train.shape[1])
y_train = y_train.reshape(y_train.shape[0],y_train.shape[1])
### invert scaling for forecast
inv_yhat_train = y_mean + (y_std*yhat_train)
inv_y_train = y_mean + (y_std*y_train)

# Calculate r square
print ('The train r squared value is: ',r2_score(inv_y_train,inv_yhat_train))
#train_acc = np.sum(np.abs((inv_yhat_train - inv_y_train)/inv_y_train)<0.1)/(inv_y_train.shape[0])
#print (train_acc, test_acc)
###
### calculate RMSE
#rmse = np.sqrt(mean_squared_error(inv_y_train.flatten(), inv_yhat_train.flatten()))
#print('Train RMSE: %.3f' % rmse)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

### calculate mean absolute percentage error
mape = np.abs((inv_y.flatten() - inv_yhat.flatten())/(inv_y.flatten()+0.0001))
##mape = sum(mape)/len(mape)
##----------------------------------------------------------------------------
## Reverse Engineering
#X_tr['pred1']=inv_yhat_train[:,0]
##X_tr['pred2']=inv_yhat_train[:,1]
##X_tr['pred3']=inv_yhat_train[:,2]
#X_te['pred1']=inv_yhat[:,0]
##X_te['pred2']=inv_yhat[:,1]
##X_te['pred3']=inv_yhat[:,2]
#
##X_tr.to_csv(r'C:\Users\Axis_Inside\Music\burgundy_testing\updated_train_data.csv')
##X_te.to_csv(r'C:\Users\Axis_Inside\Music\burgundy_testing\updated_test_data.csv')
#
##writer = pd.ExcelWriter(r'C:\Users\Axis_Inside\Music\burgundy_testing\updated_test_data.xlsx')
##X_te.to_excel(writer,'Sheet1')
##writer.save()