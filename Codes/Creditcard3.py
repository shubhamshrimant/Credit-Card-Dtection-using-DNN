# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:41:21 2020

@author: shubh
"""

import pandas as pd
import numpy as np

file=pd.read_csv("C:\\Users\\shubh\\Downloads\\archive\\fraudTrain.csv")

print(file.head())

fraud=file[file['is_fraud']==1]
not_fraud=file[file['is_fraud']==0]

print(fraud.shape,not_fraud.shape)






import tensorflow as tf
tf.__version__

dataset=file

dataset['trans_date_trans_time'] = dataset['trans_date_trans_time'].str.replace(" ","")
dataset['category'] = dataset['category'].str.replace(" ","")
dataset['street'] = dataset['street'].str.replace(" ","")
dataset['job'] = dataset['job'].str.replace(" ","")

dataset['trans_date_trans_time'] = dataset['trans_date_trans_time'].str.replace(",","")
dataset['category'] = dataset['category'].str.replace(",","")
dataset['street'] = dataset['street'].str.replace(",","")
dataset['job'] = dataset['job'].str.replace(",","")

dataset['trans_date_trans_time'] = dataset['trans_date_trans_time'].str.replace("_","")
dataset['category'] = dataset['category'].str.replace("_","")
dataset['street'] = dataset['street'].str.replace("_","")
dataset['job'] = dataset['job'].str.replace("_","")

del dataset['merchant']
del dataset['first']
del dataset['last']
#del X['street']
del dataset['city']
del dataset['dob']
del dataset['trans_num']


X=dataset.iloc[:,2:-1]

y=dataset.iloc[:,-1]



print(X)


#del X['trans_num']
print(X)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


X.iloc[:,1] = le.fit_transform(X.iloc[:,1])
print(X)

X.iloc[:,3] = le.fit_transform(X.iloc[:,3])
print(X)

X.iloc[:,4] = le.fit_transform(X.iloc[:,4])
print(X)

X.iloc[:,5] = le.fit_transform(X.iloc[:,5])
print(X)

X.iloc[:,10] = le.fit_transform(X.iloc[:,10])
print(X)





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_res, y_res = sm.fit_sample(X_train, y_train.ravel()) 

print('After Oversampling, the shape of train_X: {}'.format(X_res.shape)) 
print('After Oversampling, the shape of train_y: {} \n'.format(y_res.shape)) 
  
print("After oversampling, counts of label '1': {}".format(sum(y_res == 1))) 
print("After oversampling, counts of label '0': {}".format(sum(y_res == 0))) 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_res = sc.fit_transform(X_res)
X_test = sc.transform(X_test)


ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the third hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy','Precision','Recall'])

# Training the ANN on the Training set
ann.fit(X_res, y_res, batch_size = 32, epochs = 10)

#Predictin on test data and accuracy score
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)



data=[[2703186189652095,'entertainment',300.34, 'M','561PerryCove', 'HI', 75002, 40.192, -90.24, 5002, 'teacher',1011121314,43.1507,-112.154]]

print(data)



data = np.array(data)
data[:,1] = le.fit_transform(data[:,1])
print(data)
data[:,3] = le.fit_transform(data[:,3])
print(data)

data[:,4] = le.fit_transform(data[:,4])
print(data)

data[:,5] = le.fit_transform(data[:,5])
print(data)

data[:,10] = le.fit_transform(data[:,10])
print(data)




data = sc.transform(data)
pred = ann.predict(data)

print(pred)
pred=(pred>0.5)
print(pred)

#Saving the Model

from tensorflow.keras.models import model_from_json

model_json = ann.to_json()
with open("final_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
ann.save_weights("final_model.h5")
print("Saved model to disk")

#Opening model and evaluating
'''

json_file = open('final_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("final_model.h5")
print("Loaded model from disk")


loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
pred11 = loaded_model.predict(X_test)
print(pred11)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

'''
