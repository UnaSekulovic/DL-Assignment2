import pickle
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.keras import layers

#load data
dict = pickle.load(open('california-housing-dataset.pkl', 'rb'))
x_train, y_train = dict['x_train'], dict['y_train']
x_test, y_test = dict['x_test'], dict['y_test']

#data overview
df = pd.DataFrame(x_train)
#profile = ProfileReport(df, title="Pandas Profiling Report")
#profile.to_file("data_profile_report.html")

#split train set into train and validation set
x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
#df1 = pd.DataFrame(x_train2)
#profile1 = ProfileReport(df1, title="Pandas Profiling Report")
#profile1.to_file("data_profile_report1.html")
#df2 = pd.DataFrame(x_val)
#profile2 = ProfileReport(df2, title="Pandas Profiling Report")
#profile2.to_file("data_profile_report2.html")
#df.describe()

#normalization
scaler = preprocessing.StandardScaler()
scaler.fit(x_train2)

x_train_norm = scaler.transform(x_train2)
x_val_norm = scaler.transform(x_val)
x_test_norm = scaler.transform(x_test)

model = tf.keras.Sequential()
#input layer
model.add(layers.Dense(units=8, activation='relu'))
#hidden layers
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=64, activation='relu'))

#output layer
model.add(layers.Dense(units=1))

model.compile(tf.keras.optimizers.Adam(0.00275), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

history = model.fit(x_train_norm, y_train2, epochs=100, batch_size=32, validation_data=(x_val_norm, y_val))
