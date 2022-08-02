import time
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import db.dbconnect as dbc

from matplotlib.colors import rgb2hex
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix, classification_report


## statics
CLASS_LABELS = ['Legitimate', 'Fraudulent']

## get dataset from db
## Open dataset containing 17,880 real-life job ads 
## 17,014 legitimate and 866 fraudulent job ads published between 2012 to 2014
## available at Employment Scam Aegean Dataset http://emscad.samos.aegean.gr/ 
csvfile_path = 'db/emscad_v1.csv'
df_raw = pd.read_csv(csvfile_path)

## clean data
## Drop similar columns
df_cleaned = df_raw.copy()

# Replace NaN in function with department
df_cleaned.loc[df_cleaned['function'].isnull(),'function'] = df_cleaned['department']
df_cleaned = df_cleaned.drop(columns=['department'])

## Standardise data
df_cleaned[['salary_range_min', 'salary_range_max']] = df_cleaned['salary_range'].str.split('-', 1, expand=True)
df_cleaned['salary_range_min'] = df_cleaned['salary_range_min']
df_cleaned['salary_range_min'] = df_cleaned['salary_range_min'].fillna(-1)
df_cleaned['salary_range_min'] = df_cleaned['salary_range_min'].astype(int)
df_cleaned['salary_range_max'] = df_cleaned['salary_range_max']
df_cleaned['salary_range_max'] = df_cleaned['salary_range_max'].fillna(-1)
df_cleaned['salary_range_max'] = df_cleaned['salary_range_max'].astype(int)
df_cleaned[['country', 'state', 'city']] = df_cleaned['location'].str.split(',', 2, expand=True)

df_balanced = df_cleaned[df_cleaned['in_balanced_dataset'] == 't'] # 16980 false 900 true
df_balanced = df_balanced.drop(columns=['salary_range', 'location', 'in_balanced_dataset'])
df_cleaned = df_cleaned.drop(columns=['salary_range', 'location', 'in_balanced_dataset'])

## Convert string and HTML data to categorical/nominal data type
def to_cat(df):
    df = df.replace(r'^\s+$', np.nan, regex=True)
    for col in df.columns:
        if 'salary' in col:
            continue
        df[col] = df[col].astype('category').cat.codes
    return df

df_balanced = to_cat(df_balanced)
df_cleaned = to_cat(df_cleaned)

# Datasets and classes
classes = df_balanced['fraudulent']
df_balanced = df_balanced.drop(columns=['fraudulent'])

classes_all = df_cleaned['fraudulent']
df_cleaned = df_cleaned.drop(columns=['fraudulent'])

ds = df_balanced
ds_all = df_cleaned



## split training and validation dataset
# oversample fraudulent samples
ds_all, classes_all = SMOTE().fit_resample(ds_all, classes_all)

# scale to standardise dataset
scaled_features = StandardScaler().fit_transform(ds.values)
ds = pd.DataFrame(scaled_features, index=ds.index, columns=ds.columns)
scaled_features = StandardScaler().fit_transform(ds_all.values)
ds_all = pd.DataFrame(scaled_features, index=ds_all.index, columns=ds_all.columns)

X_tr, X_te, y_tr, y_te = train_test_split(ds, classes ,random_state=108, test_size=0.27)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, random_state=108, test_size=0.27)


## build model
# initialise parameters
n_batch, epochs, out, n_features = 32, 100, 1, len(ds.columns)
# input layer
visible = tf.keras.layers.Input(shape=(n_features,), name='input source')
# normalisation layers
norm_batch1 = tf.keras.layers.BatchNormalization()
norm_batch2 = tf.keras.layers.BatchNormalization()
norm_batch3 = tf.keras.layers.BatchNormalization()
# dropout layers
drop = tf.keras.layers.Dropout(0.25) 
# fully connected layers
act = 'relu'
ki = 'he_uniform'
hidden1 = tf.keras.layers.Dense(64, activation=act, kernel_initializer=ki) 
hidden2 = tf.keras.layers.Dense(16, activation=act, kernel_initializer=ki)
hidden3 = tf.keras.layers.Dense(4, activation=act, kernel_initializer=ki)
output = tf.keras.layers.Dense(out, activation='relu')
# define model
model = tf.keras.Sequential(
    [ visible,
      hidden1,
      drop,
      norm_batch1,
      hidden2,
      drop,
      norm_batch2,
      hidden3,
      drop,
      norm_batch3,
      output
    ], name='fraud_detection_mlp')

model.summary()
print('\n\n')

## train model
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
callbacks_list = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=5e-2, patience=5, verbose=1,
                                             mode='auto', min_delta=1e-4, cooldown=0, min_lr=0),
        tf.keras.callbacks.EarlyStopping(patience=12, min_delta=1e-4, restore_best_weights=True),
        ]

model.compile(optimizer=opt,loss='mae', metrics='accuracy')

start_t = time.time()
history = model.fit(X_tr, y_tr, 
                    batch_size=n_batch,
                    epochs=epochs, 
                    verbose=True, 
                    shuffle=True, 
                    workers=4,
                    validation_data=(X_val, y_val),  
                    callbacks=callbacks_list,
                    use_multiprocessing=True)
end_t = time.time()
print("{t} seconds for training".format(t=end_t-start_t))
print("\n\n")

## save model (non-baseline models cannot be saved)
model.save('static/fraud_detection_mlp.h5', options=tf.saved_model.SaveOptions(save_debug_info=True)) 



CLASS_LABELS = ['Legitimate', 'Fraudulent']

## evaluate model
model.evaluate(X_te, y_te)

# plot validation accuracy and loss graph
if history != None:
    try:
        print(model.metrics_names)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.savefig('figures/fraud_detection_mlp_accuracy.png')
        plt.show()
        plt.clf()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.savefig('figures/fraud_detection_mlp_loss.png')
        plt.show()
        plt.clf()
    except:
        pass

## prediction
start = time.time()
prediction = model.predict(X_te, use_multiprocessing=True, workers=4)
end = time.time()
pred_time = end-start
print('\n\n')
print("{t} seconds for {d} prediction samples".format(t=end-start, d=len(prediction)))
print('\n\n')

for row in prediction:
    idx = 0 
    for col in row:
        row[idx] = col >= 0.5
        idx += 1

# confusion matrix
cm = confusion_matrix(prediction, y_te, labels=[0, 1])
print('\n\n')
plt.figure()
sns.heatmap(cm, cmap='plasma', xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
plt.savefig('figures/fraud_detection_mlp_confusion.png')
plt.show()
plt.clf()

# accuracy, f1, precision and recall
report = classification_report(prediction, y_te, labels=[0, 1], digits=6, output_dict=True)
report = pd.DataFrame(data=report)
report = (report.T)
#with open('fraud_detection_mlp_report.csv', "w") as file:
#    for elem in report:
#        file.write(elem)
print(report)