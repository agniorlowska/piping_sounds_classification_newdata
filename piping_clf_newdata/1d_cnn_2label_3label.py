import os
import classification as clf
import numpy as np
import feature_extraction as ft
import functions as fc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import my_tools as mt
import  pandas as pd
from IPython.display import display, Audio

import tensorflow as tf
from tensorflow import keras

#some global variables
SAMPLING_RATE = 22050
VALID_SPLIT = 0.1
SHUFFLE_SEED =  43
SAMPLES_TO_DISPLAY = 10
SCALE = 0.5

BATCH_SIZE = 128
EPOCHS = 50

target_names = ['tooting', 'quacking','bees']
class_names = ['tooting', 'quacking','bees']

#training data - our dataset
toot_directory = "/home/agnieszka/Desktop/data_piping/folds_only_piping/tooting" 
quack_directory = "/home/agnieszka/Desktop/data_piping/folds_only_piping/quacking"
toot_quack_directory = "/home/agnieszka/Desktop/data_piping/piping_dataset_with_bees"



#testing data - japanese dataset
test_directory = "/home/agnieszka/Desktop/data_piping/piping-all-db-1sec"

toot1_directory =  '/Users/agnieszkaorlowska/Downloads/piping-db-1sec/'
quack1_directory = '/Users/agnieszkaorlowska/Downloads/quacking-db-1sec/'

#training
names = os.listdir(toot_quack_directory)
audio_paths = []
labels = []
for label, name in enumerate(names):
    print("Processing speaker {}".format(name,))
    
    speaker_sample_paths = [
        os.path.join(toot_quack_directory, filepath)
        for filepath in os.listdir(toot_quack_directory)
        if filepath.endswith(".wav")
    ]
    q = fc.queen_info(name)
    print(q)
    labels.append(q)
    print(len(labels))
    print(len(speaker_sample_paths))
    audio_paths = speaker_sample_paths
    
    
       
print(
    "Training: Found {} files belonging to {} classes.".format(len(audio_paths), len(set(labels)))
)

rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)

# Split into training and validation
num_val_samples = int(VALID_SPLIT * len(audio_paths))
print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]


print("Using {} files for validation.".format(num_val_samples))
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]



train_ds = fc.paths_and_labels_to_dataset_train(train_audio_paths, train_labels)
train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
    BATCH_SIZE
)

valid_ds = fc.paths_and_labels_to_dataset_train(valid_audio_paths, valid_labels)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)



#Transform audio wave to the frequency domain using `audio_to_fft`
train_ds = train_ds.map(
    lambda x, y: (fc.audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

valid_ds = valid_ds.map(
    lambda x, y: (fc.audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
)
valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)


#testing
names = os.listdir(test_directory)
test_audio_paths = []
test_labels = []
minutes = []
for label, name in enumerate(names):
    print("Processing speaker {}".format(name,))
    
    speaker_sample_paths = [
        os.path.join(test_directory, filepath)
        for filepath in os.listdir(test_directory)
        if filepath.endswith(".wav")
    ]
    minute = ft.second_info(name)
    minutes.append(minute)
    print(len(test_labels))
    print(len(speaker_sample_paths))
    test_audio_paths = speaker_sample_paths
    test_minutes = minutes
    
print(
    "Testing: Found {} files belonging to {} classes.".format(len(test_audio_paths), len(set(labels)))
)


rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)


test_ds = fc.paths_and_labels_to_dataset_test(test_audio_paths, test_minutes)
test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(599)

test_ds = test_ds.map(
    lambda x, y: (fc.audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)




model = clf.make_model_1DCNN((SAMPLING_RATE // 2, 1), len(names))
model.summary()

model.compile(
    optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model_save_filename = "model.h5"

earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=(valid_ds),
    callbacks=[earlystopping_cb, mdlcheckpoint_cb],
)

print(model.evaluate(valid_ds))
y_pred = []

print(test_audio_paths)
print(test_labels)


df = pd.DataFrame()


for audios, minutes in test_ds.take(1):
    # Get the signal FFT
    #ffts = fc.audio_to_fft(audios)
    #print(len(ffts))
    # Predict
    y_pred = model.predict(audios)
    y_pred = np.argmax(y_pred, axis=-1)
    # Take random samples
    print(len(audios))
    #print(audios)
    print(len(minutes))
    print(minutes)
    print(len(y_pred))
    print(y_pred)

df['Time Stamp'] = minutes
rounded_predictions = y_pred
print(rounded_predictions)
df['Predicted label (binary)'] = rounded_predictions.tolist()

new_labels = []
for i in rounded_predictions:
    if i == 0:
        new_label = 'tooting'
    elif i == 1:
        new_label = 'quacking'
    elif i == 2:
        new_label = 'bees'
    new_labels.append(new_label)
        

df['Predicted label'] = new_labels
    
                
df = df.sort_values(by='Time Stamp',ascending=True)
print(df)


df.to_csv("/home/agnieszka/Desktop/201405140302HSExtractorBPF150-500_predictions_mean_STFT_DA_3label_1dcnn.csv")
    
    
 

