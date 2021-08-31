import re
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import my_tools as mt
import tensorflow as tf
import librosa


SAMPLING_RATE = 22050
VALID_SPLIT = 0.1
TEST_SPLIT = 0.3
SHUFFLE_SEED =  43
SAMPLES_TO_DISPLAY = 10
SCALE = 0.5

BATCH_SIZE = 128
EPOCHS = 50

def paths_and_labels_to_dataset_train(audio_paths, labels):
    audios_from_paths = []
    for path in audio_paths:
        audio, sr = librosa.load(path, sr=SAMPLING_RATE)
        x = audio.shape[0]
        audio = np.reshape(audio, (x, 1))
        audios_from_paths.append(audio)
        #audio = mt.array_to_tensor(audio)
    print(len(audios_from_paths))
    audio_ds = audios_from_paths
    audio_ds = tf.data.Dataset.from_tensor_slices(audio_ds)
    # """Constructs a dataset of audios and labels."""
    # path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    # print(path_ds)
    # audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def paths_and_labels_to_dataset_test(audio_paths, minutes):
    audios_from_paths = []
    for path in audio_paths:
        audio, sr = librosa.load(path, sr=SAMPLING_RATE)
        x = audio.shape[0]
        audio = np.reshape(audio, (x, 1))
        audios_from_paths.append(audio)
        #audio = mt.array_to_tensor(audio)
    print(len(audios_from_paths))
    audio_ds = audios_from_paths
    minutes_ds   = minutes
    audio_ds = tf.data.Dataset.from_tensor_slices(audio_ds)
    minutes_ds = tf.data.Dataset.from_tensor_slices(minutes_ds)
    return tf.data.Dataset.zip((audio_ds, minutes_ds))





def path_to_audio(path):
    """Reads and decodes an audio file."""
    print(path)
    audio, sr = librosa.load(path, sr=SAMPLING_RATE)
    x = audio.shape[0]
    audio = np.reshape(audio, (x, 1))
    audio = mt.array_to_tensor(audio)
    print(audio.shape)
    #audio = tf.io.read_file(path)
    #audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def audio_to_fft(audio):
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)
    fft = tf.math.abs(fft[:, : (audio.shape[1] // 2), :])
    return fft



def queen_info(filepath):
  filename = os.path.basename(filepath)
  filename = filename.lower()
  filename = filename.strip()
  info = re.split(pattern = r"[-_]", string = filename)
  #info = np.asarray(info)
  if info[0] == 'pip':
    name = 0             # 0 = pipping
  elif info[0] == 'quack':
    name = 1             # 1 = quacking
  elif info[1] == ' missing queen ':
    name = 2 
  elif info[1] == ' active ':
    name = 2 
  elif info[4] == 'no':
    name = 2
  elif info[4] == 'queenbee':
    name = 2
  return name

def prepare_training_testing(train_input_paths, train_input_labels, test_input_paths, test_input_labels):
    num_val_samples = int(VALID_SPLIT * len(train_input_paths))
    train_audio_paths = train_input_paths[:-num_val_samples]
    train_labels = train_input_labels[:-num_val_samples]
    valid_audio_paths = train_input_paths[-num_val_samples:]
    valid_labels = train_input_labels[-num_val_samples:]
    test_audio_paths = test_input_paths
    test_labels = test_input_labels
    train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
    
    train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
        BATCH_SIZE
    )
    
    valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
    valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)
    
    test_ds = paths_and_labels_to_dataset(test_audio_paths, test_labels)
    test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)
    
    train_ds = train_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    valid_ds = valid_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    test_ds = test_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    return train_ds, valid_ds, test_ds

def training_and_evaluation(model, train_ds, valid_ds, test_ds, earlystopping_cb, mdlcheckpoint_cb, target_names):
    history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=(valid_ds),
    callbacks=[earlystopping_cb, mdlcheckpoint_cb],
    )   
    print(model.evaluate(valid_ds))
    
    for audios, labels in test_ds.take(1):
        # Get the signal FFT
        #ffts = audio_to_fft(audios)
        #print(len(ffts))
        # Predict
        y_pred = model.predict(audios)
        y_pred = np.argmax(y_pred, axis=-1)
        audios = audios.numpy()
        labels = labels.numpy()
        cnf_matrix = confusion_matrix(labels, y_pred)
        np.set_printoptions(precision=2)
        
        mt.plot_confusion_matrix(cnf_matrix, classes=target_names,
                              title='Confusion matrix:')
        plt.show()
        print ('\nClasification report for fold:\n', 
               classification_report(labels, y_pred, target_names=target_names ))
        
        return y_pred, labels
    