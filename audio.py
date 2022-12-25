import mido
from keras.optimizers import Adam
from keras.models import load_model
from mido import MidiFile, MidiTrack, Message
from keras.layers import LSTM, Dense, Activation, Dropout, Flatten
from keras.preprocessing import sequence
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import tensorflow as tf
import os


notes = []
for song in os.listdir("chillhopdata"):
    mid = MidiFile("chillhopdata/" + song)
    for msg in mid:
        if not msg.is_meta and msg.channel == 0 and msg.type == 'note_on':
            data = msg.bytes()
            notes.append(data[1])

scaler = MinMaxScaler()
notes = list(scaler.fit_transform(np.array(notes).reshape(-1, 1)))

notes = [list(note) for note in notes]

X = []
y = []

n_prev = 30

for i in range(len(notes) - n_prev):
  X.append(notes[i:i+n_prev])
  y.append(notes[i+n_prev])

X_test = X[-300:]
X = X[:-300]
y = y[:-300]

'''
model = Sequential()
model.add(LSTM(256, input_shape=(n_prev, 1), return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(128, input_shape=(n_prev, 1), return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(64, input_shape=(n_prev, 1), return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(32, input_shape=(n_prev, 1), return_sequences=False))
model.add(Dropout(0.6))
model.add(Dense(1))
model.add(Activation('linear'))
print(model.summary())

optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer)

model.fit(np.array(X), np.array(y), batch_size=32, epochs=2000, verbose=1)

model.save("trained-models/audio_model.h5")
print("Model Saved")
'''

def music():
    model = load_model('trained-models/audio_model.h5')

    prediction = model.predict(np.array(X_test))
    prediction = np.squeeze(prediction)
    prediction = np.squeeze(scaler.inverse_transform(prediction.reshape(-1, 1)))
    prediction = [int(i) for i in prediction]

    mid = MidiFile()
    track = MidiTrack()
    t = 0
    for note in prediction:
        note = np.asarray([147, note, 67])
        bytes = note.astype(int)
        msg = Message.from_bytes(bytes[0:3])
        t += 1
        msg.time = t
        track.append(msg)
    mid.tracks.append(track)
    mid.save('exported-art/music.mid')
