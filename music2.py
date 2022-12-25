import collections
import datetime
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf

from keras.models import load_model
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

def music():
    data_dir = pathlib.Path('data/maestro-v2.0.0')
    if not data_dir.exists():
      tf.keras.utils.get_file(
          'maestro-v2.0.0-midi.zip',
          origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
          extract=True,
          cache_dir='.', cache_subdir='data',
      )

    filenames = glob.glob(str(data_dir/'**/*.mid*'))
    print('Number of files:', len(filenames))

    sample_file = filenames[1]
    print(sample_file)

    pm = pretty_midi.PrettyMIDI(sample_file)

    print('Number of instruments:', len(pm.instruments))
    instrument = pm.instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    print('Instrument name:', instrument_name)

    def midi_to_notes(midi_file: str) -> pd.DataFrame:
      pm = pretty_midi.PrettyMIDI(midi_file)
      instrument = pm.instruments[0]
      notes = collections.defaultdict(list)

      sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
      prev_start = sorted_notes[0].start

      for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

      return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

    def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
      mse = (y_true - y_pred) ** 2
      positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
      return tf.reduce_mean(mse + positive_pressure)

    model = load_model('trained-models/music_model.h5', custom_objects={'mse_with_positive_pressure':
    mse_with_positive_pressure})

    def notes_to_midi(
      notes: pd.DataFrame,
      out_file: str,
      instrument_name: str,
      velocity: int = 100,
    ) -> pretty_midi.PrettyMIDI:

      pm = pretty_midi.PrettyMIDI()
      instrument = pretty_midi.Instrument(
          program=pretty_midi.instrument_name_to_program(
              instrument_name))

      prev_start = 0
      for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

      pm.instruments.append(instrument)
      pm.write(out_file)
      return pm

    def predict_next_note(
            notes: np.ndarray,
            keras_model: tf.keras.Model,
            temperature: float = 1.0) -> int:

        assert temperature > 0

        inputs = tf.expand_dims(notes, 0)

        predictions = model.predict(inputs)
        pitch_logits = predictions['pitch']
        step = predictions['step']
        duration = predictions['duration']

        pitch_logits /= temperature
        pitch = tf.random.categorical(pitch_logits, num_samples=1)
        pitch = tf.squeeze(pitch, axis=-1)
        duration = tf.squeeze(duration, axis=-1)
        step = tf.squeeze(step, axis=-1)

        step = tf.maximum(0, step)
        duration = tf.maximum(0, duration)

        return int(pitch), float(step), float(duration)

    temperature = 2.0
    num_predictions = 120
    seq_length = 25
    vocab_size = 128

    key_order = ['pitch', 'step', 'duration']
    raw_notes = midi_to_notes(sample_file)

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

    input_notes = (
        sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
      pitch, step, duration = predict_next_note(input_notes, model, temperature)
      start = prev_start + step
      end = start + duration
      input_note = (pitch, step, duration)
      generated_notes.append((*input_note, start, end))
      input_notes = np.delete(input_notes, 0, axis=0)
      input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
      prev_start = start

    generated_notes = pd.DataFrame(
        generated_notes, columns=(*key_order, 'start', 'end'))

    generated_notes.head(10)

    out_file = 'exported-art/output.mid'
    out_pm = notes_to_midi(
        generated_notes, out_file=out_file, instrument_name=instrument_name)
