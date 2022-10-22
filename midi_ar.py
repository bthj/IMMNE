import os
import ipywidgets
import json
import pickle
import fractions

import numpy as np
import matplotlib.pyplot as plt

from IPython import display
from soundfont_utils import setup_soundfonts, SFInstrument, CustomMidiFile

# Based on Marius Aasan's Jupyter notebook:

def get_song_dicts():
    song_dicts = None
    if not os.path.isfile('MIDI/song_dicts.pickle'):
        song_dicts = {}
        roots = ['MIDI/unprocessed/', 'MIDI/processed/']
        for root in roots:
            files = [root + f for f in os.listdir(root) if f.endswith('.mid')]
            for file in files:
                print(f'Processing: {file}')
                mf = CustomMidiFile(file)
                song_dicts[file] = mf.chord_dict
        # Serialize the song_dicts
        with open('MIDI/song_dicts.pickle', 'wb') as file:
            pickle.dump(song_dicts, file)
    else:
        with open('MIDI/song_dicts.pickle', 'rb') as file:
            song_dicts = pickle.load(file)

    return song_dicts

def get_AR(chord_dict, step=1):
    # Only step=1 atm.
    ar = np.zeros((128, 128))
    keylist = sorted(list(chord_dict.keys()))

    for i in range(len(keylist)-1):
        cur = chord_dict[keylist[i]]['note']
        nxt = chord_dict[keylist[i+1]]['note']
        for n in cur:
            for m in nxt:
                ar[n,m] += 1

    return ar

# returns:
# -- 1-step transition matrix as an autoregressive model
# -- transition probabilities
# -- entropy
def get_note_transition_matrix_prob_and_entropy():
    song_dicts = get_song_dicts()

    ar = np.zeros((128, 128))
    for dct in song_dicts.values():
        tmpar = get_AR(dct)
        ar += tmpar / tmpar.sum()**.5

    par = (ar+1e-5) / (ar+1e-5).sum(1)[:,None]
    har = -par * np.log2(par)

    return ar, par, har
