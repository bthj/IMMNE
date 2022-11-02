import os
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


'''
EXTRINSIC MOTIVATION:

-   Added a method for constructing a duration matrix for extrinsic and intrinsic
    motivation.

    NOTE:   Duration must be applied as seperate matrix for the rewards. For a
            note+duration model we require 4 matrices:
             - Extrinsic pitch,
             - Intrinsic pitch,
             - Extrinisic duration,
             - Intrinsic duration.

            The function 'get_AR_dur' constructs the Extrinsic duration matrix.
            Intrinsic duration is created and updated similar to the pitch matrix,
            using 'init_intrinsic_matrix'.
'''

def get_AR(chord_dict, step=1, prior=0.5):
    '''Creates Extrinsic Pitch Matrix'''
    # Only step=1 atm.
    ar = np.zeros((128, 128)) + 0.5
    keylist = sorted(list(chord_dict.keys()))

    for i in range(len(keylist)-1):
        cur = chord_dict[keylist[i]]['note']
        nxt = chord_dict[keylist[i+1]]['note']
        for n in cur:
            for m in nxt:
                ar[n,m] += 1

    return ar

# Two handy globals for handling durations
Q_DUR_FRACTIONS = [
    fractions.Fraction(numerator=i, denominator=24) for i in range(1,4*24+1)
]

Q_DUR_ARRAY = np.array([float(c) for c in Q_DUR_FRACTIONS])


def match_dur(dur, q=Q_DUR_ARRAY):
    '''Matches duration to closest discretized duration'''
    diffs = np.abs(dur - q)
    return np.argmin(diffs)


def get_AR_dur(chord_dict, q=Q_DUR_ARRAY, step=1, prior=0.5):
    '''Allows 1-step agent to apply durations in conjunction with pitches.
    '''
    # Only step=1 atm.
    ardr = np.zeros((len(q), len(q))) + prior
    keylist = sorted(list(chord_dict.keys()))

    for k in range(len(keylist)-1):
        curd = chord_dict[keylist[k]]['duration']
        nxtd = chord_dict[keylist[k+1]]['duration']

        for nd in curd:
            nd = match_dur(nd, q)
            for md in nxtd:
                md = match_dur(md, q)
                ardr[nd, md] +=1

    return ardr


def state2dur(dur_state):
    '''Utility function for mapping states to durations.
    '''
    return match_dur(dur_state)

'''
INTRINSIC MOTIVATION:

-   Added some options for one_step intrinsic motivation.
    Can now choose between Shannon entropy, Beta entropy,
    Shannon KL-divergence or Dirichlet KL-divergence.

'''

def init_intrinsic_matrix(shape=(128,128), prior=0.5):
    '''Used to initialize the experience matrix for intrinsic motivation'''
    return np.zeros(shape) + prior


def get_Shannon_entropy_and_update(prev_state, new_state, cur_har):
    '''Function that retrieves entropy of state given previous and updates matrix.

    Args:
        prev_state (int): Previous state / note
        new_state (int): The next state
        cur_har (int): Entropy Matrix

    Returns:
        float: Entropy as intrinsic reward for transition prev_step -> new_step
    '''
    p = (cur_har[prev_state, new_state] + 1e-5) / (cur_har[prev_state] + 1e-5).sum()
    cur_har[prev_state, new_state] += 1 # No need to return this, just update
    return -p*np.log2(p)

def get_Beta_entropy_and_update(prev_state, new_state, cur_har):
    '''Alternative to get_entropy_and_update using Beta entropy.

    Args:
        prev_state (int): Previous state / note
        new_state (int): The next state
        cur_har (int): Entropy Matrix

    Returns:
        float: Entropy as intrinsic reward for transition prev_step -> new_step
    '''
    from scipy.stats import beta as betadist

    # Fix error with Beta entropy
    alpha = np.copy(cur_har[prev_state])
    a = alpha[new_state]
    b = np.sum(alpha) - a

    return betadist.entropy(a, b)

def get_Shannon_KL_and_update(prev_state, new_state, cur_har):
    '''Alternative to get_entropy_and_update using Shannon KL divergence

    The KL divergence could be a better choice for intrinsic reward, as it returns
    a measure of distance between the distributions before and after the update.

    Args:
        prev_state (int): Previous state / note
        new_state (int): The next state
        cur_har (int): Entropy Matrix

    Returns:
        float: KL Divergence as intrinsic reward for transition prev_step -> new_step
    '''
    ps = (cur_har[prev_state] + 1e-5) / (cur_har[prev_state] + 1e-5).sum()
    cur_har[prev_state, new_state] += 1
    qs = (cur_har[prev_state] + 1e-5) / (cur_har[prev_state] + 1e-5).sum()
    return -(qs*np.log2(qs/ps)).sum()

def get_Dirichlet_KL_and_update(prev_state, new_state, cur_har):
    '''Alternative to get_entropy_and_update; using Dirichlet KL divergence

    Args:
        prev_state (int): Previous state / note
        new_state (int): The next state
        cur_har (int): Entropy Matrix

    Returns:
        float: KL Divergence as intrinsic reward for transition prev_step -> new_step
    '''
    from scipy.special import digamma, gammaln
    alpha = np.copy(cur_har[prev_state])
    cur_har[prev_state, new_state] += 1
    beta = np.copy(cur_har[prev_state])

    # Long computation, split into parts
    A = gammaln(np.sum(beta)) - gammaln(np.sum(alpha))
    B = np.sum(gammaln(alpha)) - np.sum(gammaln(beta))
    C = (beta - alpha) * (digamma(beta) - digamma(np.sum(beta)))

    # Return scalar instead of array
    return (A + B + C).sum()

# returns:
# -- 1-step transition matrix as an autoregressive model
# -- transition probabilities
# -- entropy
def get_note_transition_matrix_prob_and_entropy():
    song_dicts = get_song_dicts()

    ar = np.zeros((128, 128))
    for dct in song_dicts.values():
        tmpar = get_AR(dct)
        ar += tmpar

    # Using fresh intrinsic matrix, ignore par
    har = init_intrinsic_matrix()

    return ar, har
