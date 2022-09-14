from music21 import *
import numpy as np
import os

def flatten(l):
    ''' Flatten a list of lists to just a list... '''
    return [item for sublist in l for item in sublist]


def get_notes(parsed_midi):
    '''
        Extract notes (and chords if you want) from a parsed midi structure.
        Seems to always fail at partitionByInstrument, so extracts only the flat structure
        Can this function be improved by only extracting particular instruments?
        Not fond of all the .append-ing, memory allocation may take a lot of time?
    '''
    notes = []
    flat_notes = []
    chords = []

    #print("Partitioning %s" % parsed_midi)

    notes_to_parse = []

    try: # file has instrument parts. always fails??
        s2 = instrument.partitionByInstrument(parsed_midi)
        for i in range(len(s2)):
            notes_to_parse.append(s2.parts[i].recurse())
    except: # file has notes in a flat structure
        #print(f'Flat structure detected in {parsed_midi}')
        notes_to_parse.append(parsed_midi.flat.notes)
    
    # time to init the arrays to hold the actual notes
    notes_row = []
    flat_notes_row = []
    chords_row = []
    
    for instrument in notes_to_parse:
        # should add: if instrument.name is in [Strings, Guitars] or similar
        for element in instrument:
            if isinstance(element, note.Note):
                notes_row.append(element)
                flat_notes_row.append(str(element.pitch))
        
            elif isinstance(element, chord.Chord):
                chords_row.append(element)
                #notes.append('.'.join(str(n) for n in element.normalOrder)) not extracting notes

        notes.append(notes_row)
        flat_notes.append(flat_notes_row)
        chords.append(chords_row)

    #with open('data/notes', 'wb') as filepath:
    #    pickle.dump(notes, filepath)
    
    # since we do not handle separate instruments later, it makes sense to flatten
    # the arrays of arrays now
    return flatten(notes), flatten(flat_notes), flatten(chords)


def unique(list):
    ''' Returns a sorted list of the unique items in the input list '''

    # initialize a null list
    unique_items = []
  
    # traverse for all elements
    for x in list:
        # check if exists in unique_list or not
        if str(x) not in unique_items:
            unique_items.append(str(x))
    
    return sorted(unique_items)


def transition_matrix(long_list):
    ''' Creates a transition matrix to predict the probability of one state following another.
        The unique states are identified in the unique_items list, which decides the size of
        the transition matrix. The transition matrix is square with dim = len(unique_items).
        
        Each row X in the matrix contains the probabilities of different states following the
        state at position X in unique_list. Each row sums to 1.0:

            E       A       G       D
        E   0.2     0.3     0.0     0.5
        A   0.1     0.0     0.9     0.0
        G   0.2     0.3     0.2     0.3
        D   0.4     0.1     0.0     0.5

        where the numbers are found in the transition matrix, and the states are found in the
        unique_items list: ['E', 'A', 'G', 'D']
        '''

    # identify unique items in long list
    unique_items = unique(long_list)
    unique_items_freq = np.zeros([len(unique_items)])
    
    # init transition matrix
    t_matrix = np.zeros([len(unique_items), len(unique_items)])

    # traverse long list and fill transition matrix
    print(f'Traversing list of length {len(long_list)}...')
    for pos, item in enumerate(long_list[:-1]):
        i = unique_items.index(str(long_list[pos]))
        j = unique_items.index(str(long_list[pos+1]))
        t_matrix[i][j] += 1
        unique_items_freq[i] += 1
    
    # count the last item of long_list in freq
    i = unique_items.index(str(long_list[-1]))
    unique_items_freq[i] += 1
    #normalise (convert to frequency)
    unique_items_freq /= np.sum(unique_items_freq)
    
    # normalise transition matrix per row (convert to probability)
    for row in t_matrix:
        row /= np.sum(row)
        
    print(f'Done. Found {len(unique_items)} unique items.')

    return t_matrix, unique_items, unique_items_freq


def notes_cleaning(flat_notes):
    ''' 
    Notes from the midis look like this: D#4, D4, D-4 etc. where D4 and D-4 are
    duplicates. This function changes all notes with strlen=2 to have strlen=3
    by inserting a '-': D4 becomes D-4. 
    NB! Inplace, so replaces old input array.
    '''

    for c, item in enumerate(flat_notes):
        if len(item) == 2:    
            flat_notes[c] = item[0] + '-' + item[1]

    return flat_notes


def notes_compress12(flat_notes):
    ''' Removes everything but the note and a possible # after the note:
        D#4 becomes D#, D-4 becomes D. Returns a new array. If used,
        can replace notes_cleaning().
    '''

    flat_notes12 = [0]*len(flat_notes)

    for c, item in enumerate(flat_notes):
        if len(item) >= 2:
            if item[1] == '#':
                flat_notes12[c] = item[0:2]
            else:
                flat_notes12[c] = item[0:1]
    
    return flat_notes12


def parse_midi_files(path, max=100):
    ''' Parses the first (#max) midi files found at path, including subfolders.
        Returns a list of midi structures.
    '''

    files_list = []
    for (root,dirs,files) in os.walk(path, topdown=True):
        for file in files:
            if file[-4:] == '.mid':
                files_list.append(os.path.join(root, file))

    print(f'Found {len(files_list)} .mid files')

    parsed_midi = []
    print(f'Parsing {min(len(files_list), max)} files', end='')
    
    for c, file in enumerate(files_list[:min(len(files_list), max)]):
        parsed_midi.append(converter.parse(file))
        if (c % 10 == 0):
            print(c, end='')
        elif (c % 5 == 0):
            print('.', end='')
        
    
    print('')

    return parsed_midi


def get_notes_list(parsed_midi):
    ''' Takes a list of parsed midi structures and extracts the notes.
        Returns one long list of notes.
    '''
    long_list = []
    for m in parsed_midi:
        notes, flat_notes, chords = get_notes(m)
        long_list += flat_notes
    
    return long_list
