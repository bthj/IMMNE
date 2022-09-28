import music21
import pretty_midi
import numpy as np
import os

def flatten(l):
    ''' Flatten a list of lists to just a list... '''
    return [item for sublist in l for item in sublist]


def midi_statistics(pm_list):

    instrument_names = []
    instruments_table = []

    for pm in pm_list:
        for instrument in pm.instruments:
            if instrument.is_drum: 
                continue # skip the drums
            
            instrument.remove_invalid_notes()
            if instrument.name not in instrument_names:
                instrument_names.append(instrument.name)
                instruments_table.append([instrument.name, 0, pretty_midi.program_to_instrument_class(instrument.program)])
                                
            instruments_table[instrument_names.index(instrument.name)][1] += len(instrument.notes)
    
    print(f'Analyzed {len(pm_list)} midis, with {len(instrument_names)} unique instruments')

    return instruments_table

def parse_to_midi(files_list):
    '''
    Takes a list of files (including path) and parses to pretty_midi structures.
    Returns lists of file names and pm structures and a table of the instruments that
    was encountered.
    '''
    song_names = []
    parsed_midis = []

    for file in files_list:
        try:
            pm = pretty_midi.PrettyMIDI(file)
        except:
            continue # skip bad files
        
        song_names.append(os.path.split(file)[1])
        parsed_midis.append(pm)
    
    print(f'Parsed {len(song_names)} of {len(files_list)} files.')

    return song_names, parsed_midis


def group_chords(notes_sorted, verbose=0):
    '''
    Takes an array of pretty_midi notes (from a single or multiple instruments)
    sorted by note.start. Returns an array of notes grouped by which notes
    are active at any time.  
    '''

    old_active = []
    list_of_actives = []

    for i, note in enumerate(notes_sorted[:-1]):
        new_active = list(filter(lambda x: x.end >= note.start, old_active))
        new_active.append(note)
        old_active = new_active
        
        if notes_sorted[i+1].start == note.start:
            continue
        
        list_of_actives.append(new_active)

    return list_of_actives


def group_chords_deprecated(notes, verbose=0):
    '''
    Returns an array of all notes, where notes that share start time are
    grouped in subarrays. Notes must be sorted by notes.start
    '''

    grouped_notes = []

    note_times = []
    chords = 0
    
    notes_in_this_chord = 0
    notes_in_chords = 0

    for c, note in enumerate(notes[:-1]):
        if notes[c].start == notes[c+1].start:
            # part of chord
            notes_in_this_chord += 1
            notes_in_chords += 1

            if notes[c].start not in note_times:
                # chord not encountered before
                chords += 1
                note_times.append(note.start)
        else:
            grouped_notes.append(notes[c-notes_in_this_chord:c+1])
            notes_in_this_chord = 0

    notes_in_chords += chords #counting the first note twice equals counting last note

    if verbose:
        print(f'Note array had {len(notes)-notes_in_chords} single notes, {chords} chords')
    
    return grouped_notes


def get_notes_from_pm(pm, verbose=0):
    '''
        Extracts notes from a pretty_midi structure, one instrument at the
        time. Chords (notes being played at the same time) are replaced with
        their root note.
    '''

    pm_notes = []

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue # skip the drums.

        instrument.remove_invalid_notes()
        sorted_notes = sorted(instrument.notes, key=lambda x: x.start)

        pm_notes.append(sorted_notes)

        if verbose >= 2:
            print(f'Instrument {instrument.name} has: \
            {len(instrument.notes)} notes, {len(instrument.pitch_bends)} pitch_bends {len(instrument.control_changes)} control_changes')
    
    return pm_notes


def get_notes(pm_list, verbose=0):
    '''
        Takes a list of pretty_midi structures and returns all the notes
        in a list structure:    pm_list[
                                    song0[
                                        instrument0[
                                            notes]
                                        instrument1[
                                            notes]
                                        ]
                                    song1[
                                        instrument0[
                                            notes]
                                        ]
                                    ]
                                ]
        (Or a list of songs with a list of instruments with a list of notes)
    '''

    notes = []
    num_instruments = 0

    for pm in pm_list:
        notes.append(get_notes_from_pm(pm, verbose=verbose))
        num_instruments += len(notes[-1])

    print(f'Extracted notes from {len(notes)} midis, {num_instruments} instruments')

    return notes


def group_chords_in_list(notes, verbose=0):
    '''
    notes is a list of lists of lists as output by get_notes
    '''

    songs = []
    for song in notes:
        ins_grouped = []
        
        for instrument in song:
            grouped = group_chords(instrument, verbose)
            ins_grouped.append(grouped)
        
        songs.append(ins_grouped)

    return songs


def replace_with_roots(chords, verbose=0):
    
    songs = []
    for song in chords:
        ins_notes = []
        
        for instrument in song:
            notes = chords_to_roots(instrument, verbose)

            ins_notes.append(notes)
        
        songs.append(ins_notes)

    return songs


def chords_to_roots(notes, verbose=0):
    ''' 
    Input is a list of lists with notes. More than one item in a sublist identifies
    this sublist as a chord. Computes the root note of that chord. Returns a
    flat list of notes with time information from the original chord.
    '''

    notes_without_chords = []

    for item in notes:
        if len(item) > 1:
            # more than one item means it is a chord
      
            # get the pitches in clear text
            chord_pitches = []
            for note in item:
                chord_pitches.append(pretty_midi.note_number_to_name(note.pitch))
            
            # find the root pitch with a music21 function
            try:
                root_pitch = music21.chord.Chord(chord_pitches).root()
                if verbose == 2:
                    print(f'Root note of {chord_pitches} is {root_pitch}')
            except:
                if verbose:
                    print(f'Could not deduct root of {notes}')
            
            # if we identified one of the pitches as the root pitch
            # keep the corresponding note and forget the rest
            if root_pitch in item:
                root_note = item[item.index(root_pitch)]
                notes_without_chords.append(root_note)

        elif len(item) == 1:
            # it was just a normal note, not a chord
            notes_without_chords.append(item)
    
    return notes_without_chords


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


def transition_matrix_pms(long_list, max_interval=2.0):
    ''' 
        long_list has to be a list of pretty_midi.notes, including timing information.

        Creates a transition matrix to predict the probability of one pitch following another.
        The unique pitches are identified in the unique_items list, which decides the size of
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
        if long_list[pos+1].start_time - long_list[pos].end_time > max_interval:
            continue # skip the note if the interval to next note is too large
        
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

#TODO: Transition matrix with relative differences instead of absolute


def transition_matrix(long_list, normalize=True):
    ''' 
        Creates a transition matrix to predict the probability of one state following another.
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
    if normalize:
        unique_items_freq /= np.sum(unique_items_freq)
    
        # normalise transition matrix per row (convert to probability)
        for row in t_matrix:
            row /= np.sum(row)
        
    print(f'Done. Found {len(unique_items)} unique items.')

    return t_matrix, unique_items, unique_items_freq


def notes_name_cleaning(flat_notes):
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


def find_midis(path, max=10000, artist=''):
    '''
    Traverses path including subdirectories and makes a files_list 
    of up to max files. Filter on artist is optional. Returns files_list.
    '''
    files_list = []
    for (root,dirs,files) in os.walk(path, topdown=True):
        for file in files:
            if file[-4:] == '.mid':
                if len(artist) == 0 or artist in root:
                    files_list.append(os.path.join(root, file))
            
            if len(files_list) >= max:
                print(f'Found {len(files_list)} .mid files (Max limit reached)')
                return files_list

    print(f'Found {len(files_list)} .mid files')
    return files_list


def get_notes_list(parsed_midis):
    ''' Takes a list of parsed midi structures and extracts the notes.
        Returns one long list of notes.
    '''
    long_list = []
    for m in parsed_midis:
        notes, flat_notes, chords = get_notes(m)
        long_list += flat_notes
    
    return long_list


# Functions below use the music21 library for parsing midis and extracting notes.
# They are replaced by functions above, based on the pretty_midi library

def parse_midi_files_music21(files_list, max=100):
    ''' Parses the first (#max) midi files in files_list.
        Returns a list of midi structures.
    '''

    parsed_midi = []
    print(f'Parsing {min(len(files_list), max)} files')
    
    for c, file in enumerate(files_list[:min(len(files_list), max)]):
        parsed_midi.append(music21.converter.parse(file))
        if (c % 10 == 0):
            print(c, end='')
        elif (c % 5 == 0):
            print('.', end='')
        
    
    print('')

    return parsed_midi


def get_notes_music21(parsed_midi):
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
        s2 = music21.instrument.partitionByInstrument(parsed_midi)
        for i in range(len(s2)):
            notes_to_parse.append(s2.parts[i].recurse())
    except: # file has notes in a flat structure
        #print(f'Flat structure detected in {parsed_midi}')
        notes_to_parse.append(parsed_midi.flat.notes)
    
    for instrument in notes_to_parse:

        notes_row = []
        flat_notes_row = []
        chords_row = []

        # should add: if instrument.name is in [Strings, Guitars] or similar?
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