import os
import time
import mido
import ipywidgets
import warnings
import json

import numpy as np
import music21 as m21

from collections import OrderedDict
from collections.abc import Iterable
from IPython import display

## TODO: Docstrings

OUTPUT = ipywidgets.Output()

def load_js_modules(*filepaths):
    for _filepath in filepaths:
        with open(_filepath, 'r') as _jscript:
            code = _jscript.read()
        with OUTPUT:
            display.display(display.Javascript(code))


def setup_soundfonts(path='./js_src/'):
    display.display(OUTPUT)
    load_js_modules(path+'tone.min.js', path+'soundfont-player.min.js')
    with OUTPUT:
        display.display(display.HTML(f'''<script> 
        if (typeof ac == 'undefined' || ac == null){{
            var ac = new AudioContext();
            var init_t = Date.now() / 1000 - ac.currentTime;
        }}
        </script>'''))
        display.display(display.HTML(f'''<script> 
        var shift_t = Date.now() / 1000 - {time.time()};
        </script>'''))
        display.display(display.Javascript(f'console.log(shift_t)'))
        
    OUTPUT.clear_output(wait=True)
    return OUTPUT
        

def get_fluidnames():
    with open('fluidnames.json') as infile:
        names = json.load(infile)    
    return names


class SFInstrument:
    
    _memodict = {}
    _defkwargs = {'instrument':'acoustic_grand_piano', 'soundfont':'FluidR3_GM'}
    _fluidnames = get_fluidnames()
    
    def __new__(cls, *args, **kwargs):
        kwargs = cls.filter_kwargs(**kwargs)
        id_kw = tuple(kwargs.items())
        if id_kw not in cls._memodict:
            self = super(SFInstrument, cls).__new__(cls)
            self.__dict__.update(kwargs)
            cls._memodict[id_kw] = self
        return cls._memodict[id_kw]
    
    @staticmethod
    def filter_kwargs(**kwargs):
        if 'instrument' in kwargs:
            msg = f'''{kwargs["instrument"]} is not a valid instrument!
            Check SFInstrument.valid_instruments for list of available instruments.
            '''
            assert kwargs['instrument'] in SFInstrument._fluidnames, msg
            
        defaults = SFInstrument._defkwargs
        outkwargs = {k:v for k,v in defaults.items()}
        kwargs = {k:v for k,v in kwargs.items() if k in defaults}
        outkwargs.update(kwargs)
        return outkwargs
    
    def __init__(self, *args, **kwargs):
        src = f'''
        <script>
        var sf_{id(self)} = Soundfont.instrument(
            ac, "{self.instrument}", {{ soundfont: "{self.soundfont}" }}
        );
        </script>
        '''
        with OUTPUT:
            display.display(display.HTML(src))
            
    @property
    def valid_instruments(self):
        return self._fluidnames
            
    def set_ac_params(self):
        src = f'''
        <script>
        var ac_start = Date.now()/1000-init_t;
        var ac_shift = ac_start-ac.currentTime-shift_t;
        </script>
        '''
        with OUTPUT:
            display.display(display.HTML(src))
        
    
    def play(self, *events, from_start=True):
        if from_start:
            self.set_ac_params()
        notestr = ""
        ct = 'ac_start-ac_shift'
        for event in events:
            if not isinstance(event, Iterable):
                notestr += f'{self.instrument}.play({event});'
            else:
                if len(event) == 1:
                    n = event[0]
                    notestr += f'{self.instrument}.play({n});'
                elif len(event) == 2:
                    n, t = event
                    notestr += f'{self.instrument}.play({n}, {t}+{ct});'
                elif len(event) == 3:
                    n, t, d = event
                    notestr += f'{self.instrument}.play({n}, {t}+{ct}, {{ duration : {d} }});'
                else:
                    n, t, d, g = event
                    notestr += f'{self.instrument}.play({n}, {t}+{ct}, {{ duration : {d}, gain: {g} }});'
        
        src = f'sf_{id(self)}.then(function ({self.instrument}) {{ {notestr} }});'
        with OUTPUT:
            display.display(display.Javascript(src))
        OUTPUT.clear_output(wait=True)
                
        
    def stop(self):
        notestr = f'{self.instrument}.stop();'
        src = f'sf_{id(self)}.then(function ({self.instrument}) {{ {notestr} }});'
        with OUTPUT:
            display.display(display.Javascript(src))
        OUTPUT.clear_output(wait=True)
                        
            
            
            
class CustomMidiFile:
    
    q_den = 24
    max_dur = 16.0
    start_octave = 2
    end_octave = 10
        
    def __init__(self, fpath, normalize=True):
        self.fpath = fpath
        self.midi = mido.MidiFile(fpath)
        self.events = self.create_events()
        self.default_tempo = 6e5
        self.stream, self.original_key, self.new_key = self.create_stream(normalize)
        self.chord_dict = self.create_chord_dict()
        
        
    def quantize_roll(self, t, q_den, max_dur=None):
        if max_dur is not None:
            t = min(t, max_dur)
        tfrac = round((t % (1/q_den)) * q_den)
        return int(t // (1/q_den)) + tfrac

    def dct2roll(self, q_den=None, max_dur=None, start_octave=None, end_octave=None):
        # Not processed by default in init, as it is rather costly.
        # TODO: create a roll2dct, and roll2event function.
        if q_den is None:
            q_den = self.q_den
        if max_dur is None:
            max_dur = self.max_dur
        if start_octave is None:
            start_octave = self.start_octave
        if end_octave is None:
            end_octave = self.end_octave
        
        dct = self.chord_dict
        keylist = sorted(list(dct.keys()))
        lastkey = keylist[-1]
        lastdur = max(dct[lastkey]['duration'])
        
        # Generate piano roll: Finalize end up to a bar of 4 quarternotes
        tbar = 4*q_den
        tend = self.quantize_roll(lastkey + lastdur, q_den=q_den)
        tend += tbar - (tend % tbar)
        pstart, pend = 12*start_octave, 12*end_octave
        roll = np.zeros((2, pend - pstart, tend))
        
        for time in keylist:
            notes = np.array(dct[time]['note']) - pstart
            vel = dct[time]['velocity']
            durs = [
                self.quantize_roll(d, q_den=q_den, max_dur=max_dur) 
                for d in dct[time]['duration']
            ]
            time = self.quantize_roll(time, q_den=q_den)
            roll[0,notes,time] = 1
            for i in range(len(notes)):
                roll[1,notes[i],time:time + durs[i]] = vel[i] / 128

            # Disconnect connected note_on sections
            wheres = np.where(roll[0] > 0)
            roll[1, wheres[0], wheres[1]-1] = 0
            
        return roll
        
    def create_chord_dict(self):
        chord_dict = {}
        for event in self.events:
            cur_t = event['time']
            if cur_t not in chord_dict:
                chord_dict[cur_t] = {'note': [], 'duration': [], 'velocity': [], 'channel': []}
            for key in chord_dict[cur_t]:
                val = event[key]
                chord_dict[cur_t][key].append(val)

        return chord_dict
                
    def create_stream(self, normalize):
        stream = m21.stream.Stream()
        for event in self.events:
            note = m21.note.Note(event['note'])
            note.offset = event['time']
            note.duration.quarterLength = event['duration']
            stream.append(note)
        
        key = stream.analyze('key')
        mode = -3*int(key.mode == 'minor')
        shift = -(key.tonic.midi % 12) + mode
        if shift < -6:
            shift += 12
           
        if normalize:
            newkey = key.transpose(shift)
            stream = stream.transpose(shift)
            for event in self.events:
                event['note'] += shift
        else:
            newkey = key
                
        return stream, key, newkey
    
    def create_events(self):
        mfdict = {}
        mfevents = []
        minnote, maxnote = 128, 0
        for track in self.midi.tracks:
            cur_t = 0.0
            for msg in track:
                if msg.type == 'note_on' or msg.type == 'note_off':                    
                    d = msg.dict()
                    mtype = msg.type
                    # Fix: note_on with vel=0 is treated as note_off
                    if d['velocity'] == 0 and msg.type == 'note_on':
                        mtype = 'note_off'
                    if d['channel'] not in mfdict:
                        mfdict[d['channel']] = {}
                    if d['note'] not in mfdict[d['channel']]:
                        mfdict[d['channel']][d['note']] = {}
                    if mtype not in mfdict[d['channel']][d['note']]:
                        mfdict[d['channel']][d['note']][mtype] = []
                    
                    if d['note'] < minnote:
                        minnote = d['note']
                    if d['note'] > maxnote:
                        maxnote = d['note']
                    
                    cur_t += d['time']
                    mfdict[d['channel']][d['note']][mtype].append((cur_t, d['velocity']))
                else:
                    cur_t += msg.dict()['time']
        
        shift_octave = 0
        if minnote < 24:
            shift_octave = 1
        if minnote < 12:
            shift_octave = 2
        if maxnote > 108:
            if shift_octave == 0:
                shift_octave = -1
            else:
                import warnings
                warnings.warn(f'Song {self.fpath} exceeds max octave for piano.')
                                
        for c in mfdict.keys():
            for n in mfdict[c].keys():
                for on, vel in mfdict[c][n]['note_on']:
                    for off, _ in mfdict[c][n]['note_off']:
                        if off > on:
                            time = self.tick2qn(on)
                            dur = self.tick2qn(off-on)
                            mfevents += [{
                                'note': n + shift_octave * 12, 
                                'time': time,
                                'duration': dur, 
                                'velocity': vel, 
                                'channel': c}
                            ]
                            break
                        else:
                            pass

        mfevents.sort(key=lambda x: x['time'])
        return mfevents
            
    def get_octaves(self):
        octaves = []
        for event in mf.events:
            octave = event['note'] // 12
            octaves.append(octave)      
        return octaves
        
    def tick2qn(self, tick):
        return tick / self.midi.ticks_per_beat
    
    def qn2tick(self, qn):
        return self.midi.ticks_per_beat * qn
    
    def get_sec_scale(self, ticks_per_beat, tempo):
        if tempo is None:
            tempo = self.default_tempo
        if ticks_per_beat is None:
            ticks_per_beat = self.midi.ticks_per_beat
        return tempo * 1e-6 / ticks_per_beat
        
    def tick2sec(self, tick, ticks_per_beat=None, tempo=None):
        scale = self.get_sec_scale(ticks_per_beat, tempo)
        return tick * scale
    
    def qn2sec(self, qn, ticks_per_beat=None, tempo=None):
        return self.tick2sec(self.qn2tick(qn), ticks_per_beat, tempo)
    
    def sec2tick(self, sec, ticks_per_beat=None, tempo=None):
        scale = self.get_sec_scale(ticks_per_beat, tempo)
        return sec / scale

    def sec2qn(self, sec, ticks_per_beat=None, tempo=None):
        return self.tick2qn(self.sec2tick(sec, ticks_per_beat, tempo))
                
    def play(self, sf, buffer=8, verbose=0, bpm=None, max_dur=4.):       
        if bpm is not None:
            tempo = int(round((60 * 1000000) / bpm))
        else:
            tempo = None
        
        time_offset = time.time()
        cur_notes = []
        first = True
        for event in self.events:
            etime = self.qn2sec(event['time'], tempo=tempo)
            edur = self.qn2sec(event['duration'], tempo=tempo)
            edur = min(edur, max_dur)
            cur_notes.append((
                event['note'], 
                etime,
                edur, 
                event['velocity'] / 128
            ))

            ctime = time.time() - time_offset + buffer
            if etime <= ctime:
                if verbose:
                    print('Playing:', cur_notes, 'ctime:', ctime, 'etime:', etime)
                sf.play(*cur_notes, from_start=first)
                first = False
                cur_notes = []
            else:
                if verbose:
                    print('Sleeping:', edur, 'ctime:', ctime, 'etime:', etime)
                # It is possible to miss notes here, safer to check next event?
                time.sleep(edur)