import numpy as np
#from music21 import *
import music21
import pdb
import torch
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from sklearn.preprocessing import MultiLabelBinarizer

import set_graph_alexander as alex_graph
path = '~/datasets_graphs/maestro-v3.0.0-midi/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'



def normalize_to_c_maj(midi_file):
    '''this is taken from http://nickkellyresearch.com/python-script-transpose-midi-files-c-minor/ '''
    #majors = dict([('A-', 4),('A', 3),('B-', 2),('B', 1),('C', 0),('D-', -1),('D', -2),('E-', -3),('E', -4),('F', -5),('G-', 6),('G', 5)])

    #minors = dict([('A-', 1),('A', 0),('B-', -1),('B', -2),('C', -3),('D-', -4),('D', -5),('E-', 6),('E', 5),('F', 4),('G-', 3),('G', 2)])


    majors = dict([('A-', 4),('G#', 4),('A', 3),('A#', 2),('B-', 2),('B', 1),('C', 0),('C#', -1),('D-', -1),('D', -2),('D#', -3),('E-', -3),('E', -4),('F', -5),('F#', 6),('G-', 6),('G', 5)])

    minors = dict([('G#', 1), ('A-', 1),('A', 0),('A#', -1),('B-', -1),('B', -2),('C', -3),('C#', -4),('D-', -4),('D', -5),('D#', 6),('E-', 6),('E', 5),('F', 4),('F#', 3),('G-', 3),('G', 2)])

    key = midi_file.analyze('key')
    print(key)

    if key.mode == 'major':
        halfSteps = majors[key.tonic.name]
    if key.mode == 'minor':
        halfSteps = minors[key.tonic.name]

    newscore = midi_file.transpose(halfSteps)
    key = newscore.analyze('key')
    print('new key: ', key)
    #newscore.show()
    #normalized_fileName = 'c_file_name'
    #newscore.write('midi', normalized_fileName)
    return newscore

def project_to_single_octave():
    pass

def midi_to_bars():
    pass

def window_to_chord():
    pass

def extract_quarter(bar_list, amount_quarter = 4):
    bars_in_quarter = []
    for i in range(amount_quarter):
        quarter_list = [chord[0] for chord in bar_list if (chord[1]<i+1 and chord[2]>=i)]
        if len(quarter_list)> 0:
            quarter_list = list(set( [noty for note_lis in quarter_list for noty in note_lis]))
        else:
            quarter_list = []
        bars_in_quarter.append(quarter_list)
    return bars_in_quarter

def get_node_features(output_tensor, chromatic_scale=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']):
    node_features = pd.DataFrame()
    for feature in range(0, output_tensor.shape[-1]):
        # put into nodes x features format
        tmp = pd.DataFrame(output_tensor[:, :, feature].numpy(), columns=chromatic_scale, index=np.arange(-output_tensor.shape[0]+1, 1, 1))
        tmp_stack = tmp.T.stack()

        # get new indices
        new_ind = tmp_stack.index.get_level_values(0) + '_' +  tmp_stack.index.get_level_values(1).astype(str)
        tmp_stack.index = new_ind

        # append into dataframe column for given feature
        node_features[feature] = tmp_stack
    return torch.tensor(node_features.values)

def midi_to_vectors(midi_file):
    midi_file_chord = midi_file.chordify()
    chord_list = []
    for bar in midi_file_chord.measures(1, 4):#len(midi_file_chord) is max amount of bars
        bar_list = []
        for chord in bar.getElementsByClass('Chord'):
            if chord.isNote:
                print('ERROR! NOT A CHORD')
            elif chord.isChord:
                chord_encoded = ([note.midi for note in chord.pitches], chord.offset, chord.offset + chord.duration.quarterLength)
                bar_list.append(chord_encoded)
        bars_in_quarter = extract_quarter(bar_list)
        chord_list.append(bars_in_quarter)
    #return chord_list

    segment_list = []
    for bar in chord_list:
        bar_tensor= []
        for timestep in bar:
            a = np.array(timestep)%12
            mlb= MultiLabelBinarizer(classes= list(range(12)))
            multi_hot = mlb.fit_transform([a]) 
            signal = [[el] for el in multi_hot[0]]
            signal_tensor = torch.tensor(signal, dtype = torch.float)
            #print(signal_tensor)
            #pdb.set_trace()
            bar_tensor.append(signal_tensor)
        segment_list.append(torch.stack(bar_tensor))
    output_tensor = torch.cat(segment_list) #dim: 16,12,1

    # get node features
    node_features = get_node_features(output_tensor)
    return node_features

if __name__ == '__main__':
    token_midi = music21.converter.parse(path)
    #normalize_to_c_maj(token_midi)
    #token_midi.measure(4).show('text')

    G = alex_graph.create_graph()
    print(G.nodes)
    for (n1, n2, d) in G.edges(data=True):
        d.clear()
    #print(G.edges(data=True))
    #pdb.set_trace()
    pyg_G = from_networkx(G)
    #print(pyg_G)

    output_tensor = midi_to_vectors(token_midi)

    print(output_tensor.size())

    #pdb.set_trace()
    pyg_G.x = output_tensor
    print(pyg_G.x)
    
    '''
    segment_list = []
    for bar in output_list:
        bar_tensor= []
        for timestep in bar:
            a = np.array(timestep)%12
            mlb= MultiLabelBinarizer(classes= list(range(12)))
            multi_hot = mlb.fit_transform([a]) 
            signal = [[el] for el in multi_hot[0]]
            signal_tensor = torch.tensor(signal, dtype = torch.float)
            #print(signal_tensor)
            #pdb.set_trace()
            bar_tensor.append(signal_tensor)
        segment_list.append(torch.stack(bar_tensor))
    torch.cat(segment_list)
    '''

    #data = Data(x=signal_tensor, edge_index = pyg_G.edges)

#TODO check 4/4 or not -> do soft encoding
#TODO sliding window to slice
#desired output: (files, bar/4 ,16,12, 1)
#desired output: (files, bar/4 ,192, 1)
#input to GNN: (1239781, 192, node_features)
# desired parameters: amout_timesteps = 16?




