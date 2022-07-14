import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def set_edges(df, key, edge_list):
    df[key] = [1.0 if item in edge_list else 0.0 for item in df.index]
    return df

def create_tonnetz_adjacency(chromatic_scale):
    c_edges = ['A', 'F', 'E', 'G', 'D#', 'G#']
    csh_edges = ['A', 'F#', 'A#', 'F', 'G#', 'E']
    d_edges = ['A#', 'G', 'B', 'F#', 'A', 'F']
    dsh_edges = ['B', 'G#', 'C', 'G', 'A#', 'F#']
    e_edges = ['C', 'A', 'C#', 'G#', 'B', 'G']
    f_edges = ['C#', 'A#', 'D', 'A', 'C', 'G#']
    fsh_edges = ['D', 'B', 'D#', 'A#', 'C#', 'A']
    g_edges = ['D#', 'C', 'E', 'B', 'D', 'A#']
    gsh_edges = ['E', 'C#', 'F', 'C', 'D#', 'B']
    a_edges = ['F', 'D', 'F#', 'C#', 'E', 'C']
    ash_edges = ['F#', 'D#', 'G', 'D', 'F', 'C#']
    b_edges = ['G', 'E', 'G#', 'D#', 'F#', 'D']
    edges = [c_edges, csh_edges,
             d_edges, dsh_edges,
             e_edges,
             f_edges, fsh_edges,
             g_edges, gsh_edges,
             a_edges, ash_edges,
             b_edges]
    
    tonnetz = np.zeros(shape=(len(chromatic_scale), len(chromatic_scale)))
    tonnetz_df = pd.DataFrame(tonnetz, columns=chromatic_scale, index=chromatic_scale)
        
    for note, note_edges in zip(chromatic_scale, edges):
        tonnetz_df = set_edges(tonnetz_df, note, note_edges)
    return tonnetz_df.to_numpy()


def create_graph():
    chromatic_scale = ['C', 'C#',
                   'D', 'D#',
                   'E',
                   'F', 'F#',
                   'G', 'G#',
                   'A', 'A#',
                   'B']

    # delta time is the interval in bars which the note data will be sampled
    delta_time = 0.25

    # P is the number of time steps (size delta time) in the past we will use
    P = int(4/delta_time)






    # tonnetz
    tonnetz_adj = create_tonnetz_adjacency(chromatic_scale)
    tonnetz_G = nx.from_numpy_matrix(tonnetz_adj)
    tonnetz_G = nx.relabel_nodes(tonnetz_G, lambda x: chromatic_scale[x])
    print(tonnetz_adj.shape)

    # time graph
    time_G = nx.path_graph(P)
    time_G = nx.relabel_nodes(time_G, lambda x: str(int(x)-P+1))
    time_adj = np.array(nx.adjacency_matrix(time_G).todense())
    print(time_adj.shape)

    # time extended tonnetz
    tonnetz_df = pd.DataFrame(tonnetz_adj, columns=chromatic_scale, index=chromatic_scale)
    time_df = pd.DataFrame(time_adj, columns=list(time_G.nodes), index=list(time_G.nodes))
    extended_G = nx.cartesian_product(tonnetz_G, time_G)  # calculate extended graph as cartesian product
    extended_adj = np.array(nx.adjacency_matrix(extended_G).todense())
    print(extended_adj.shape)





    # amend node labels for easy data processing later
    cp = pd.DataFrame(extended_adj, columns=pd.MultiIndex.from_product([tonnetz_df, time_df]))
    cp.columns = cp.columns.get_level_values(0) + '_' +  cp.columns.get_level_values(1).astype(str)
    cp.index = cp.columns
    new_labels = dict(zip(extended_G.nodes, cp.columns))
    extended_G = nx.relabel_nodes(extended_G, new_labels)




    '''
    # visualise extended graph to be used in GNN
    plt.figure(num=None, figsize=(30, 30))
    nx.draw(extended_G, with_labels=True)
    '''
    return extended_G
