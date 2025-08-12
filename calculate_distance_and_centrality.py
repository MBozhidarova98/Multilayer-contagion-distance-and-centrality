# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:53:41 2025

@author: mbozhidarova
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 20 11:51:30 2025

@author: mbozhidarova
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 15:42:59 2025

@author: mbozhidarova
"""
import pandas as pd
import numpy as np
import networkx as nx

df = pd.read_csv('\\Trade_data_full.csv')

df = df[['exp', 'imp','t','gtrade','unified_sector']]
df = df.dropna()

#len(df) #Out[10]: 16374518
companies = set(df['exp'])
#Out[35]: 62 #so we have 62 countries to work with

# Remove the ones with 0.0 trade as we don't need it:
df = df[df['gtrade']!=0]
#[5680593 rows x 5 columns]

# Get separate dataset for each year:
yearly_data = dict()
for i in set(df['t']):
    yearly_data[i] = df[df['t']==i]

networks = dict()
for year in yearly_data.keys():
    dict_edges = dict()
    for sector in set(yearly_data[year]['unified_sector']):
        df_year = yearly_data[year][yearly_data[year]['unified_sector']==sector]
        # 1. Create ordered 'pair' (vectorized)
        df_year['i'] = df_year[['exp', 'imp']].min(axis=1)
        df_year['j'] = df_year[['exp', 'imp']].max(axis=1)
        
        # 2. Aggregate total trade between each pair
        trade_between_pairs = df_year.groupby(['i', 'j'], as_index=False)['gtrade'].sum()
        
        # 3. Total trade per country (exports + imports)
        trade_as_exp = df_year.groupby('exp')['gtrade'].sum()
        trade_as_imp = df_year.groupby('imp')['gtrade'].sum()
        total_trade = (trade_as_exp.add(trade_as_imp, fill_value=0)).to_dict()
        
        # 4. Map total trade for i and j
        trade_between_pairs['total_trade_i'] = trade_between_pairs['i'].map(total_trade)
        trade_between_pairs['total_trade_j'] = trade_between_pairs['j'].map(total_trade)
        
        # 5. Compute weights
        trade_between_pairs['weight_i_to_j'] = trade_between_pairs['gtrade'] / trade_between_pairs['total_trade_j']
        trade_between_pairs['weight_j_to_i'] = trade_between_pairs['gtrade'] / trade_between_pairs['total_trade_i']
        
        # 6. Build two arrays for edges: one for (i → j), one for (j → i)
        edges_i_j = trade_between_pairs[['i', 'j', 'weight_i_to_j']].to_numpy()
        edges_j_i = trade_between_pairs[['j', 'i', 'weight_j_to_i']].to_numpy()
        
        # 7. Stack them vertically
        edges = np.vstack([edges_i_j, edges_j_i])
        
        # 8. Convert to list of tuples 
        edges_list = [tuple(x) for x in edges]
        
        # 9. Store
        dict_edges[sector] = edges_list

    networks[year] = dict_edges

for year in networks.keys():
    for sector in set(yearly_data[year]['unified_sector']):
        print(len(networks[year][sector]))


########### Do network analysis for each year ##############
#Calculate Contagion distance:
#First make a single-layered network by taking the mean of the weight on each layer
for year in networks.keys():
    for sector in networks[year].keys():
        networks[year][sector] =  [(i, j, 1 - np.log(weight)) for i,j,weight in networks[year][sector]]

# Get network for each year for each sector
graphs_per_sector = dict()
for year in networks.keys():
    dict_graphs = dict()
    for sector in set(df['unified_sector']):
        edges = networks[year][sector]
        G = nx.DiGraph()
        G.add_weighted_edges_from(edges)
        dict_graphs[sector] = G
    graphs_per_sector[year] = dict_graphs

#Overlap networks per year
overlapped_networks = dict()
for year in networks.keys():
    all_data=[]
    for sector in networks[year].keys():
        all_data += networks[year][sector]
    min_values={}
    for a, b, value in all_data:
        key = (a, b)
        if key not in min_values or value < min_values[key]:
            min_values[key] = value
    result = [(a, b, value) for (a, b), value in min_values.items()]
    overlapped_networks[year] = result


graphs = dict()
for year in overlapped_networks.keys():
    edges = overlapped_networks[year]
    G = nx.DiGraph()
    G.add_nodes_from(companies)
    G.add_weighted_edges_from(edges)
    graphs[year] = G

def contagion_distance(G,i,j):
    if nx.has_path(G,i,j):
        cd = nx.shortest_path_length(G,i,j,weight='weight')
    else:
        cd=np.inf
    return(cd)


def contagion_distance_all(G):
    dist_dict = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    
    records = dict()
    for i, targets in dist_dict.items():
        for j, d in targets.items():
            if i != j:  
                records[(i, j)]= d
    return(records)

distances_over_time=dict()
#let's do for time 0:
for year in graphs.keys():
    G=graphs[year]
    dist=contagion_distance_all(G)
    distances_over_time[year]=dist
    
df = pd.DataFrame(distances_over_time)
df.reset_index(inplace=True)
df.rename(columns={'level_0': 'Company1','level_1':'Company2'}, inplace=True)
df.to_csv('Contagion_distances_per_year.csv',index=False)

############### Calculate contagion centrality ############
def mu(G,i,dist):
    s=0
    cntr=0
    for j in G.nodes():
        if i!=j:
            if (i,j) in dist.keys():
                s=s+dist[(i,j)]
                cntr=cntr+1
    if cntr==0:
        return np.inf
    return(s/cntr)

def sigma(G,i,dist):
    s=0
    cntr=0
    for j in G.nodes():
        if i!=j:
            if (i,j) in dist.keys():
                s=s+(dist[(i,j)]-mu(G,i,dist))**2
                cntr=cntr+1
    if cntr==0 or cntr==1:
        return np.inf
    return(np.sqrt(s/(cntr-1)))

def contagion_centrality(G,i,dist):
    return(1/(np.sqrt(mu(G,i,dist)**2+sigma(G,i,dist)**2)))

centralities_over_time=dict()
for year in distances_over_time.keys():
    dist=distances_over_time[year]
    G = graphs[year]
    centralities=dict()
    for i in G.nodes:
        centralities[i]=contagion_centrality(G,i,dist)
    centralities_over_time[year]=centralities


df = pd.DataFrame(centralities_over_time)
df.reset_index(inplace=True)
df.rename(columns={'index': 'Company'}, inplace=True)
df.to_csv('Contagion_centrality_per_year.csv',index=False)

#Do per sector:
distances_over_time=dict()
#let's do for time 0:
for year in graphs_per_sector.keys():
    for sector in graphs_per_sector[year]:
        G=graphs_per_sector[year][sector]
        dist=contagion_distance_all(G)
        distances_over_time[(year,sector)]=dist
        
df = pd.DataFrame(distances_over_time)
df.reset_index(inplace=True)
df.rename(columns={'level_0': 'Company1','level_1':'Company2'}, inplace=True)
df.to_csv('Contagion_distances_per_year_per_sector.csv',index=False)

#Calculate contagion centralities per year per sector
centralities_over_time=dict()
for year in graphs_per_sector:
    for sector in graphs_per_sector[year]:
        dist=distances_over_time[(year,sector)]
        G = graphs_per_sector[year][sector]
        centralities=dict()
        for i in G.nodes:
            centralities[i]=contagion_centrality(G,i,dist)
        centralities_over_time[(year,sector)]=centralities


df = pd.DataFrame(centralities_over_time)
df.reset_index(inplace=True)
df.rename(columns={'index': 'Company'}, inplace=True)
df.to_csv('Contagion_centrality_per_year_per_sector.csv',index=False)
    