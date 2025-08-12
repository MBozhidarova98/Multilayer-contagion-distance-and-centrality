# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:56:30 2025

@author: mbozhidarova
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

df = pd.read_csv('Contagion_centrality_per_year_per_sector.csv')

#Extract the sector labels from the first row
sector_row = df.iloc[0]
df = df.drop(index=0)

#Melt the DataFrame to long format
df_long = df.melt(id_vars='Company', var_name='Col', value_name='Centrality')

#Map columns to sector and year
df_long['Sector'] = df_long['Col'].map(sector_row)
df_long['Year'] = df_long['Col'].apply(lambda x: str(x).split('.')[0] if '.' in str(x) else str(x))

#Clean and convert data types
df_long = df_long.dropna(subset=['Centrality', 'Sector', 'Year'])
df_long['Centrality'] = pd.to_numeric(df_long['Centrality'], errors='coerce')
df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce').astype(int)

df_long['Sector'] = df_long['Sector'].replace({
    '14-Finacial Intermediation and Business Activity': '14-Finacial Intermediation\n and Business Activity',
    '6-Petroleum, Chemical and Non-Metallic Minerals': '6-Petroleum, Chemical\n and Non-Metallic Minerals',
})
mean_centrality_per_company_year = df_long.groupby(['Company', 'Year'])['Centrality'].mean().reset_index()
mean_centrality_per_company_year.rename(columns={'Centrality': 'Mean Centrality'}, inplace=True)

df_pivoted = mean_centrality_per_company_year.pivot_table(index='Company', columns='Year', values='Mean Centrality', aggfunc='mean')

# Now get the multiplex contagion centrality:
df = pd.read_csv('Contagion_centrality_per_year.csv')
df = df.dropna() #use only the ones for which we have data

df = df[~df['Company'].isin(['ROW'])]
df.set_index('Company', inplace=True)
df.columns = df.columns.astype(int)

######### Find if the rankings are similar ###############
common_companies = df.index.intersection(df_pivoted.index)

# Subset both DataFrames to common companies
df = df.loc[common_companies]
df_pivoted = df_pivoted.loc[common_companies]

# Rank each year (column) for both DataFrames

ranks_multilevel = df.rank(axis=0, method='min', ascending=False)
ranks_aggregated = df_pivoted.rank(axis=0, method='min', ascending=False)

# Compare top-k centrality ranking
ks = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,62]
# Initialize containers for results
overlap_results = pd.DataFrame(index=df.columns, columns=ks, dtype=float)
jaccard_results = pd.DataFrame(index=df.columns, columns=ks, dtype=float)

for year in df.columns:
    for k in ks:
        # Get top-k indices (lowest rank means most central)
        top_k_agg = ranks_aggregated[year].nsmallest(k).index
        top_k_multi = ranks_multilevel[year].nsmallest(k).index
        
        # Calculate overlap
        intersection_size = len(set(top_k_agg).intersection(set(top_k_multi)))
        overlap = intersection_size / k
        
        # Calculate Jaccard index: intersection / union
        union_size = len(set(top_k_agg).union(set(top_k_multi)))
        jaccard = intersection_size / union_size if union_size > 0 else np.nan
        
        # Store results
        overlap_results.at[year, k] = overlap
        jaccard_results.at[year, k] = jaccard

# Convert index to int for better plotting if years are numeric
try:
    overlap_results.index = overlap_results.index.astype(int)
    jaccard_results.index = jaccard_results.index.astype(int)
except:
    pass

# Plotting heatmaps
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.heatmap(overlap_results.astype(float), annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Top-k Overlap'})
plt.title("Top-k Overlap Between Rankings")
plt.xlabel("Top k")
plt.ylabel("Year")

plt.subplot(1, 2, 2)
sns.heatmap(jaccard_results.astype(float), annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'Jaccard Index'})
plt.title("Top-k Jaccard Index Between Rankings")
plt.xlabel("Top k")
plt.ylabel("Year")

plt.tight_layout()
plt.show()

#Do the same but for contagion distances: 
df_cd = pd.read_csv('Contagion_distances_per_year_per_sector.csv')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sector_row = df_cd.iloc[0][2:]
df_cd = df_cd.drop(index=0)

df = df_cd.copy()

# Create a mapping of columns to base years
year_columns = df.columns[2:]  # Skip 'Company1' and 'Company2'
year_map = year_columns.to_series().apply(lambda x: str(x).split('.')[0])


# Create a dictionary: year -> list of columns belonging to that year
from collections import defaultdict

year_groups = defaultdict(list)
for col, year in year_map.items():
    year_groups[year].append(col)

# Compute mean per year across all sectors
mean_per_year = pd.DataFrame()
mean_per_year[['Company1', 'Company2']] = df[['Company1', 'Company2']]

for year, cols in year_groups.items():
    mean_per_year[year] = df[cols].astype(float).mean(axis=1)

mean_per_year.columns = ['Company1', 'Company2'] + sorted([str(y) for y in year_groups.keys()])

df = pd.read_csv('Contagion_distances_per_year.csv')

df = df.dropna() #use only the ones for which we have data

df = df[~df['Company1'].isin(['ROW']) & ~df['Company2'].isin(['ROW'])]


# 1. Align on common pairs
common_pairs = pd.merge(
    df[['Company1', 'Company2']],
    mean_per_year[['Company1', 'Company2']],
    on=['Company1', 'Company2']
)

# Merge in the values from both
df_aligned = pd.merge(common_pairs, df, on=['Company1', 'Company2'])
mean_aligned = pd.merge(common_pairs, mean_per_year, on=['Company1', 'Company2'])

# Extract year columns
year_cols = [col for col in df.columns if col not in ['Company1', 'Company2']]
year_cols = sorted(year_cols, key=int)  # sort in chronologic order

# Define % thresholds
percent_thresholds = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Prepare results
overlap_results = pd.DataFrame(index=year_cols, columns=percent_thresholds, dtype=float)
jaccard_results = pd.DataFrame(index=year_cols, columns=percent_thresholds, dtype=float)

# Compute for each year and threshold
for year in year_cols:
    # Get values and rank (lower = more contagious)
    df_year = df_aligned[['Company1', 'Company2', year]].copy()
    mean_year = mean_aligned[['Company1', 'Company2', year]].copy()

    # Drop nans
    df_year = df_year.dropna()
    mean_year = mean_year.dropna()

    # Ensure consistent pairs after dropping nans
    merged = pd.merge(df_year, mean_year, on=['Company1', 'Company2'], suffixes=('_agg', '_mean'))
    
    # Rank
    merged['rank_agg'] = merged[f'{year}_agg'].rank(method='min', ascending=True)
    merged['rank_mean'] = merged[f'{year}_mean'].rank(method='min', ascending=True)

    total_pairs = len(merged)

    for pct in percent_thresholds:
        k = int(np.ceil(pct / 100 * total_pairs))

        top_agg = set(merged.nsmallest(k, 'rank_agg')[['Company1', 'Company2']].apply(tuple, axis=1))
        top_mean = set(merged.nsmallest(k, 'rank_mean')[['Company1', 'Company2']].apply(tuple, axis=1))

        # Overlap and Jaccard
        intersection = top_agg & top_mean
        union = top_agg | top_mean

        overlap_results.at[year, pct] = len(intersection) / k
        jaccard_results.at[year, pct] = len(intersection) / len(union) if union else np.nan


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()

sns.heatmap(jaccard_results.astype(float), annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Jaccard Index'})
plt.xlabel("Top $x$% of pairs with lowest contagion distance",fontsize=12)
plt.ylabel("Year",fontsize=12)
plt.tight_layout()
plt.savefig('Jaccard_index_contagion_distances.pdf',bbox_inches = 'tight')
plt.show()


################### Do a qq plot of the contagion distances and centralities ######
df = pd.read_csv('Contagion_centrality_per_year_per_sector.csv')

# Extract sector labels 
sector_row = df.iloc[0]
df = df.drop(index=0)

# Melt df to long format
df_long = df.melt(id_vars='Company', var_name='Col', value_name='Centrality')

# Map columns to sector and year
df_long['Sector'] = df_long['Col'].map(sector_row)
df_long['Year'] = df_long['Col'].apply(lambda x: str(x).split('.')[0] if '.' in str(x) else str(x))

# Clean and convert data types
df_long = df_long.dropna(subset=['Centrality', 'Sector', 'Year'])
df_long['Centrality'] = pd.to_numeric(df_long['Centrality'], errors='coerce')
df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce').astype(int)

df_long['Sector'] = df_long['Sector'].replace({
    '14-Finacial Intermediation and Business Activity': '14-Finacial Intermediation\n and Business Activity',
    '6-Petroleum, Chemical and Non-Metallic Minerals': '6-Petroleum, Chemical\n and Non-Metallic Minerals',
})
mean_centrality_per_company_year = df_long.groupby(['Company', 'Year'])['Centrality'].mean().reset_index()
mean_centrality_per_company_year.rename(columns={'Centrality': 'Mean Centrality'}, inplace=True)

df_pivoted = mean_centrality_per_company_year.pivot_table(index='Company', columns='Year', values='Mean Centrality', aggfunc='mean')

# Now get the multiplex contagion centrality:
df = pd.read_csv('Contagion_centrality_per_year.csv')
df = df.dropna() #use only the ones for which we have data

df = df[~df['Company'].isin(['ROW'])]
df.set_index('Company', inplace=True)
df.columns = df.columns.astype(int)

# Choose the year to compare
year = 2000

# Extract centrality values for that year
vals1 = df[year]
vals2 = df_pivoted[year]

# Ensure matching indices
common_companies = df.index.intersection(df_pivoted.index)
vals1 = vals1.loc[common_companies]
vals2 = vals2.loc[common_companies]

# Rank the values (lower rank = higher centrality)
ranks1 = vals1.rank(ascending=False)
ranks2 = vals2.rank(ascending=False)

# Q–Q Plot of rankings
plt.figure(figsize=(8, 6))
plt.scatter(ranks1, ranks2, alpha=0.7,s=50)
plt.plot([ranks1.min(), ranks1.max()], [ranks1.min(), ranks1.max()], 'r--')
plt.xlabel('Ranking in multilayer contagion centrality',fontsize=14)
plt.ylabel('Ranking in aggregated contagion centrality',fontsize=14)
plt.tight_layout()
plt.show()


# Define multiple years to compare
years = [2000, 2010, 2020]

# Set up the figure and axes
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

for ax, year in zip(axes, years):
    # Extract centrality values for the year
    vals1 = df[year]
    vals2 = df_pivoted[year]

    # Match company indices
    common_companies = df.index.intersection(df_pivoted.index)
    vals1 = vals1.loc[common_companies]
    vals2 = vals2.loc[common_companies]

    # Compute rankings (lower rank = higher centrality)
    ranks1 = vals1.rank(ascending=False)
    ranks2 = vals2.rank(ascending=False)

    # Q–Q scatter plot
    ax.scatter(ranks1, ranks2, s=60, alpha=0.7)
    ax.plot([ranks1.min(), ranks1.max()], [ranks1.min(), ranks1.max()], 'r--')
    ax.set_xlabel('Rank with multilayer contagion centrality',fontsize=12)
    if ax == axes[0]:
        ax.set_ylabel('Ranking with aggregated contagion centrality',fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('QQ_plot_rankings.pdf',bbox_inches = 'tight')
plt.show()


#Now do qq plot for the contagion distances:
df_cd = pd.read_csv('Contagion_distances_per_year_per_sector.csv')

sector_row = df_cd.iloc[0][2:]
df_cd = df_cd.drop(index=0)
df = df_cd.copy()
year_columns = df.columns[2:]
year_map = year_columns.to_series().apply(lambda x: str(x).split('.')[0])

year_groups = defaultdict(list)
for col, year in year_map.items():
    year_groups[year].append(col)

# Compute mean per year across all sectors
mean_per_year = pd.DataFrame()
mean_per_year[['Company1', 'Company2']] = df[['Company1', 'Company2']]

for year, cols in year_groups.items():
    mean_per_year[year] = df[cols].astype(float).mean(axis=1)

mean_per_year.columns = ['Company1', 'Company2'] + sorted([str(y) for y in year_groups.keys()])

df = pd.read_csv('Contagion_distances_per_year.csv')

df = df.dropna() #use only the ones for which we have data

df = df[~df['Company1'].isin(['ROW']) & ~df['Company2'].isin(['ROW'])]


# Align on common pairs
common_pairs = pd.merge(
    df[['Company1', 'Company2']],
    mean_per_year[['Company1', 'Company2']],
    on=['Company1', 'Company2']
)

# Merge in the values from both
df_aligned = pd.merge(common_pairs, df, on=['Company1', 'Company2'])
mean_aligned = pd.merge(common_pairs, mean_per_year, on=['Company1', 'Company2'])

# Define years to compare
years = ['2000', '2010', '2020']

# Set up the figure and axes
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

for ax, year in zip(axes, years):
    # Extract centrality values for the year
    vals1 = df_aligned[year]
    vals2 = mean_aligned[year]

    # Match company indices
    common_companies = df_aligned.index.intersection(mean_aligned.index)
    vals1 = vals1.loc[common_companies]
    vals2 = vals2.loc[common_companies]

    # Compute rankings (lower rank = higher centrality)
    ranks1 = vals1.rank(ascending=False)
    ranks2 = vals2.rank(ascending=False)

    # Q–Q scatter plot
    ax.scatter(ranks1, ranks2, s=60, alpha=0.7)
    ax.plot([ranks1.min(), ranks1.max()], [ranks1.min(), ranks1.max()], 'r--')
    ax.set_xlabel('Rank with multilayer contagion distance',fontsize=12)
    if ax == axes[0]:
        ax.set_ylabel('Ranking with aggregated contagion distance',fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('QQ_plot_rankings_distances.pdf',bbox_inches = 'tight')
plt.show()
