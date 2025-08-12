# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:47:03 2025

@author: mbozhidarova
"""

"""
Created on Thu Apr 24 11:54:18 2025

@author: mbozhidarova
"""
import pandas as pd
# Trade data downloaded from https://wits.worldbank.org/gvc/gvc-indicators-metadata.html
df = pd.read_csv('gvc_trade_WITS_2023.csv')

df = df[df['t']>=2000] #Choose only after year 2000

df = df[df['exp']!=df['imp']] #choose only trade between countries

# Choose only columns that we would need and disregard others:
needed_columns = ['exp', 'imp', 'sect', 't', 'source', 'gtrade', 'gvc', 'gvcbp', 'gvcfp', 'gvcmix']

df = df[needed_columns]

#Combine all sectors into a single one:
df = df[['exp', 'imp', 'sect', 't', 'source','gtrade']]
#len(set(df['exp']))
#Out[35]: 190 #so we have 190 countries to work with

df_sectors = pd.read_csv(path+'\\gvc-sectors.csv')
df_sectors = df_sectors[['sect', 'source', 'sect_name']]

#Combine sectors manually:
sector_names = df_sectors[df_sectors['source']=='eora']['sect_name']
sector_names_combined = ['1-Agriculture and Fishing','2-Mining and Quarrying',
                         '3-Food and Beverages','4-Textiles and Wearing Apparel',
                         '5-Wood and Paper','6-Petroleum, Chemical and Non-Metallic Minerals',
                         '7-Metal Products','8-Electrical and Machinery',
                         '9-Transport Equipment','10-Electricity, Gas and Water',
                         '11-Construction','12-Transport','13-Post and Telecommunications',
                         '14-Finacial Intermediation and Business Activity','15-Public Administration']

sectors=dict()
sectors['1-Agriculture and Fishing'] = [('eora',1),('eora',2),('wiodo',1),('wiodn',1),
                                        ('wiodn',2),('wiodn',3),('wiodlr',1),('adb',1),
                                        ('tiva',1),('tiva',2)]
sectors['2-Mining and Quarrying'] = [('eora',3),('wiodo',2),('wiodn',4),('wiodlr',2),
                                     ('adb',2),('tiva',3),('tiva',4),('tiva',5)]
sectors['3-Food and Beverages'] = [('eora',4),('wiodo',3),('wiodn',5),('wiodlr',3),
                                     ('adb',3),('tiva',6)]
sectors['4-Textiles and Wearing Apparel'] = [('eora',5),('wiodo',4),('wiodo',5),('wiodn',6),('wiodlr',4),
                                     ('adb',4),('adb',5),('tiva',7)]
sectors['5-Wood and Paper'] = [('eora',6),('wiodo',6),('wiodo',7),('wiodn',7),('wiodn',8),('wiodlr',5),
                                     ('adb',6),('adb',7),('tiva',8),('tiva',9)]
sectors['6-Petroleum, Chemical and Non-Metallic Minerals'] = [('eora',7),('wiodo',8),('wiodo',9),
                                                              ('wiodo',10),('wiodo',11),('wiodn',10),
                                                              ('wiodn',11),('wiodn',12),('wiodn',13),
                                                              ('wiodn',14),('wiodlr',6),('wiodlr',7),('wiodlr',8),
                                     ('wiodlr',9),('adb',8),('adb',9),('adb',10),('adb',11),('tiva',10),('tiva',11),
                                     ('tiva',12),('tiva',13),('tiva',14)]
sectors['7-Metal Products'] = [('eora',8),('wiodo',12),('wiodn',15),('wiodn',16),('wiodlr',10),
                                     ('adb',12),('tiva',15),('tiva',16)]
sectors['8-Electrical and Machinery'] = [('eora',9),('wiodo',13),('wiodo',14),('wiodn',17),('wiodn',18),('wiodn',19),
                                         ('wiodlr',11),('wiodlr',12),('adb',13),('adb',14),('tiva',17),('tiva',18),('tiva',19)]
sectors['9-Transport Equipment'] = [('eora',10),('wiodo',15),('wiodn',20),('wiodn',21),
                                         ('wiodlr',13),('adb',15),('tiva',20),('tiva',21)]
sectors['10-Electricity, Gas and Water'] = [('eora',13),('wiodo',17),('wiodn',24),('wiodn',25),
                                            ('wiodn',26),('wiodlr',15),
                                     ('adb',17),('tiva',23),('tiva',24)]
sectors['11-Construction'] = [('eora',14),('wiodo',18),('wiodn',27),('wiodlr',16),
                                     ('adb',18),('tiva',25)]
sectors['12-Transport'] = [('eora',19),('wiodo',23),('wiodo',24),('wiodo',25),('wiodo',26),
                           ('wiodn',31),('wiodn',32),('wiodn',33),('wiodn',34),
                           ('wiodlr',19),('adb',23),('adb',24),('adb',25),('adb',26),
                           ('tiva',27),('tiva',28),('tiva',29),('tiva',30)]
sectors['13-Post and Telecommunications'] = [('eora',20),('wiodo',27),('wiodn',35),('wiodn',39),('wiodlr',20),
                                     ('adb',27),('tiva',31),('tiva',34),('tiva',35)]
sectors['14-Finacial Intermediation and Business Activity'] = [('eora',21),('wiodo',28),('wiodn',40),('wiodn',41),
                                                               ('wiodn',42),('wiodn',45),('wiodlr',21),('adb',28),('tiva',36)]
sectors['15-Public Administration'] = [('eora',22),('wiodo',31),('wiodn',50),('wiodn',51),('wiodlr',23),
                                     ('adb',31),('tiva',39),('tiva',40)]


# Create a reverse mapping for quick lookup
lookup = {
    (source, sect): key
    for key, pairs in sectors.items()
    for source, sect in pairs
}

# Assuming df is your DataFrame with 'source' and 'sect' columns
df['unified_sector'] = df.apply(lambda row: lookup.get((row['source'], row['sect'])), axis=1)

df = df[['exp', 'imp', 't', 'gtrade', 'unified_sector']]

#because after 2016 we have only 63 companies reporting we will do it like this, so we have data for all those 63 companies:
df = df[df['exp'].isin(set(df[df['t']==2022]['exp']))]
df = df[df['imp'].isin(set(df[df['t']==2022]['imp']))]


df.to_csv(path+'\\Trade_data_full.csv',index=False) #save the trade data for future use
