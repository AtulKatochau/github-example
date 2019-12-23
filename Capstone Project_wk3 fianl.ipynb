#!/usr/bin/env python
# coding: utf-8

# CAPSTONE PROJECT - CLUSTERING TORONTO NEIGHBOURHOOD DATA

# In[3]:


import requests # library to handle requests
import pandas as pd # library for data analsysis
import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation

get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values

# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 
    
# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize

get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium # plotting library

print('Folium installed')
print('Libraries imported.')


# In[38]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library

print('Libraries imported.')


# QUESTION 1
# 
# SCRAPPING WEB SITES FOR TORONTO DATA
# 

# In[4]:


url = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'


# In[63]:


df = pd.read_html(url)
len(df)
toronto_neigh = df[0]
toronto_neigh.head()


# In[64]:


toronto_neigh.describe()


# In[65]:


toronto_neigh.shape


# In[66]:


toronto_neigh.replace(np.nan,"Not assigned", inplace= True)
toronto_neigh.columns


# In[67]:


toronto_neigh.shape


# In[68]:


toronto_neigh.head()


# DATA WRANGLING PHASE

# In[69]:


# Remove the 'Not assigned' rows as they are useless
neigh = toronto_neigh
toronto_df = neigh[ (neigh['Borough'] != 'Not assigned')]
toronto_df.head()


# In[70]:


# group the Neighbourhood by Postcode
df_gp_neigh = toronto_df.groupby('Postcode').Neighbourhood.agg( [('Neighbourhood', ','.join) ])
df_gp_neigh.reset_index(inplace=True)
df_gp = pd.merge(df_gp_neigh, toronto_df[['Postcode','Borough']], on='Postcode')
df_gp.drop_duplicates(inplace=True)
df_gp.head()


# In[71]:


# update "Not assigned" Neighbourhood with the Borough
df_gp['Neighbourhood'].replace('Not assigned', df_gp['Borough'],inplace=True)
# verify that there is no more 'Not assigned' borough
df_gp[ (df_gp['Neighbourhood'] == 'Not assigned') | (df_gp['Borough'] == 'Not assigned')] 


# In[72]:


df_gp.shape


# In[73]:


df_gp.head()


# ###QUESTION - 2 #####

# In[74]:


# get the geospatial data
url_geo = 'http://cocl.us/Geospatial_data'
df_geo = pd.read_csv(url_geo)
df_geo.rename({'Postal Code':'Postcode'}, axis='columns', inplace=True)
df_geo.head()


# MERGING GEOSPATIAL DATA TO THE DATAFRAME

# In[105]:


df_toronto_geo = pd.merge (df_gp, df_geo, on ='Postcode')
df_toronto_geo.head()


# In[76]:


df_toronto_geo.dropna().shape


# In[77]:


df_toronto_geo['Borough'].value_counts()


# #####QUESTION 3 ######

# In[78]:


toronto_data = df_toronto_geo [ df_toronto_geo['Borough'].str.contains('Downtown Toronto')]
toronto_data.head()


# In[79]:


CLIENT_ID = 'ECGBI3GEB1KD3LFY53E05UD2O5WZ42WPIAYLN4KZJFFL00CB' # your Foursquare ID
CLIENT_SECRET = 'FWCXBMA1ZFWKAJIPHHZOOY3QIPL3HK1Z5YKWUOCJMCEPIHPW' # your Foursquare Secret
VERSION = '20180604'
LIMIT = 100
radius = 400
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[80]:


from bs4 import BeautifulSoup


# In[81]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[82]:


toronto_venues = getNearbyVenues(names=toronto_data['Neighbourhood'],
                                   latitudes=toronto_data['Latitude'],
                                   longitudes=toronto_data['Longitude']
                                  )


# In[83]:


toronto_venues.head()


# In[84]:


# get the number of venues in Toronto
print(toronto_venues.shape)


# In[85]:


# how many venues per neighboorhood do we have ?
toronto_venues.groupby('Neighborhood')['Venue Category'].count().head()


# In[86]:


print('There are {} uniques venue categories.'.format(len(toronto_venues['Venue Category'].unique())))


# # Get line for each venue alongwith its category as dummy columns

# In[87]:


# one hot encoding for each category
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['YNeighborhood'] = toronto_venues['Neighborhood'] 

toronto_onehot.columns[-1]

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]
toronto_onehot.rename(columns={'YNeighborhood': 'TheNeighborhood'}, inplace=True)

toronto_onehot.head()


# In[88]:


toronto_onehot.shape


# In[89]:


toronto_grouped = toronto_onehot.groupby('TheNeighborhood').mean().reset_index()
toronto_grouped.head()


# In[90]:


toronto_grouped.shape


# In[91]:


num_top_venues = 5

for hood in toronto_grouped['TheNeighborhood'].head():
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['TheNeighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[92]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['TheNeighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['TheNeighborhood'] = toronto_grouped['TheNeighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[93]:


# set number of clusters
kclusters = 5
toronto_grouped_clustering = toronto_grouped.drop('TheNeighborhood', 1)

#inertias = np.zeros((kclusters-1))
#for n in range(1,kclusters):
    
# run k-means clustering
kmeans = KMeans (n_clusters=kclusters, random_state=0, n_init=12).fit(toronto_grouped_clustering)
#   inertias[n-1] = kmeans.inertia_
kmeans.labels_[0:10]
# display the inertia    
#inertias


# In[94]:


# Display the information on the centers
print("number of clusters:", len(kmeans.cluster_centers_))
#print("cluster center:" , kmeans.cluster_centers_)
print("number of iterations:", kmeans.n_iter_)
print("inertia (Sum of squared distances of samples to cluster center): ", kmeans.inertia_)


# In[95]:


neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
neighborhoods_venues_sorted.astype({'Cluster Labels':'int32'},copy=False)

toronto_merged = toronto_data

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.merge(neighborhoods_venues_sorted.set_index('TheNeighborhood'), how='left', right_on='TheNeighborhood', left_on='Neighbourhood')


# In[96]:


toronto_merged = toronto_merged[ toronto_merged['Neighbourhood'] != "Queen\'s Park"]
toronto_merged.head()


# In[97]:


# display the number of borough per cluster
toronto_merged['Cluster Labels'].value_counts()


# In[98]:


address = 'Toronto'

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of toronto are {}, {}.'.format(latitude, longitude))


# In[99]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)

# set color scheme for the clusters
x = np.arange(len(toronto_merged['Cluster Labels'].value_counts()))
ys = [i + x + (i*x)**2 for i in range(len(toronto_merged['Cluster Labels'].value_counts()))]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]
count=0

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + '/ Cluster is ' + str(cluster), parse_html=True)  
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster)],
        fill=True,
        fill_color=rainbow[int(cluster)],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[100]:


####Cluster 1 - Coffee Shops and Restaurants ###
toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[101]:


###Cluster 2 - Parks and Playground ###
toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[102]:


####Cluster 3 - Airport and Airport Lounge ###

toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[103]:


### Cluster 4 - cafe ###

toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[104]:


####Cluster 5 - Grocery Store ###

toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




