#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint


# In[2]:
df = pd.read_csv('data/AIS_2021_01_01.csv')


# In[3]:
df.size


# In[4]:
df.head()


# In[5]:
#drop nulls and duplicates
df_clean = df.dropna(axis=0, how='any', inplace=False)
df_clean.drop_duplicates(keep='first', inplace=True)
df = pd.DataFrame()


# In[6]:
df_clean.head()


# In[7]:
#where are these vessels located?
df_clean.plot.scatter(x='LON', y='LAT', c = 'purple')


# In[8]:
# Creating dataframe just for very slow/stopped vessels

#very slow is < 0.5 knots
df_slow = df_clean[df_clean.SOG < 0.5]


# In[9]:
df_slow.head()


# In[10]:
df_slow["LAT"].max(), df_slow["LAT"].min()


# In[11]:
df_slow["LON"].max(), df_slow["LON"].min()


# In[12]:
df_slow.plot.scatter(x= 'LON', y="LAT", c = 'purple')


# In[13]:
# check size to see how feasible to plot
df_slow.shape


# In[14]:
df_slow_reduced = df_slow[(df_slow["LAT"] < 30) & (df_slow["LAT"] > 20) & (df_slow["LON"] > -90) & (df_slow["LON"] < -70)]


# In[15]:
df_slow_reduced.shape


# In[16]:
df_slow_reduced.head()


# In[17]:
# plot df_slow_reduced
df_slow_reduced.plot.scatter(x= "LON", y="LAT", c = 'purple')


# In[18]:
latitude = df_slow["LAT"].mean()
longitude = df_slow["LON"].mean()

DFW_Map = folium.Map(location=[latitude, longitude], zoom_start=10)
DFW_Map


# In[25]:
# Pick location for folium map
latitude = 26
Longitude = -80


# In[26]:
# map vessels with folium--- use VesselName as label
slow_vessels = folium.map.FeatureGroup()
latitudes = list(df_slow_reduced.LAT)
longitudes = list(df_slow_reduced.LON)
labels = list(df_slow_reduced.VesselName)
for Lat, Lon, labels in zip(latitudes, longitudes, labels):
    folium.Marker(
        location=[Lat, Lon],
        popup= labels,
        icon= folium.Icon(color='red', icon='info-sign')
    ).add_to(DFW_Map)


# In[27]:
DFW_Map


# In[28]:
# CLUSTERING - DBSCAN
df_slow_reduced.head()


# In[29]:
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
coords = df_slow_reduced[['LAT', 'LON']].to_numpy()


# In[30]:
# first DBSCAN with epsilon = 1.5km
kms_per_radian = 6371.0088
epsilon = 1.5 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
print('Number of clusters: {}'.format(num_clusters))


# In[32]:
df_slow_reduced["label"] = cluster_labels


# In[33]:
# plot label #1
fg = sns.FacetGrid(data=df_slow_reduced, hue="label", height=6, aspect=1.61)
fg.map(plt.scatter, 'LAT', 'LON').add_legend()


# In[34]:
kms_per_radian = 6371.0088
epsilon = 20 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
print('Number of clusters: {}'.format(num_clusters))


# In[37]:
# plot label 2
df_slow_reduced["label2"] = cluster_labels

fg = sns.FacetGrid(data=df_slow_reduced, hue='label2', aspect=1.61)
fg.map(plt.scatter, 'LAT', 'LON').add_legend()


# In[41]:
kms_per_radian = 6371.0088
epsilon = 100 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
print('Number of clusters: {}'.format(num_clusters))


# In[42]:
df_slow_reduced["label3"] = cluster_labels


# In[43]:
# plot label 3
fg = sns.FacetGrid(data=df_slow_reduced, hue='label3', aspect=1.61)
fg.map(plt.scatter, 'LAT', 'LON').add_legend()


# In[44]:
df_sample = df.head(5000)
df_sample.to_csv("data/sample.csv", index=False)

