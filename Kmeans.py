{\rtf1\ansi\ansicpg1252\cocoartf2708
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Code for running the K-Means algorithm\
!pip install sklearn\
from sklearn.cluster import KMeans\
import pandas as pd\
\
#from sklearn.preprocessing import MinMaxScaler\
from matplotlib import pyplot as plt\
import numpy as np\
from numpy import arange\
from sklearn.decomposition import PCA\
import plotly.graph_objs as go\
from plotly.offline import init_notebook_mode, iplot\
import plotly.graph_objects as go\
#init_notebook_mode()\
def configure_plotly_browser_state():\
import IPython\
display(IPython.core.display.HTML('''\
<script src="/static/components/requirejs/require.js"></script>\
<script>\
requirejs.config(\{\
paths: \{\
base: '/static/base',\
plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',\
\},\
\});\
</script>\
'''))\
from google.colab import files\
uploaded = files.upload()\
#df= pd.read_csv("Data.csv")\
df= pd.read_csv("KMeans2.csv")\
#df = df.loc[:, ~df.columns.str.contains('^Unnamed')]\
df.head()\
#df=df.drop(['Difficulty Level','Avg Time'], axis=1)\
#df=df.drop(['Difficulty Level 3','Time*Failure Rate','Failure Rate','Avg Time in Seconds','Total Attributes','Sample Number'], axis=1)\
#df=df.drop(['Difficulty Level 3','Time*Failure Rate','Total Attributes','Sample Number'], axis=1)\
df=df.drop(['Normalized Time','Sample Number'], axis=1)\
df.head()\
df.info()\
cols = df.columns[0:]\
cols\
#Elbow Method\
distorsions = []\
for k in range(1, 10):\
kmeans = KMeans(n_clusters=k)\
kmeans.fit(df[df.columns[2:]])\
distorsions.append(kmeans.inertia_)\
\
fig = plt.figure(figsize=(15, 5))\
plt.plot(range(1, 10), distorsions)\
plt.grid(True)\
plt.title('Elbow curve')\
from sklearn.metrics import silhouette_score\
#Silhouette scores\
range_n_clusters = [2, 3, 4, 5, 6]\
for n_clusters in range_n_clusters:\
clusterer = KMeans(n_clusters=n_clusters)\
preds = clusterer.fit_predict(df[df.columns[2:]])\
centers = clusterer.cluster_centers_\
\
score = silhouette_score(df[df.columns[2:]], preds)\
print("For n_clusters = \{\}, silhouette score is \{\})".format(n_clusters, score))\
km=KMeans(n_clusters=3)\
#y=km.fit_predict(df)\
df["cluster"] = km.fit_predict(df[df.columns[2:]])\
df.tail()\
# Principal component separation to create a 2-dimensional picture\
pca = PCA(n_components = 2)\
df['x'] = pca.fit_transform(df[cols])[:,0]\
df['y'] = pca.fit_transform(df[cols])[:,1]\
df = df.reset_index()\
df.tail()\
trace0 = go.Scatter(x = df[df.cluster == 0]["x"],\
y = df[df.cluster == 0]["y"],\
name = "Cluster 1",\
mode = "markers",\
marker = dict(size = 10,\
color = "rgba(15, 152, 152, 0.5)",\
line = dict(width = 1, color = "rgb(0,0,0)")))\
trace1 = go.Scatter(x = df[df.cluster == 1]["x"],\
y = df[df.cluster == 1]["y"],\
name = "Cluster 2",\
mode = "markers",\
marker = dict(size = 10,\
color = "rgba(180, 18, 180, 0.5)",\
line = dict(width = 1, color = "rgb(0,0,0)")))\
trace2 = go.Scatter(x = df[df.cluster == 2]["x"],\
y = df[df.cluster == 2]["y"],\
name = "Cluster 3",\
mode = "markers",\
marker = dict(size = 10,\
color = "rgba(132, 132, 132, 0.8)",\
line = dict(width = 1, color = "rgb(0,0,0)")))\
# trace3 = go.Scatter(x = df[df.cluster == 3]["x"],\
# y = df[df.cluster == 3]["y"],\
# name = "Cluster 4",\
# mode = "markers",\
# marker = dict(size = 10,\
# color = "rgba(122, 122, 12, 0.8)",\
# line = dict(width = 1, color = "rgb(0,0,0)")))\
#Required Code to run plotly in a cell\
configure_plotly_browser_state()\
init_notebook_mode(connected=False)\
data = [trace0, trace1, trace2]\
\
iplot (data)\
from google.colab import drive\
drive.mount('drive')\
df.to_csv('drive/My Drive/KMeansResultWith3Clusters.csv')\
}