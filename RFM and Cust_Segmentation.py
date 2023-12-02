#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st



# In[2]:


get_ipython().system('pip install xplotter --upgrade')


# In[3]:


# pip install -U yellowbrick


# In[4]:


# In[5]:



# # Data Loading
# 

# In[6]:


#Importing the Libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)


from warnings import filterwarnings
filterwarnings('ignore')

# Visualization Libraries
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go




# Xplotter
from xplotter.insights import *
from xplotter.formatter import format_spines


# In[ ]:





# In[7]:


df = pd.read_csv("F:\\Downloads\\archive(3)\\Sample - Superstore.csv")
print("Number of datapoints:", len(df))
df.head()


# # Data Cleaning and Analysis

# In[8]:


#Information about features 
df.info()


# In[9]:


#checking for negative values
numeric = df.select_dtypes(include=['int' , 'float'])
neg_value = numeric[numeric<0].sum()
neg_value


# In[10]:


#check for duplicate rows
df.duplicated()


# In[11]:


#checking for  null values
df.isnull().sum()


# we observe that the above dataset has no null values
# 

# Now we will be exploring the unique values in the categorical features to get a clear idea of the data.

# In[12]:


print("Total categories in column Country")
df.Country.value_counts()


# In[13]:


print("Total categories in column Category")
df.Category.value_counts()


# In[14]:


print("total categories in column Product Name")

df["Product Name"].unique() #gives all unique elements



# In[15]:


df.Quantity.value_counts()


# we observe that min whole quantity ordered was 3 , and max was 12

# In[16]:


df.Sales.describe()


# # Exploratory Data Analysis

# Since our Order Date column is in normal date form, to perform a deep dive analysis on customer's time ordering behaviour, it's advised to break them into small pieces such as year, month, and even the day of week it belongs to for easier data insights to get.

# In[17]:


# Changing the data type for date columns
timestamp_cols = ['Order Date']
for col in timestamp_cols:
    df[col] = pd.to_datetime(df[col])


# In[18]:


#extracting year

df['order_year'] = df['Order Date'].dt.year
df.sample(2)


# In[19]:


df.rename(columns={'Customer ID':'cust_id', 'Order ID':'order_id'}, inplace=True)


# In[20]:


#extracting month
df['order_month'] = df['Order Date'].dt.month_name()
df.sample(2)


# In[21]:


# extract date of week
df['date_dow'] = df['Order Date'].dt.day_name()
df.sample(1)


# In[22]:


df.head()


# # Calculating RFM values

# In[23]:


print(df['Order Date'].min())


# In[24]:


print(df['Order Date'].max())


# In[25]:


curnt_date = datetime.now().date()
curnt_date


# In[ ]:





# Calculate Recency

# In[26]:


df['Recency'] = (curnt_date - df['Order Date'].dt.date).dt.days

# It gives us the number of days since the customer’s last purchase, 
# representing their recency value.


# Calculate Frequency

# In[27]:


frequency_data = df.groupby('cust_id')['order_id'].count().reset_index()


# we calculated the frequency for each customer. We grouped the data by ‘CustomerID’ and counted the number of unique ‘OrderID’ values to determine the number of purchases made by each customer. It gives us the frequency value, representing the total number of purchases made by each customer.

# In[28]:


frequency_data.rename(columns={'order_id' : 'Frequency'},inplace=True)
#Renames the order_id column to Frequency.


# In[29]:


df = df.merge(frequency_data , on='cust_id' , how = 'left')

#how specifies the type of join


#  Merges the df and frequency_data on the cust_id column 
#  using a left join. 
# This means that all rows from the df DataFrame will be included in the merged 
# DataFrame, even if there are no corresponding rows in the frequency_data DataFrame 

# In[ ]:





# Calculate Monetary Value

# In[30]:


monetary_data = df.groupby('cust_id')['Sales'].sum().reset_index()


# we calculated the monetary value for each customer. We grouped the data by ‘CustomerID’ and summed the ‘Sales’ values to calculate the total amount spent by each customer. It gives us the monetary value, representing the total monetary contribution of each customer.

# In[31]:


monetary_data.rename(columns={'Sales' : 'MonetaryValue'} , inplace=True)
#renaming sales column 


# In[32]:


df = df.merge(monetary_data, on='cust_id' , how='left')
#merge monetary_data col to cust_id column using left join


# In[33]:


df.head()


# 

# # Calculating RFM scores

# In[34]:


# Define scoring criteria for each RFM value

recency_scores = [5, 4, 3, 2, 1]  # Higher score for lower recency (more recent)
frequency_scores = [1, 2, 3, 4, 5]  # Higher score for higher frequency
monetary_scores = [1, 2, 3, 4, 5]  # Higher score for higher monetary value


# In[35]:


#Calculate RFM scores
   
df['RecencyScore'] = pd.cut(df['Recency'], bins=5, labels=recency_scores)
df['FrequencyScore'] = pd.cut(df['Frequency'], bins=5, labels=frequency_scores)
df['MonetaryScore'] = pd.cut(df['MonetaryValue'], bins=5, labels=monetary_scores)


# we used the pd.cut() function to divide recency, frequency, and monetary values into bins. We define 5 bins for each value and assign the corresponding scores to each bin.
# 
# 
# The pd.cut() function takes three arguments:
# 
# The column to be discretized (in this case, df['Recency']).
# The number of bins to use (in this case, 5).
# The labels to assign to each bin (in this case, the recency_scores list).
# 
# pd.cut() fn returns series of categorical values

# In[36]:


# Convert RFM scores to numeric type

df['RecencyScore'] = df['RecencyScore'].astype(int)
df['FrequencyScore'] = df['FrequencyScore'].astype(int)
df['MonetaryScore'] = df['MonetaryScore'].astype(int)


# In[ ]:





# # RFM value segmentation

# In[37]:


# Calculate RFM score by combining the individual scores
df['RFM_Score'] = df['RecencyScore'] + df['FrequencyScore'] + df['MonetaryScore']


# In[38]:


# Create RFM segments based on the RFM score

segment_labels = ['Low-Value', 'Mid-Value', 'High-Value']
df['Value Segment'] = pd.qcut(df['RFM_Score'], q=3, labels=segment_labels)


# The pd.qcut() function takes three arguments:
# 
# The column to be discretized (in this case, df['RFM_Score']).
# 
# The number of quantiles to use (in this case, 3).
# 
# The labels to assign to each quantile (in this case, the segment_labels list).

# In[39]:


#FINAL RESULT

df.head()


# In[ ]:





# In[ ]:





# In[40]:


sns.distplot(df['Recency'])


# In[41]:


sns.distplot(df['Frequency'])


# In[42]:


sns.distplot(df['MonetaryValue'])


# we observe that our plots are not even close to normal distribution , so we 
# will use transformation techniques

# # applying Log Transformation

# In[43]:


from sklearn.preprocessing import  PowerTransformer
from sklearn.preprocessing import FunctionTransformer


# In[44]:


pt = FunctionTransformer(func=np.log1p)


# In[45]:


X = df['Recency']
Y = df['Frequency']
Z = df['MonetaryValue']


# In[46]:


rec_pt = pt.fit_transform(X)
freq_pt = pt.fit_transform(Y)
mont_pt = pt.fit_transform(Z)


# In[47]:


sns.distplot(rec_pt)


# In[48]:


sns.distplot(freq_pt)


# In[49]:


sns.distplot(mont_pt)


# In[ ]:





# In[50]:


rfm_df_new = df[[ 'cust_id', 'Recency' , 'Frequency' , 'MonetaryValue' , 'RecencyScore' , 'MonetaryScore' , 'RFM_Score' , 'Value Segment' ]]
rfm_df_new


# # Now, we will create a dataset of some selective columns for our further analysis

# In[51]:


rfm_df_new = df[[ 'cust_id', 'Recency' , 'Frequency' , 'MonetaryValue' , 'RecencyScore' , 'MonetaryScore' , 'RFM_Score' , 'Value Segment' ]]
rfm_df_new


# # Plotting RFM segement distribution

# In[52]:


import plotly.express as px

segment_counts = rfm_df_new['Value Segment'].value_counts()
segment_counts.columns = ['Value Segment','Count']

pastel_colors = px.colors.qualitative.Pastel

#create bar chart
fig_segment_dist = px.bar(segment_counts ,y ='Value Segment', 
                         color='Value Segment',
                         color_discrete_sequence = pastel_colors,
                         title = 'RFM Value Segment Distribution')

#update the layout
fig_segment_dist.update_layout(xaxis_title='RFM Value Segment',
                              yaxis_title = 'Count',
                              showlegend = False)

#show the figure
fig_segment_dist.show()


# # RFM Customer Segments

# Now let’s create and analyze RFM Customer Segments that are broader classifications based on the RFM scores. These segments, such as “Champions”, “Potential Loyalists”, and “Can’t Lose” provide a more strategic perspective on customer behaviour and characteristics in terms of recency, frequency, and monetary aspects

# In[53]:


rfm_df_new['RFM_Score'].max()
rfm_df_new['RFM_Score'].min()


# # (We Create 5 Customer Segments)

# In[54]:


# New column for RFM customer Segments
rfm_df_new['RFM_cust_seg'] = '' 

# Assign RFM segments based on RFM score
rfm_df_new.loc[rfm_df_new['RFM_Score'] >= 12, 'RFM_cust_seg'] = 'Champions'
rfm_df_new.loc[(rfm_df_new['RFM_Score'] >= 10) & (rfm_df_new['RFM_Score'] < 12), 'RFM_cust_seg'] = 'Potential Loyalists'
rfm_df_new.loc[(rfm_df_new['RFM_Score'] >= 8) & (rfm_df_new['RFM_Score'] < 10), 'RFM_cust_seg'] = 'At Risk Customers'
rfm_df_new.loc[(rfm_df_new['RFM_Score'] >= 4) & (rfm_df_new['RFM_Score'] < 8), 'RFM_cust_seg'] = "Can't Lose"
rfm_df_new.loc[(rfm_df_new['RFM_Score'] >= 3) & (rfm_df_new['RFM_Score'] < 4), 'RFM_cust_seg'] = "Lost"


# In[55]:


rfm_df_new['RFM_cust_seg'].value_counts()


# rfm_df_new.loc[...]: This is using the loc accessor to select rows in the DataFrame that meet the specified conditions.
# 
# '(rfm_df_new['RFM_Score'] >= 6) & (rfm_df_new['RFM_Score'] < 9): This is a logical AND operation combining the two conditions, meaning it selects rows where the 'RFM_Score' is both greater than or equal to 6 and less than 9.
# 
# 'RFM_cust_seg': This is the column that will be updated based on the conditions.

# In[56]:


#print updated data with RFM segments
print(rfm_df_new[['cust_id','RFM_cust_seg']])


# # Plotting the above segments

# In[57]:


segment_product_counts = rfm_df_new.groupby(['Value Segment' , 'RFM_cust_seg']).size().reset_index(name='Count')

segment_product_counts = segment_product_counts.sort_values('Count' ,ascending=False)

fig_treemap_segment_product = px.treemap(segment_product_counts,
                                        path=['Value Segment' , 'RFM_cust_seg'],
                                        values='Count' ,
                                        color='Value Segment' , color_discrete_sequence=px.colors.qualitative.Pastel,
                                        title='RFM Customer Segments by Value')
fig_treemap_segment_product.show()


# Treemaps are a visualization technique that displays hierarchical data as nested rectangles. Each branch of the hierarchy is given a colored rectangle, and smaller rectangles represent sub-branches. The size of each rectangle corresponds to a certain attribute, such as the number of items, sales, or any other quantitative measure.
# 
# px.treemap() is a function from the Plotly Express library, used to create treemaps.
# 
# The segment_product_counts DataFrame is provided as the data source.
# 
# path specifies the hierarchical structure of the treemap, indicating that it should be organized by 'Value Segment' and 'RFM_cust_seg'.
# 
# values specifies the numerical values associated with each segment, which, in this case, is the 'Count' column.
# 
# color is set to 'Value Segment', and color_discrete_sequence defines the color palette for different 'Value Segment' values.
# 
# title sets the title of the treemap.

# # Now let’s analyze the correlation of the RFM scores within the champions segment:

# In[58]:


import plotly.graph_objects as go

champions_segment = rfm_df_new[rfm_df_new['RFM_cust_seg'] == 'Champions']

correlation_matrix = champions_segment[['RecencyScore','Frequency' , 'MonetaryScore']].corr()

#visualize correlation using heatmap
fig_heatmap = go.Figure(data=go.Heatmap(
z=correlation_matrix.values,
x=correlation_matrix.columns,
y=correlation_matrix.columns,
colorscale = 'RdBu',
colorbar=dict(title='Correlation')))

fig_heatmap.update_layout(title='correlation matrix of RFM values with Champions Segment')

fig_heatmap.show()


# 'champions_segment = rfm_df_new[rfm_df_new['RFM_cust_seg'] == 'Champions']':
# The code then filters a DataFrame (rfm_df_new) to create a new DataFrame (champions_segment). It selects only those rows where the value in the 'RFM_cust_seg' column is 'Champions'.
# 
# 
# 'correlation_matrix = champions_segment[['RecencyScore','Frequency' , 'MonetaryScore']].corr()' :
# The next step involves calculating the correlation matrix for a subset of columns in the champions_segment DataFrame. The selected columns are 'RecencyScore', 'Frequency', and 'MonetaryScore'
# 
# 
# 'fig_heatmap = go.Figure(data=go.Heatmap' :
#  This line creates a new Plotly figure and assigns it to the variable fig_heatmap.
# The data parameter of the go.Figure constructor is used to specify the trace that will be added to the figure. In this case, a Heatmap trace is added.
# 
# z: This parameter specifies the data values for the heatmap. In this case, it's set to the values of the correlation matrix obtained from correlation_matrix.values.
# 
# x: This parameter sets the labels for the x-axis of the heatmap. It takes the column names from the correlation matrix (correlation_matrix.columns).
# 
# y: Similarly, this parameter sets the labels for the y-axis of the heatmap, using the column names.
# 
# The colorbar parameter is a dictionary that configures the colorbar associated with the heatmap.
# title: This sets the title of the colorbar to 'Correlation'.
# 

# # Now let’s have a look at the number of customers in all the segments:

# In[59]:


import plotly.express as px

# Get the segment counts
segment_counts = rfm_df_new['RFM_cust_seg'].value_counts()

# Create a bar chart with different colors for each segment
fig = px.bar(
    x=segment_counts.index,
    y=segment_counts.values,
    color=segment_counts.index,
    color_discrete_sequence=px.colors.qualitative.Pastel,
)

# Set the color of the 'Champions' segment to be different
champions_color = 'rgb(158,202,225)'
fig.update_traces(marker_color=[champions_color if segment == 'Champions'
                                else px.colors.qualitative.Pastel[i] for i, segment in enumerate(segment_counts.index)])

# Update the layout
fig.update_layout(
    title='Comparison of RFM Segments',
    xaxis_title='RFM Segments',
    yaxis_title='Number of Customers',
    showlegend=False,
)

# Show the chart
fig.show()


# In[60]:


import plotly.colors

pastel_colors = plotly.colors.qualitative.Pastel

#getting unique values
segment_counts = rfm_df_new['RFM_cust_seg'].value_counts()


# Create a bar chart to compare segment counts
fig = go.Figure(data=[go.Bar(x=segment_counts.index, y=segment_counts.values,
                            marker=dict(color=pastel_colors))])

# Set the color of the Champions segment as a different color
champions_color = 'rgb(158, 202, 225)'
fig.update_traces(marker_color=[champions_color if segment == 'Champions' else pastel_colors[i]
                                for i, segment in enumerate(segment_counts.index)],
                  marker_line_color='rgb(8, 48, 107)',
                  marker_line_width=1.5, opacity=0.6)

# Update the layout
fig.update_layout(title='Comparison of RFM Segments',
                  xaxis_title='RFM Segments',
                  yaxis_title='Number of Customers',
                  showlegend=False)

fig.show()


# "import plotly.colors"
# This line imports the plotly.colors module, which is part of the Plotly library and provides access to various color-related functions and color scales
# 
# "pastel_colors = plotly.colors.qualitative.Pastel"
# This line creates a variable pastel_colors and assigns it the Pastel color scale from the plotly.colors.qualitative module. The Pastel color scale is a set of soft, muted colors.
# 
# 
# "fig = go.Figure(data=[go.Bar(x=segment_counts.index, y=segment_counts.values, marker=dict(color=pastel_colors))])"
# These lines create a Plotly figure (fig) with a bar chart. The x-axis represents the unique segment labels, and the y-axis represents the corresponding count of customers in each segment. The marker argument is used to set the color of the bars to the Pastel color scale.
# 
# "Update Trace Properties in the Bar Chart: "
# 
# marker_color:
# It sets the color of each bar in the chart. The list comprehension is used to determine the color for each segment. If the segment is 'Champions', it uses the specific champions_color; otherwise, it uses colors from the pastel_colors sequence.
# 
# marker_line_color:
# It sets the color of the outline (border) of the bars. In this case, it's set to a dark blue color.
# 
# marker_line_width:
# It sets the width of the outline of the bars. The value is set to 1.5, indicating a relatively thicker outline.
# 
# opacity:
# It sets the transparency of the bars. The value is set to 0.6, making the bars slightly transparent.

# # Now let’s have a look at the recency, frequency, and monetary scores of all the segments:

# In[61]:


import plotly.graph_objects as go

segment_scores =rfm_df_new.groupby('RFM_cust_seg')['RecencyScore' , 'Frequency' , 'MonetaryScore'].mean().reset_index()

#create a group bar chart to compare segment scores
fig = go.Figure()

#add bars to recency score
fig.add_trace(go.Bar(
x=segment_scores['RFM_cust_seg'],
y=segment_scores['RecencyScore'],
name='Recency Score',
marker_color = 'rgb(158,202,225)'
))

#add bars to Frequency Score
fig.add_trace(go.Bar(
x=segment_scores['RFM_cust_seg'],
y=segment_scores['Frequency'],
name = 'Frequency Score',
marker_color = 'rgb(94,158,217)'
             ))

# Add bars for Monetary score
fig.add_trace(go.Bar(
x=segment_scores['RFM_cust_seg'],
y=segment_scores['MonetaryScore'],
name = 'Monetary Score',
marker_color = 'rgb(32,102,148)'
))

#update the layout
fig.update_layout(
title='comparison of RFM segments bases on R,F,M score',
xaxis_title = 'RFM Segments',
yaxis_title = 'Score',
barmode='group',
showlegend=True)

fig.show()


# # pie chart for cluster distribution

# In[62]:


import matplotlib.pyplot as plt

# Example cluster distribution data (replace with your actual data)
cluster_labels = ['Cant Lose', 'At Risk Customers', 'Potential Loyalists', 'Lost' , 'Champions']
cluster_sizes = [5399,3424,876,175,120]  # Replace with the sizes of your clusters

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(cluster_sizes, labels=cluster_labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Add a title
plt.title('Cluster Distribution')

# Show the pie chart
plt.show()


# # Summarizing my findings

# In[63]:


def rfm_values(rfm_sclaed):
    df_new = rfm_df_new.groupby(['RFM_cust_seg']).agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': ['mean', 'count']
    }).round(0)
    
    return df_new

rfm_values(rfm_df_new)


# # Conclusion

# # 1. 'At Risk Customers'

# Recency: The mean Recency value is 2446.0, indicating that these customers have not made purchases recently.
#     
# Frequency: The mean Frequency value is 17.0, suggesting a moderate level of purchasing frequency.
#     
# MonetaryValue: The mean MonetaryValue is 3824.0, indicating a moderate level of spending.
#     
# Count: There are 3424 customers in this segment.

# # 2. 'Cant Lose'
# 

# Recency: The mean Recency value is 2945.0, suggesting that these customers have not made purchases recently.
# 
# Frequency: The mean Frequency value is 13.0, indicating a lower level of purchasing frequency compared to other segments.
# 
# MonetaryValue: The mean MonetaryValue is 2762.0, indicating a lower level of spending.
# 
# Count: This segment is the largest with 5399 customers.

# # 3. 'Champions'

#  Recency: The mean Recency value is 2945.0, suggesting that these customers have not made purchases recently.
#     
# Frequency: The mean Frequency value is 13.0, indicating a lower level of purchasing frequency compared to other segments.
#     
# MonetaryValue: The mean MonetaryValue is 2762.0, indicating a lower level of spending.
#     
# Count: This segment is the largest with 5399 customers.

# # 4. 'Lost'

# Recency: The mean Recency value is 3409.0, indicating that these customers have not made purchases recently.
# 
# Frequency: The mean Frequency value is 7.0, indicating a low level of purchasing frequency.
# 
# MonetaryValue: The mean MonetaryValue is 1108.0, indicating a low level of spending.
# 
# Count: This segment has 175 customers.

# # 5. ' Potential Loyalists'

# Recency: The mean Recency value is 2405.0, indicating that these customers have made purchases more recently than some segments but less recently than others.
# 
# Frequency: The mean Frequency value is 25.0, indicating a high level of purchasing frequency.
# 
# MonetaryValue: The mean MonetaryValue is 7405.0, indicating a moderate level of spending.
# 
# Count: This segment has 876 customers.

# # In Summary

# At Risk Customers:- Customers in this segment have not made recent purchases but have a moderate level of frequency and spending.
# 
# Can't Lose:- This is the largest segment, with customers who have not made recent purchases and have lower frequency and spending.
# 
# Champions:- A smaller but high-value segment with recent and frequent purchases, representing valuable customers.
# 
# Lost:- Customers in this segment have not made recent purchases and have low frequency and spending.
# 
# Potential Loyalists:- Customers in this segment have made relatively recent and frequent purchases with moderate spending.
# 

# In[ ]:




