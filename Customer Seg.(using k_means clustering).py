#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', '"rfm and cust_segm..ipynb"')


# In[2]:


rfm_df_new


# # in this , we will make clusters on RFM analysis data using K-means , unlike previous N.B in which we declared our own segments

# 

# In[3]:


# we have already checked the skewness and have also applied log transformation
# in the previous notebool


# In[4]:


rfm_df_new


# Now we create new df of required columns for KNN/
# 

# In[5]:


rfm_df_new1 = df[[ 'cust_id', 'Recency' , 'Frequency' , 'MonetaryValue' ,'RFM_Score'  ]]
rfm_df_new1


# In[6]:


grouped_by_cust_id = rfm_df_new1.groupby('cust_id')
mean_values_by_cust_id = grouped_by_cust_id.mean()

rfm_df_new1 = mean_values_by_cust_id


# In[7]:


rfm_df_new1


# In[ ]:





# before applying standard scaler , we will apply label encoding because the
# "cust_id" column is of string data type in df rfm_df_new

# In[8]:


le = LabelEncoder()


# In[9]:


#Get list of categorical variables
s = (rfm_df_new1.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)


# In[10]:


for i in object_cols:
    rfm_df_new1[i]=rfm_df_new1[[i]].apply(le.fit_transform)
print('all features are now numerical')    


# In[ ]:





# now we apply standard scaling

# In[11]:


scaler = StandardScaler()
scaler.fit(rfm_df_new1)
rfm_scaled = pd.DataFrame(scaler.transform(rfm_df_new1),columns=rfm_df_new1.columns)


# In[12]:


rfm_scaled


# # K-means Clustering

# In[13]:


wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i , random_state=0)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)


plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (i)')
plt.ylabel('WCSS')

plt.show()

    


# In[14]:


kmeans= KMeans(n_clusters=4, random_state=0)

y_predict = kmeans.fit_predict(rfm_scaled)

rfm_scaled['clusters'] = y_predict



# y_predict = kmeans.fit_predict(rfm_scaled): Fits the KMeans model to the dataset pca_ds and obtains the cluster labels for each data point. 
# 
# rfm_scaled['clusters'] = y_predict: Creates a new column named 'clusters' in the rfm_scaled DataFrame and assigns the cluster labels obtained from the KMeans model.
# 
# rfm_scaled['clusters'] = y_predict: If there's another DataFrame named rfm_scaled, it adds a new column named 'clusters' to it and assigns the same cluster labels obtained from the KMeans model
# 
# 
# After running this code, both rfm_scaled and df will have a new column named 'clusters' that contains the assigned cluster labels based on the KMeans clustering. Each row in these DataFrames is now associated with a cluster label indicating to which cluster the corresponding data point belongs.

# In[15]:


#Plot the clusters

cmap = matplotlib.cm.viridis



x = rfm_scaled['Recency']
y = rfm_scaled['Frequency']
z = rfm_scaled['MonetaryValue']

fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=rfm_scaled["clusters"], marker='o', cmap = cmap )
ax.set_title("The Plot Of The Clusters")
plt.show()


# plotting above graph using plotly.express

# In[16]:


import plotly.express as px

#m Assuming you have the following dataframes and variables

x = rfm_scaled['Recency']
y = rfm_scaled['Frequency']
z = rfm_scaled['MonetaryValue']

clusters = rfm_scaled['clusters']

fig = px.scatter_3d(rfm_scaled, x=x, y=y, z=z, color=clusters, size_max=10, opacity=0.7,
                     title="The Plot Of The Clusters")
fig.show()


# In[17]:


rfm_scaled


# In[18]:


rfm_scaled['RFM_Score'].max()


# In[19]:


rfm_scaled['RFM_Score'].min()


# we observe that max and min values are not scaled , hence first we will normalize
# the values so that it is easy to create segments

# In[20]:


rfm_scaled['RFM_Score'] = (rfm_scaled['RFM_Score'] - rfm_scaled['RFM_Score'].min()) / (rfm_scaled['RFM_Score'].max() - rfm_scaled['RFM_Score'].min())

# Subtract the minimum value from each element in the 'RFM_Score' column
# Divides the result from the previous step by the range ( max-min) of the 'RFM_Score' column.
# Finally, the normalized values are assigned back to the 'RFM_Score' 


# In[21]:


rfm_scaled['RFM_cust_seg'] = ''

rfm_scaled.loc[rfm_scaled['RFM_Score'] >= 0.75, 'RFM_cust_seg'] = 'Champions'
rfm_scaled.loc[(rfm_scaled['RFM_Score'] >= 0.5) & (rfm_scaled['RFM_Score'] < 0.75), 'RFM_cust_seg'] = 'Potential Loyalists'
rfm_scaled.loc[(rfm_scaled['RFM_Score'] >= 0.25) & (rfm_scaled['RFM_Score'] < 0.5), 'RFM_cust_seg'] = 'At Risk Customers'
rfm_scaled.loc[(rfm_scaled['RFM_Score'] >= 0) & (rfm_scaled['RFM_Score'] < 0.25), 'RFM_cust_seg'] = "Can't Lose"


# In[22]:


rfm_scaled['RFM_cust_seg'].value_counts()


# In[ ]:





# In[23]:


# Get the cluster labels for each data point
cluster_labels = kmeans.labels_

# Calculate the cluster size of each cluster
cluster_sizes = np.unique(cluster_labels, return_counts=True)[1]

# Print the cluster sizes
print(cluster_sizes)


# pie chart for cluster distribution

# In[24]:


import matplotlib.pyplot as plt

# Example cluster distribution data (replace with your actual data)
cluster_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
cluster_sizes = [245,219,266,63]  # Replace with the sizes of your clusters

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(cluster_sizes, labels=cluster_labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Add a title
plt.title('Cluster Distribution')

# Show the pie chart
plt.show()


# # Summarizing my findings

# In[25]:


def rfm_values(rfm_sclaed):
    df_new = rfm_scaled.groupby(['clusters']).agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': ['mean', 'count']
    }).round(0)
    
    return df_new

rfm_values(rfm_scaled)


# # Conclusion

# # 1. Cluster 0

# 
# Recency: The mean Recency value is 0.0, suggesting that customers in this cluster have made recent purchases.
# 
# Frequency: The mean Frequency value is 1.0, indicating a high frequency of purchases. Customers in this cluster are making purchases frequently.
# 
# MonetaryValue: The mean MonetaryValue is close to 0.0, suggesting moderate spending. Customers in this cluster are spending at a moderate level.
# 
# Count: This cluster has a count of 245, indicating a significant number of customers.

# # 2. Cluster 1

# 
# Recency: The mean Recency value is 1.0, indicating that customers in this cluster have not made recent purchases.
# 
# Frequency: The mean Frequency value is -1.0, suggesting low frequency or no recent purchases. Customers in this cluster are not making purchases frequently.
# 
# MonetaryValue: The mean MonetaryValue is close to 0.0, indicating average spending. Customers in this cluster are spending at an average level.
# 
# Count: This cluster has a count of 219, suggesting a notable number of customers.

# # 3. Cluster 2

# 
# Recency: The mean Recency value is close to 0.0, suggesting that customers in this cluster have made recent purchases.
#     
# Frequency: The mean Frequency value is -1.0, suggesting low frequency or no recent purchases. Customers in this cluster are not making purchases frequently.
#     
# MonetaryValue: The mean MonetaryValue is close to 0.0, suggesting moderate spending. Customers in this cluster are spending at a moderate level.
#     
# Count: This cluster has the highest count, with 266 customers, indicating it is the largest segment.

# # 4. Cluster 3

# 
# Recency: The mean Recency value is close to 0.0, suggesting recent activity.
#     
# Frequency: The mean Frequency value is 2.0, indicating a very high frequency of purchases. Customers in this cluster are making purchases very frequently.
# 
# MonetaryValue: The mean MonetaryValue is 2.0, suggesting high spending. Customers in this cluster are high spenders.
# 
# Count: This cluster has a count of 63, indicating the smallest group of customers.

# # in summary

# Cluster 0 represents customers who are recent and moderate spenders but make purchases frequently.
# 
# Cluster 1 represents customers who have not made recent purchases and have average spending.
# 
# Cluster 2 represents a large group of customers who are recent and moderate spenders but do not make purchases frequently.
# 
# Cluster 3 represents a small group of high-frequency, high-spending customers with recent activity.
# 

# In[ ]:




