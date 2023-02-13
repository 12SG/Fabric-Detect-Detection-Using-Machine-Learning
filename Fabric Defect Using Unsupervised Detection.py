#!/usr/bin/env python
# coding: utf-8

# ### Task No : 4 

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM


# In[2]:


# Load the fabric image
Image = cv2.imread("C:/Users/HP/Downloads/Data/Data/Task4/Fabric1.jpg")


# In[3]:


# Convert the image to grayscale
gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)


# In[4]:


# Segment the image using thresholding
thresh, seg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# In[5]:


# Calculate the feature vectors for each segmented region
feature_vectors = []
for i in np.unique(seg):
    if i == 0:
        continue
    region = np.zeros_like(gray)
    region[seg == i] = 255
    features = cv2.HuMoments(cv2.moments(region)).flatten()
    feature_vectors.append(features)


# In[6]:


# Train the one-class SVM model
model = OneClassSVM(kernel='rbf', nu=0.1)
model.fit(feature_vectors)


# In[7]:


# Predict the normal behavior of each segmented region
predictions = model.predict(feature_vectors)


# In[8]:


# Identify the potential fabric defects
defects = np.where(predictions == -1)


# In[9]:


defect_centers = []
for i in defects[0]:
    region = np.zeros_like(gray)
    region[seg == (i+1)] = 255
    M = cv2.moments(region)
    if M["m00"] == 0:
        continue
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    defect_centers.append((cX, cY))


# In[10]:


# Plot the original image with the defects highlighted
plt.imshow(Image)
for (x, y) in defect_centers:
    plt.scatter(x, y, color='red')
plt.show()


# In[ ]:




