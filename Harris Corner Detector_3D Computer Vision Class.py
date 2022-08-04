#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


''' Load image using OpenCV '''
# Upload the LMS image to Google Drive and point to its location.
# Note OpenCV reads image as BGR.
#img_bgr = cv2.imread("checkerboard.png")
img_bgr = cv2.imread("maskimage.png")
# Normalize image to between 0 and 1.
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float) / 255.0

# Show output
plt.figure(figsize=(10, 10))
plt.imshow(img, cmap='gray')
plt.show()


# In[3]:


# Perform Sobel filtering along the x-axis.
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Ix = cv2.filter2D(img, -1, sobel_x)
# Perform Sobel filtering along the y-axis.
''' DO IT YOURSELF for sobel_y and Iy '''
sobel_y = sobel_x.T
ly = cv2.filter2D(img, -1, sobel_y)

plt.subplot(1,2,1)
plt.imshow(Ix)
plt.subplot(1,2,2)
plt.imshow(ly)


# In[4]:


''' Approximate local error surface '''
window_size = 3
offset = int(np.floor(window_size/2))

det = np.zeros(img.shape)
trace = np.zeros(img.shape) 
matrix_R = np.zeros(img.shape)

# For each pixel in image
for y in range(offset, img.shape[0]-offset):
  for x in range(offset, img.shape[1]-offset):
    
    # Build ROI window around the current pixel
    # Note numpy uses height-by-width convention (row x column)
    window_x = Ix[y-offset:y+offset+1, x-offset:x+offset+1]
    ''' DO IT YOURSELF for window_y '''
    window_y = ly[y-offset:y+offset+1, x-offset:x+offset+1]
    
    # Estimate elements of matrix M.
    Sxx = np.sum(window_x * window_x)
    #print(Sxx)
    ''' DO IT YOURSELF for Syy and Sxy '''
    Syy = np.sum(window_y * window_y)
    #print(Syy)
    Sxy = np.sum(window_y * window_x)
    #print(Sxy)
    # Compute determinant of M and trace of M.
    # Note numpy uses height-by-width convention (row x column)
    trace[y,x] = Sxx + Syy
    ''' DO IT YOURSELF for det[y,x] '''
    det[y,x] = (Sxx * Syy) - (Sxy**2)   


# In[5]:


# Set hyperparameters
alpha = 0.05
beta = 0.1

# Compute response map
''' DO IT YOURSELF!: R = det(M) - alpha * trace(M)^2 '''
R = det - alpha * (trace**2)

# Use thresholding to discard responses with low amplitude
''' DO IT YOURSELF! R is discarded if R < beta * max(R) '''
maxR = np.max(R)

for i in range(0, R.shape[0]):
    for j in range(0, R.shape[1]):
        if R[i,j] < beta * maxR:
            R[i,j] = 0
            
''' DO IT YOURSELF! Define gaussian_2d kernel as a numpy array. '''
gaussian_1d = cv2.getGaussianKernel(5,3)
gaussian_2d = np.outer(gaussian_1d, gaussian_1d.transpose())

R = cv2.filter2D(R, -1, gaussian_2d)
# Show the response map
plt.figure(figsize=(10, 10))
plt.imshow(R, cmap='gray')
plt.show()


# In[6]:


# Set NMS window size
window_size = 3
offset = int(np.floor(window_size/2))
output_img = np.zeros(img.shape) 

# For each pixel, perform non-maximal suppressi
for y in range(offset, img.shape[0]-offset):
  for x in range(offset, img.shape[1]-offset):
    
    if R[y,x] == 0.0:
      # If the response map value is 0, then we can skip
      continue

    center_value = R[y,x]
    ''' DO IT YOURSELF! get max_value of the 3x3 block ''' 
    block_3x3 = R[y:y+window_size,x:x+window_size]
    max_value = np.amax(block_3x3)
    
    # If the center value is not the same as the maximum value of the 3x3 block, 
    # then it's not maximum, so suppress.
    # Otherwise, let the pixel survive.
    if center_value != max_value:
        ''' DO IT YOURSELF for output_img[y,x] '''
        # suppress
        output_img[y,x] = 0
    
    else:
        ''' DO IT YOURSELF for output_img[y,x] '''
        output_img[y,x] = 1.0


# In[7]:


''' Extract feature points and draw on the image '''
y, x = np.where(output_img==1.0)

for i in range(0, len(x)):
    output_vis = cv2.circle(img_bgr, (x[i], y[i]), 3, (0, 0, 255))
    print('x:%d,y:%d' % (x[i],y[i]))

plt.imshow(output_vis, cmap='gray')
plt.show()

#print('rgb',output_vis)
#cv2.imshow('test',output_vis)


# In[ ]:




