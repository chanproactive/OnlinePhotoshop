from os import dup2
import streamlit as st
import numpy as np
from PIL import Image
import altair as alt
import pandas as pd
from scipy import signal
from scipy import ndimage
from astropy.convolution import RickerWavelet2DKernel
import weightedstats as ws
import cv2

def uploadImage(lable):
    #Upload file to app
    file = st.file_uploader(lable,type=['jpg','jpeg','png','DCM'])
    if file is None:
        st.stop()
    img = Image.open(file)
    #Convert PIL object to np array
    img = np.array(img, dtype=np.uint16)
    return img

# convert image to gray scale function
def rgb2gray(img):
    return np.around(np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])).astype(int)
# Inverse image function
def inverseImg(img):
    inv_img = 255-img
    return inv_img

#Create histogram data flame
def createHist(img,channels):  
    hist_df = pd.DataFrame(range(bins_size),columns=['intensity'])
    cum_hist_df = pd.DataFrame(range(bins_size),columns=['intensity'])
    if channels==1:
        hist,bin_edges = np.histogram(img,bins = bins_size,range = (0,255))
        hist_df[str(channels)] = hist
        cum_hist_df[str(channels)] = np.cumsum(hist)
        return hist_df,cum_hist_df
    else:
        for channel in range(channels):
            hist,bin_edges = np.histogram(img[:][:][channel],bins = bins_size,range = (0,255))
            hist_df[str(channel)] = hist
            cum_hist_df[str(channel)] = np.cumsum(hist)
        return hist_df,cum_hist_df         

#plot histogram
def plothist(hist_df,cum_hist_df,color_set):
    #Convert Wide-form Data frame to Long-form which require for altair library
    histlong = hist_df.melt('intensity', var_name='color', value_name='count')
    #show histogram using altair
    rgb = alt.Chart(histlong).mark_area(opacity=0.8,interpolate='step').encode(
    x='intensity',
    y='count:Q',
    color = alt.Color('color:N', scale=alt.Scale( range=color_set))).interactive()
    st.text('histogram')
    st.altair_chart(rgb, use_container_width=True)
    #Convert Wide-form Data frame to Long-form which require for altair library
    cum_histlong = cum_hist_df.melt('intensity', var_name='color', value_name='count')
    #show histogram using altair
    rgb = alt.Chart(cum_histlong).mark_area(opacity=0.8,interpolate='step').encode(
    x='intensity',
    y='count:Q',
    color = alt.Color('color:N', scale=alt.Scale( range=color_set))).interactive()
    st.text('cumulative histogram')
    st.altair_chart(rgb, use_container_width=True)

# Modify intensity fucntion
def modIntensity(img,c,b):
    img = img*c+b
    img = np.where(img<255,img,255)
    img = np.where(img>0,img,0)
    return img

# threshold function
def threshold(img,th,a0,a1):
    img = np.where(img<th,a0,img)
    img = np.where(img>=th,a1,img)
    return img

def autoContrast(img,amin =0,amax=255):
    #Create gray scale image of it is not so
    if len(img.shape)==3:
        imgGray = rgb2gray(img)
    else:
        imgGray = img
    #Set defult alow and ahigh
    alow = np.min(imgGray)
    ahigh = np.max(imgGray)
    #Create array of possible intensity value
    i = np.arange(256)
    #Select image range base on q value
    if mod_auto_contrast:
        #Create cumulative histrogram
        cum_hist = np.cumsum(np.histogram(imgGray,bins = bins_size,range = (0,255))[0])
        #Get low and high pixel value form quantiles
        pix_low = img.shape[0]*img.shape[1]*q
        pix_high = img.shape[0]*img.shape[1]*(1-q)
        #Filter out out bound intensity values 
        i = i[(cum_hist>=pix_low) & (cum_hist<=pix_high)]
        #Set new alow and ahigh to be edge of the intensity range
        alow = np.min(i)
        ahigh = np.max(i)
    #Linearly transfrom image 
    img =amin+(img-alow)*((amax-amin)/(ahigh-alow))
    # map out bound to be amin nad a max
    img = np.where(img<=alow,amin,img)
    img = np.where(img>=ahigh,amax,img)
    return img

# histogram Equalization function
def histEqualization(img):
    #check whether the image is in gray scale if not convert it
    if len(img.shape)==3:
        imgGray = rgb2gray(img)
    else:
        imgGray = img
    #Create cumulative histrogram
    cum_hist = np.cumsum(np.histogram(imgGray,bins = bins_size,range = (0,255))[0])
    #Calcalate new intrnsity value array
    newIntensity = np.around((cum_hist*255/cum_hist[-1]))
    #map image to new intrnsity value
    return np.interp(img,np.arange(256),newIntensity)

# image specification function
def histSpec(img):
    #Create array of possible intensity value
    i = np.arange(256)
    #Let user upload the second image
    img2 = uploadImage('Upload your second image here')
    #check whether the image is in gray scale if not convert it first
    if len(img.shape)==3:
        imgGray = rgb2gray(img)
    else:
        imgGray = img
    if len(img2.shape)==3:
        imgGray2 = rgb2gray(img2)
    else:
        imgGray2 = img2
    #plot info about image 2
    st.image(imgGray2)
    hist ,cum_hist = createHist(imgGray2,1)
    plothist(hist,cum_hist,['white'])

    #calculate normalized cdf of both image 
    cdf = np.cumsum(np.histogram(imgGray,bins = bins_size,range = (0,255))[0])
    cdf = cdf/cdf[-1]
    cdf2 = np.cumsum(np.histogram(imgGray2,bins = bins_size,range = (0,255))[0])
    cdf2 = cdf2/cdf2[-1]

    #Find new intensity value by compair 2 cdf
    newIntensity = np.interp(cdf,cdf2,i)
    #map image to new intrnsity value
    return np.interp(img,i,newIntensity)

def gaussianKern(kernlen=5, std=1):
    ax = np.linspace(-(kernlen - 1) / 2., (kernlen - 1) / 2., kernlen)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(std))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)
    
def laplaceKern(kernlen=5,a =2):
    laplace1D = signal.ricker(kernlen, a)
    laplace2D = np.outer(laplace1D, laplace1D)
    return  laplace2D

def laplaceKern2(width):
    return RickerWavelet2DKernel(width)._array
    
    
def convolve(img,kernel):
    #calculate kernel L,K kernel width parameter
    L = ((kernel.shape[0]-1)//2)
    K = ((kernel.shape[1]-1)//2)

    #create copy of image with padding
    img_copy = np.pad(img,((L,L),(K,K)),mode='reflect')

    # iterate to each pixel
    for v in range(L,img_copy.shape[0]-L):
        for u in range(K,img_copy.shape[1]-K):
            #Get intensity val of interested zone 
            Hot_Zone = img_copy[v-L:v+L+1,u-K:u+K+1]
            #Compute new intensity value and replace the old one
            img[v-L,u-K] = np.around(np.sum(Hot_Zone*kernel))
    return img

def convolve2(image, kernel):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(image.shape)
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
    return output

def convolve3(img, kernel):
    if len(img.shape)==3:
        kernel = kernel[:, :, np.newaxis]
    return signal.fftconvolve(img, kernel, mode='same')

def non_linear(img,size,mode):
    #calculate kernel L,K kernel width parameter
    L = ((size.shape[0]-1)//2)
    K = ((size.shape[1]-1)//2)

    #create copy of image with padding
    img_copy = np.pad(img,((L,L),(K,K)),mode='reflect')

    if mode == 'max':
    # iterate to each pixel
        for v in range(L,img_copy.shape[0]-L):
            for u in range(K,img_copy.shape[1]-K):
                #Get intensity val of interested zone 
                Hot_Zone = img_copy[v-L:v+L+1,u-K:u+K+1]
                #Find max value in hot_zone and replace old value
                img[v-L,u-K] = np.around(np.max(Hot_Zone))
    elif mode == 'min':
        for v in range(L,img_copy.shape[0]-L):
            for u in range(K,img_copy.shape[1]-K):              
                Hot_Zone = img_copy[v-L:v+L+1,u-K:u+K+1]
                #Find min value in hot_zone and replace old value
                img[v-L,u-K] = np.around(np.min(Hot_Zone))
    elif mode == 'median':
        for v in range(L,img_copy.shape[0]-L):
            for u in range(K,img_copy.shape[1]-K):  
                Hot_Zone = img_copy[v-L:v+L+1,u-K:u+K+1]
                #Find median value in hot_zone and replace old value
                img[v-L,u-K] = np.around(np.median(Hot_Zone))
    elif mode == 'weighted_median':
        for v in range(L,img_copy.shape[0]-L):
            for u in range(K,img_copy.shape[1]-K):           
                Hot_Zone = img_copy[v-L:v+L+1,u-K:u+K+1]
                #Compute weighted median value in hot_zone and replace old value
                img[v-L,u-K] = np.around(ws.numpy_weighted_median(Hot_Zone, weights = size))
    return img

def non_max_suppression(img, gradient_direction):
    angles = np.rad2deg(gradient_direction)
    angles[angles < 0] += 180
    image_row, image_col = img.shape
    output = np.zeros(img.shape)
    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            angle = angles[row, col]
            #angle 0
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                before_pixel = img[row, col - 1]
                after_pixel = img[row, col + 1]
            #angle 45
            elif (22.5 <= angle < 67.5):
                before_pixel = img[row + 1, col - 1]
                after_pixel = img[row - 1, col + 1]
            #angle 90
            elif (67.5 <= angle < 112.5):
                before_pixel = img[row - 1, col]
                after_pixel = img[row + 1, col]
            #angle 135
            elif (112.5 <= angle < 157.5):
                before_pixel = img[row - 1, col - 1]
                after_pixel = img[row + 1, col + 1]

            if img[row, col] >= before_pixel and img[row, col] >= after_pixel:
                output[row, col] = img[row, col]
            else:
                output[row, col] = 0
    return output

def threshold(image, low, high, weak):
 
    output = np.zeros(image.shape)
    strong = 255
    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))
 
    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak
    return output

def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img
#set view/ init valable
bins_size = 256
# st.set_page_config(layout="centered")
imgGray_cb= st.sidebar.checkbox('Gray scale image')
inv_img_cb = st.sidebar.checkbox('Invert color') 
show_RGB_hist = st.sidebar.checkbox('Show color histogram',True) 
contrast = st.sidebar.slider('contrast', 0.0, 2.0,1.0)
brightness = st.sidebar.slider('brightness', -100,100,0 )
two_tone_cb = st.sidebar.checkbox('convert to binary image') 
if  two_tone_cb:
    th = st.sidebar.slider('threshold', 0,255,150 )
    a0 = st.sidebar.slider('a0', 0,255,0 )
    a1 = st.sidebar.slider('a1', 0,255,255 )
auto_contrast_cb = st.sidebar.checkbox('Auto contrast',False)
if auto_contrast_cb:
    amin = st.sidebar.slider('amin', 0, 255,0)
    amax = st.sidebar.slider('amax', 0,255,255)
    mod_auto_contrast = st.sidebar.checkbox('mod auto contrast') 
    if  mod_auto_contrast:
        q = st.sidebar.slider('percent quantiles', 0.0,5.0,0.0,0.01)/100
hist_eq = st.sidebar.checkbox('Histogram Equalization',False)
hist_spec = st.sidebar.checkbox('Histogram Specification',False)
filter = st.sidebar.checkbox('Filter',False)
if filter:
    filter_type = st.sidebar.radio(
     "Filter type",
     ('Box filter', 'Gauss filter', 'Laplace filter','Custom'))
nonlinear_filter = st.sidebar.checkbox('Nonliner Filter',False)
if nonlinear_filter:
    nonlinear_filter_type = st.sidebar.radio(
        "nonlinear filter type",
        ('Max filter', 'Min filter', 'Median filter','Weighted Median filter'))
edge_detection = st.sidebar.checkbox('edge detection',False)
if edge_detection:
    edge_detection_type = st.sidebar.radio(
        "Edge detection type",
        ('Prewitt Operator', 'Sobel Operator', 'Robert Operator','Compass filter','Canny Operator','Laplace sharpening','Unsharp mask sharpening'))
    if edge_detection_type == "Laplace sharpening":
        sharpening_strength = st.sidebar.slider('Sharpening strength', 1,20 ,0)
    if edge_detection_type == "Unsharp mask sharpening":
        sharpening_strength = st.sidebar.slider('Sharpening strength', 1,50 ,0)
#Call uploadImage function 
img = uploadImage('Upload your image here')
# img = np.random.random_integers(0, 255, (500, 500))
#Convert image to gray scale
if imgGray_cb:
    img = rgb2gray(img)
#Call inverseImg function
if inv_img_cb:
     img = inverseImg(img)

#Call modIntensity function
img= modIntensity(img,contrast,brightness)
#Call autoContrast function
if auto_contrast_cb:
    img = autoContrast(img,amin,amax)
#Call threshold function
if two_tone_cb:  
    img = threshold(img,th,a0,a1)
if hist_eq:
    img = histEqualization(img)
    # print(img)
if hist_spec:
    img = histSpec(img)

if filter:
    if filter_type == 'Box filter':
        size = st.sidebar.slider('kernel size', 3,100,step = 2)
        kernel = np.ones((size,size))
        kernel = kernel/np.sum(kernel)
    if filter_type == 'Laplace filter':
        width = st.sidebar.slider('width', 0.1,2.0,0.1 )
        kernel = laplaceKern2(width)
    if filter_type == 'Gauss filter':
        size = st.sidebar.slider('kernel size', 3,100,3 )
        kernel = gaussianKern(size,size/3)
    if filter_type == 'Custom':
        pre_kernal = []
        kernel = np.empty((3,2))
        text = st.sidebar.text_area('Insert filter here')
        if text == '':
            st.stop()
        else:
            text = text.split()
            for t in text:
                pre_kernal.append( [int(x) for x in t.split(',')])
        kernel = np.array(pre_kernal)
        kernel = kernel/np.sum(kernel)
    img = convolve3(img,kernel)
    img = (255*(img - np.min(img))/np.ptp(img))       

if nonlinear_filter:
    if nonlinear_filter_type == 'Max filter':
        size = st.sidebar.slider('kernel size', 3,100,step = 2)
        kernel = np.ones((size,size))
        mode = 'max'
        img = ndimage.maximum_filter(img,(size,size))
        # img = non_linear(img,kernel,mode)
    if nonlinear_filter_type == 'Min filter':
        size = st.sidebar.slider('kernel size', 3,100,step = 2)
        kernel = np.ones((size,size))
        mode = 'min'
        img = ndimage.minimum_filter(img,(size,size))
    if nonlinear_filter_type == 'Median filter':
        size = st.sidebar.slider('kernel size', 3,100,step = 2)
        kernel = np.ones((size,size))
        mode = 'median'
        img = ndimage.median_filter(img,(size,size))
    if nonlinear_filter_type == 'Weighted Median filter':
        pre_kernal = []
        kernel = np.empty((3,2))
        text = st.sidebar.text_area('Insert filter here')
        if text == '':
            st.stop()
        else:
            text = text.split()
            for t in text:
                pre_kernal.append( [int(x) for x in t.split(',')])
        kernel = np.array(pre_kernal)
        mode = 'weighted_median'
        img = non_linear(img,kernel,mode)
    # img = non_linear(img,kernel,mode)
if edge_detection:
    if edge_detection_type == 'Prewitt Operator':
        # Create Prewitt Operator for x axis
        kernelx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        # Apply the convolution using the horizontal mask
        imgx = convolve3(img,kernelx)
        # Apply the convolution using the horizontal mask transpose
        imgy = convolve3(img,kernelx.T)
        # Create gradient vector
        img  = np.sqrt(np.square(imgx)+np.square(imgy))
        # Normallize the image to be in between 0-255
        img = img*255.0/img.max()
    
    elif edge_detection_type == 'Sobel Operator':
        kernelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        imgx = convolve3(img,kernelx)
        imgy = convolve3(img,kernelx.T)
        img  = np.sqrt(np.square(imgx)+np.square(imgy))
        img = img*255.0/img.max()

    elif edge_detection_type == 'Robert Operator':
        kernelx = np.array([[0,1],[-1,0]])
        imgx = convolve3(img,kernelx)
        imgy = convolve3(img,kernelx.T)
        img  = np.sqrt(np.square(imgx)+np.square(imgy))
        img = img*255.0/img.max()

    elif edge_detection_type == 'Compass filter':
        #Create kernel
        kernal_H0 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        kernal_H1 = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
        kernal_H2 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        kernal_H3 = np.array([[0,-1,-2],[1,0,-1],[2,1,0]])
        #Apply the convolution to the image
        D0 = convolve3(img,kernal_H0)
        D1 = convolve3(img,kernal_H1)
        D2 = convolve3(img,kernal_H2)
        D3 = convolve3(img,kernal_H3)
        #Find maximum among the 4 convoluted image and store in img
        img = np.maximum.reduce([np.abs(D0),np.abs(D1),np.abs(D2),np.abs(D3)])
        #Normalizaion
        img = img*255.0/img.max()
        #img = (255*(img - np.min(img))/np.ptp(img))        
    elif edge_detection_type == 'Canny Operator':
        img = convolve3(img,gaussianKern(9))
        kernelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        imgx = convolve3(img,kernelx)
        imgy = convolve3(img,kernelx.T)
        img = np.hypot(imgy, imgx)     
        theta = -np.arctan2(imgy, imgx)
        img = non_max_suppression(img,theta)
        img = img*255.0/img.max() 
        img = threshold(img,5,20,100)
        img = hysteresis(img,100)

    elif edge_detection_type == 'Laplace sharpening':
        print("hello")
        kernel = np.array([[0,0,-1,-1,-1,0,0],
                            [0,-1,-3,-3,-3,-1,0],
                            [-1,-3,0,7,0,-3,-1],
                            [-1,-3,7,24,7,-3,-1],
                            [-1,-3,0,7,0,-3,-1],
                            [0,-1,-3,-3,-3,-1,0],
                            [0,0,-1,-1,-1,0,0]])
        img = img-sharpening_strength*convolve3(img,kernel/np.sum(np.abs(kernel)))
        img = (255*(img - np.min(img))/np.ptp(img))
        print(img.min(),img.max())
    
    elif edge_detection_type == 'Unsharp mask sharpening':
        img = img+sharpening_strength*(img-convolve3(img,gaussianKern(20))) 
        img = (255*(img - np.min(img))/np.ptp(img))
        print(img.min(),img.max())
#show image
img = img.astype(int)
st.image(img,clamp=False)
# fig, ax = plt.subplots()
# ax.imshow(img, cmap='gray')
# st.pyplot(fig)

#crete color set for histogram
color_set = ['red', 'green', 'blue']

#Check number of channel if the image  
# if only 1 channel the set histogram to white 
if len(img.shape)==2:
    channels = 1
    color_set = ['white']

else: 
    channels = img.shape[2]
# if showRGB is not checked, show the same as 1 channel histrogram 
    if not show_RGB_hist:
        channels = 1
        img = rgb2gray(img)
        color_set = ['white']
col1,col2 = st.columns(2)
hist_df,cum_hist_df= createHist(img,channels)

plothist(hist_df,cum_hist_df,color_set)


print("....................")

