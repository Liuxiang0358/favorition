import cv2
import numpy as np

f = 0.5

# Read images : src image will be cloned into dst
im = cv2.imread("001.jpg")
obj= cv2.imread("1.jpg")
#obj= cv2.imread("160.jpg")
#obj = Contrast_and_Brightness(1.5,1,obj)
obj = cv2.resize(obj,dsize=None,fx=f,fy=f,interpolation=cv2.INTER_LINEAR)
# Create an all white mask
gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
#gray_mask = gray[gray != 0]
#mask = 255 * np.ones(obj.shape, obj.dtype)

#obj = 255 * np.ones(obj.shape, obj.dtype)


mask = np.zeros(obj.shape, obj.dtype)
a = np.zeros(obj.shape[:-1], obj.dtype)
a[gray != 0] = 255
mask[:,:,0]= a
mask[:,:,1]= a
mask[:,:,2]= a
kernel = np.ones((2,2), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)
#a = np.zeros(obj.shape[:-1], obj.dtype)
#mean = np.mean(im).astype(np.uint8)
#mean = 0
#a[gray == 0] = mean
#obj[:,:,0]= a
#obj[:,:,1]= a
#obj[:,:,2]= a
# The location of the center of the src in the dst
width, height, channels = im.shape
center = (int(height/2), int(width/2))

# Seamlessly clone src into dst and put the results in output
normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)

# Write results
cv2.imwrite("opencv-normal-clone-example.jpg", normal_clone)
cv2.imwrite("opencv-mixed-clone-example.jpg", mixed_clone)