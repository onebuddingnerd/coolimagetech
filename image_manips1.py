import cv2
import numpy as np
from matplotlib import pyplot as plt

# create a cv2 object of an image,
# and display it in a window that waits
# for the 'X' press 
def demo1():
	# read image
	i1 = cv2.imread("./messi1.png")
	
	# display image
	cv2.imshow("i1",i1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# creates cv2 object of image
# (0 second param. for grayscale);
# waits for esc key (27) to exit
# 's' key press to save and exit
def demo2():
	i2 = cv2.imread('messi1.png',0)
	cv2.imshow('i2',i2)
	k = cv2.waitKey(0)
	if k == 27:         # wait for ESC key to exit
	    cv2.destroyAllWindows()
	elif k == ord('s'): # wait for 's' key to save and exit
	    cv2.imwrite('messigray.png',i2)
	    cv2.destroyAllWindows()


# show in matplotlib UI (grey messi)
def demo3():
	i3 = cv2.imread('messi1.png',0)
	plt.imshow(i3, cmap = 'gray', interpolation = 'bicubic')
	plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	plt.show()


# uncomment!
#demo3() 
#demo2()
#demo1()

