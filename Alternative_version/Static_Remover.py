import cv2 as cv
import numpy as np

'''
References used:
- Processing: https://www.youtube.com/watch?v=oXlwWbU8l2o&t=9326s
- ROI: https://learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
'''

class Remover:
	def __init__(self, path):
		self.img = cv.imread(path)
	
	# Use ROI from OpenCV for the user to selected an area in the image and return its coordinates
	def coords_roi(self, img):
		roi = cv.selectROI(img)
		x_begin = int(roi[1])
		x_end = int(roi[1]+roi[3])
		y_begin = int(roi[0])
		y_end = int(roi[0]+roi[2])
		return [x_begin, x_end], [y_begin, y_end]
	
	'''
    Core method, it orchestrates all of the image processing and handles the interactive UI
    '''
	def remove_watermark(self):
		x_roi, y_roi = self.coords_roi(self.img)
		x_img, y_img = self.img.shape[:2] #image width and height

		#Create a black background and white rectangle (with ROI coordinates) to act as a mask for the image
		black_background = np.zeros((x_img, y_img), dtype = 'uint8')
		white_rectangle = cv.rectangle(
			black_background.copy(), 
			(y_roi[0], x_roi[0]), 
			(y_roi[1], x_roi[1]), 
			color = 255, 
			thickness= -1)

		#Captures the rectangle that has the ROI coordinates
		mask = cv.bitwise_or(self.img, self.img, mask = white_rectangle)

		#Convert mask to grayscale and apply adaptive threshold
		mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
		mask = cv.adaptiveThreshold(mask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

		#Adaptive Threshold will paint white not only the outlines but also the edge of the ROI rectangle. 
		#To avoid this we create a smaller rectangle to use as a mask in an AND, removing this edge.
		crop_border = cv.rectangle(
			black_background.copy(), 
			(y_roi[0]+1, x_roi[0]+1), 
			(y_roi[1]+1, x_roi[1]+1), 
			color = 255, 
			thickness= -1)
		mask = cv.bitwise_and(mask, mask, mask = crop_border)

		#Take what is actually contoured on the mask (not isolated paintings) and paint it white. The rest goes black
		ret, thresh = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
		contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
		mask = cv.drawContours(mask, contours, -1, 255, thickness=2)

		#Erosion, dilation and an average blur on the edges to reduce the perception of different colors)
		mask = cv.erode(mask, (9,9))
		mask = cv.dilate(mask, (9,9))
		mask = cv.blur(mask, (7,7))

		#Paint contours by FMM method modified by TELEA
		final_image = cv.inpaint(self.img, mask, 3, cv.INPAINT_TELEA)

		cv.imshow('Image', final_image)
		cv.waitKey(0)