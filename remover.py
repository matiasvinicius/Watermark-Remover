import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
	img = cv.imread('NOME.jpg')
	roi = cv.selectROI(img)

	x_begin = int(roi[1])
	x_end = int(roi[1]+roi[3])
	y_begin = int(roi[0])
	y_end = int(roi[0]+roi[2])

	background = np.zeros((img.shape[0], img.shape[1]), dtype = 'uint8')
	rectangle = cv.rectangle(background.copy(), (y_begin, x_begin), (y_end, x_end), color = 255, thickness= -1)

	masked = cv.bitwise_or(img, img, mask = rectangle)
	masked = cv.resize(masked, None, fx=0.5, fy=0.5, interpolation= cv.INTER_AREA)
	masked = cv.resize(masked, (img.shape[1], img.shape[0]), interpolation= cv.INTER_AREA)
	masked = cv.blur(masked, (3, 3))
	masked = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)
	# masked = cv.Canny(masked, 127, 255)
	# masked = cv.adaptiveThreshold(masked, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
	
	ret, thresh = cv.threshold(masked, 127, 255, cv.THRESH_BINARY)
	contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
	masked = cv.drawContours(masked, contours, 0, 255, thickness=cv.FILLED)
	masked = cv.erode(masked, (9,9))
	masked = cv.dilate(masked, (9,9))

	dst = cv.inpaint(img, masked, 3, cv.INPAINT_NS)

	cv.imshow('Img', dst)
	cv.waitKey(0)
