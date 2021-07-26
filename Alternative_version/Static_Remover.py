import cv2 as cv
import numpy as np

class Remover:
	def __init__(self, path):
		self.img = cv.imread(path)
	
	def coords_roi(self, img):
		roi = cv.selectROI(img)
		x_begin = int(roi[1])
		x_end = int(roi[1]+roi[3])
		y_begin = int(roi[0])
		y_end = int(roi[0]+roi[2])
		return [x_begin, x_end], [y_begin, y_end]
	
	def remove_watermark(self):
		x_roi, y_roi = self.coords_roi(self.img)
		x_img, y_img = self.img.shape[:2]

		black_background = np.zeros((x_img, y_img), dtype = 'uint8')
		white_rectangle = cv.rectangle(
			black_background.copy(), 
			(y_roi[0], x_roi[0]), 
			(y_roi[1], x_roi[1]), 
			color = 255, 
			thickness= -1)

		mask = cv.bitwise_or(self.img, self.img, mask = white_rectangle)
		mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
		mask = cv.adaptiveThreshold(mask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
		crop_border = cv.rectangle(
			black_background.copy(), 
			(y_roi[0]+1, x_roi[0]+1), 
			(y_roi[1]+1, x_roi[1]+1), 
			color = 255, 
			thickness= -1)
		mask = cv.bitwise_and(mask, mask, mask = crop_border)

		ret, thresh = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
		contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
		mask = cv.drawContours(mask, contours, -1, 255, thickness=2)
		mask = cv.erode(mask, (9,9))
		mask = cv.dilate(mask, (9,9))
		mask = cv.blur(mask, (7,7))

		final_image = cv.inpaint(self.img, mask, 3, cv.INPAINT_TELEA)

		cv.imshow('Image', final_image)
		cv.waitKey(0)