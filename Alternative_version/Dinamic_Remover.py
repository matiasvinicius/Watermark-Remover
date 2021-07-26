import cv2 as cv
import numpy as np

class Remover:
	def __init__(self, path, framerate):
		self.video = cv.VideoCapture(path)
		self.framerate = framerate
	
	def coords_roi(self, img):
		roi = cv.selectROI(img)
		x_begin = int(roi[1])
		x_end = int(roi[1]+roi[3])
		y_begin = int(roi[0])
		y_end = int(roi[0]+roi[2])
		return [x_begin, x_end], [y_begin, y_end]
	
	def remove_watermark(self):
		video_mod = cv.VideoWriter('video_mod.mp4', cv.VideoWriter_fourcc('M','J','P','G'), self.framerate, (int(self.video.get(3)),int(self.video.get(4))))

		is_true = True
		for i in range(200):
			if not(is_true): break
			is_true, frame = self.video.read()
		x_roi, y_roi = self.coords_roi(frame)
		x_img, y_img = frame.shape[:2]

		while True:
			is_true, frame = self.video.read()
			if not(is_true): break

			black_background = np.zeros((x_img, y_img), dtype = 'uint8')
			white_rectangle = cv.rectangle(
				black_background.copy(), 
				(y_roi[0], x_roi[0]), 
				(y_roi[1], x_roi[1]), 
				color = 255, 
				thickness= -1)

			masked = cv.bitwise_or(frame, frame, mask = white_rectangle)

			masked = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)
			masked = cv.adaptiveThreshold(masked, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

			crop_border = cv.rectangle(black_background.copy(), (y_roi[0]+2, x_roi[0]+2), (y_roi[1]-2, x_roi[1]-2), color = 255, thickness= -1)
			masked = cv.bitwise_and(masked, masked, mask = crop_border)

			ret, thresh = cv.threshold(masked, 127, 255, cv.THRESH_BINARY)
			contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
			masked = cv.drawContours(masked, contours, -1, 255, thickness=4)

			masked = cv.erode(masked, (9,9))
			masked = cv.dilate(masked, (9,9))
			
			masked = cv.blur(masked, (3,3))

			frame_mod = cv.inpaint(frame, masked, 3, cv.INPAINT_TELEA)
			
			cv.imshow('Video', frame_mod)

			video_mod.write(frame_mod)

			key = cv.waitKey(1)
			if key == ord("q"): break

		cv.destroyAllWindows()