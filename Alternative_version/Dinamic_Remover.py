import cv2 as cv
import numpy as np

'''
References used:
- Processing: https://www.youtube.com/watch?v=oXlwWbU8l2o&t=9326s
- ROI: https://learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
'''

class Remover:
	def __init__(self, path, framerate):
		self.video = cv.VideoCapture(path)
		self.framerate = framerate
	
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
		#Save the modified video in the code directory (without sound)
		video_mod = cv.VideoWriter('video_mod.mp4', cv.VideoWriter_fourcc('M','J','P','G'), self.framerate, (int(self.video.get(3)),int(self.video.get(4))))

		#The watermark doesn't always appear at the beginning of the video, so we ran 200 frames to display the ROI screen
		is_true = True
		for i in range(200):
			if not(is_true): break
			is_true, frame = self.video.read()
		x_roi, y_roi = self.coords_roi(frame)
		x_img, y_img = frame.shape[:2]  #Frames width and height

		#For each frame of the video
		while True:
			is_true, frame = self.video.read()
			if not(is_true): break

			#Create a black background and white rectangle (with ROI coordinates) to act as a mask for the image
			black_background = np.zeros((x_img, y_img), dtype = 'uint8')
			white_rectangle = cv.rectangle(
				black_background.copy(), 
				(y_roi[0], x_roi[0]), 
				(y_roi[1], x_roi[1]), 
				color = 255, 
				thickness= -1)

			#Captures the rectangle that has the ROI coordinates
			masked = cv.bitwise_or(frame, frame, mask = white_rectangle)

			#Convert mask to grayscale and apply adaptive threshold
			masked = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)
			masked = cv.adaptiveThreshold(masked, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
		
			#Adaptive Threshold will paint white not only the outlines but also the edge of the ROI rectangle. 
			#To avoid this we create a smaller rectangle to use as a mask in an AND, removing this edge.
			crop_border = cv.rectangle(black_background.copy(), (y_roi[0]+2, x_roi[0]+2), (y_roi[1]-2, x_roi[1]-2), color = 255, thickness= -1)
			masked = cv.bitwise_and(masked, masked, mask = crop_border)

			#Take what is actually contoured on the mask (not isolated paintings) and paint it white. The rest goes black
			ret, thresh = cv.threshold(masked, 127, 255, cv.THRESH_BINARY)
			contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
			masked = cv.drawContours(masked, contours, -1, 255, thickness=4)

			#Erosion, dilation and an average blur on the edges to reduce the perception of different colors)
			masked = cv.erode(masked, (9,9))
			masked = cv.dilate(masked, (9,9))
			
			masked = cv.blur(masked, (3,3))
			
			#Paint contours by FMM method modified by TELEA
			frame_mod = cv.inpaint(frame, masked, 3, cv.INPAINT_TELEA)
			
			cv.imshow('Video', frame_mod)

			#Saves the frame in 'video_mod.mp4'
			video_mod.write(frame_mod)

			key = cv.waitKey(1)
			if key == ord("q"): break

		cv.destroyAllWindows()