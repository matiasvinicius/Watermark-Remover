import cv2 as cv
import numpy as np
from numpy.lib.function_base import blackman

FILE_NAME = 'NOME.mp4'
video = cv.VideoCapture(FILE_NAME)
framerate = 60
video_mod = cv.VideoWriter('video2.mp4', cv.VideoWriter_fourcc('M','J','P','G'), framerate, (int(video.get(3)),int(video.get(4))))
is_true, frame = video.read()
roi = cv.selectROI(frame)

x_begin = int(roi[1])
x_end = int(roi[1]+roi[3])
y_begin = int(roi[0])
y_end = int(roi[0]+roi[2])

while True:
	is_true, frame = video.read()
	if not(is_true): break
	
	background = np.zeros((frame.shape[0], frame.shape[1]), dtype = 'uint8')
	rectangle = cv.rectangle(background.copy(), (y_begin, x_begin), (y_end, x_end), color = 255, thickness= -1)

	masked = cv.bitwise_or(frame, frame, mask = rectangle)
	masked = cv.resize(masked, None, fx=0.5, fy=0.5, interpolation= cv.INTER_AREA)
	masked = cv.resize(masked, (frame.shape[1], frame.shape[0]), interpolation= cv.INTER_AREA)
	masked = cv.blur(masked, (3, 3))
	masked = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)

	ret, thresh = cv.threshold(masked, 127, 255, cv.THRESH_BINARY)
	contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
	masked = cv.drawContours(masked, contours, 0, 255, thickness=cv.FILLED)
	masked = cv.erode(masked, (9,9))
	masked = cv.dilate(masked, (9,9))

	frame_mod = cv.inpaint(frame, masked, 3, cv.INPAINT_NS)
	
	cv.imshow('VIDEO MODIFICADO', frame_mod)
	video_mod.write(frame_mod)
	
	key = cv.waitKey(1)
	if key == ord("q"): break

cv.destroyAllWindows()

