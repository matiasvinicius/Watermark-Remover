import math
import cv2 as cv
import numpy as np

'''
References used:
- https://www.youtube.com/watch?v=oXlwWbU8l2o&t=9326s
- https://www.programmersought.com/article/8437785923/
- https://stackoverflow.com/questions/59975604/how-to-inverse-a-dft-with-magnitude-with-opencv-python
- https://hicraigchen.medium.com/digital-image-processing-using-fourier-transform-in-python-bcb49424fd82
'''

class Remover:
    def __init__(self, path):
        self.img = cv.imread(path)
    
    def get_selection_coordinates(self):
        roi = cv.selectROI(self.img)

        # x as rows and y as cols
        col1 = int(roi[0])
        col2 = int(roi[0] + roi[2])
        row1 = int(roi[1])
        row2 = int(roi[1] + roi[3])

        return (col1, row1), (col2, row2)

    def distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def gaussian_mask(self, img_shape, radius):
        mask = np.zeros(img_shape[:2])
        rows, cols = img_shape[:2]
        center = (rows / 2, cols / 2)
        for x in range(cols):
            for y in range(rows):
                mask[y, x] = math.exp(((-self.distance((y, x), center) ** 2) / (2 * (radius ** 2))))
        return mask

    def fourier_low_pass(self, img, radius):
        fourier = np.fft.fft2(img)
        centered = np.fft.fftshift(fourier)

        # Gaussian mask for keeping only low frequencies (blur borders)
        mask = self.gaussian_mask(img, radius)

        filtered = centered * mask
        shifted_back = np.fft.ifftshift(filtered)
        inv_fourier = np.fft.ifft2(shifted_back)
        img_back = np.abs(inv_fourier).clip(0, 255).astype(np.uint8)  # convert to a opencv readable format

        return img_back

    def callback(val):
        pass

    def watermark_remover(self):
        # Gray scale by brightness using these proportions as the current version of opencv: I = 0.299R + 0.587G + 0.114B
        gray_scaled = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        blurred_img = cv.medianBlur(self.img, 23) # blurred image

        # Get the water mark coordinates
        point1, point2 = self.get_selection_coordinates()

        # Mask for the selection
        background = np.zeros((self.img.shape[0], self.img.shape[1]), dtype='uint8')
        mask = cv.rectangle(background.copy(), point1, point2, color=255, thickness=-1)

        threshold = 180

        cv.namedWindow('Result')
        cv.createTrackbar('Threshold','Result', threshold, 255, self.callback)

        while True:
            threshold = cv.getTrackbarPos('Threshold','Result')
            # Edge detection
            canny = cv.Canny(gray_scaled, threshold//3, threshold)
            canny = cv.dilate(canny,np.ones((3,3), np.uint8))

            # Apply mask
            canny_water_mark = cv.bitwise_and(canny, canny, mask=mask)

            # Water mark removal by substituting the water mark area by the median blurred image
            img_cp = self.img.copy()
            for row in range(point1[1],point2[1]):
                for col in range(point1[0],point2[0]):
                    if canny_water_mark[row][col] > 150: # watermark area
                        img_cp[row][col] = blurred_img[row][col]

            cv.imshow('Result', img_cp)

            if cv.waitKey(1) &0xFF == ord('q'):
                break
