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

    '''
    Core method, it orchestrates all of the image processing and handles the interactive UI
    '''
    def watermark_remover(self):
        # Gray scale by brightness using these proportions as the current version of opencv: I = 0.299R + 0.587G + 0.114B
        gray_scaled = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

        # Apply a median blur using a kernel of 23x23 in the whole image and store in a new variable
        blurred_img = cv.medianBlur(self.img, 23) # blurred image
        # blurred_img = self.fourier_low_pass(gray_scaled, 10) # fourier tentative

        # Get the water mark coordinates
        point1, point2 = self.get_selection_coordinates()

        # Create a mask for the selection
        background = np.zeros((self.img.shape[0], self.img.shape[1]), dtype='uint8')
        mask = cv.rectangle(background.copy(), point1, point2, color=255, thickness=-1)

        # Set the initial Canny's high threshold and configure the UI for enabling tune this parameter
        threshold = 180
        cv.namedWindow('Result')
        cv.createTrackbar('Threshold','Result', threshold, 255, self.callback)

        # Constantly process the image according to the Canny threshold and show it up on the UI
        while True:
            threshold = cv.getTrackbarPos('Threshold','Result')
            # Edge detection applying canny with the tuned threshold
            canny = cv.Canny(gray_scaled, threshold//3, threshold)

            # Do a dilate operation in order to fill the canny edges, it uses a 3x3 kernel
            canny = cv.dilate(canny,np.ones((3,3), np.uint8))

            # Apply the user selected area mask to deal only with the watermark
            canny_water_mark = cv.bitwise_and(canny, canny, mask=mask)

            # Water mark removal by substituting the water mark area by the median blurred image pixel by pixel on
            # the selected area
            img_cp = self.img.copy()
            for row in range(point1[1],point2[1]):
                for col in range(point1[0],point2[0]):
                    if canny_water_mark[row][col] > 150: # watermark area where canny detected edges
                        img_cp[row][col] = blurred_img[row][col]

            # Update the image on the UI
            cv.imshow('Result', img_cp)

            # Stop processing when the letter 'q' is pressed and then close the UI
            if cv.waitKey(1) &0xFF == ord('q'):
                break

    # Use ROI from OpenCV for the user to selected an area in the image and return its coordinates
    def get_selection_coordinates(self):
        roi = cv.selectROI(self.img)

        # x as rows and y as cols
        col1 = int(roi[0])
        col2 = int(roi[0] + roi[2])
        row1 = int(roi[1])
        row2 = int(roi[1] + roi[3])

        return (col1, row1), (col2, row2)

    # callback called everytime the user tune the trackbar on the UI, it is doing nothing
    def callback(val, val2):
        pass

    '''
    The functions below were used for testing a Fourier filter intentioned to allow only low frequencies and avoid
    high frequencies (edges) as a way to remove the water mark. It uses a gaussian circle mask for processing the 
    Fourier magnitude leaving only the low frequency part
    '''
    def fourier_low_pass(self, img, radius):
        fourier = np.fft.fft2(img) # fourier transform
        centered = np.fft.fftshift(fourier) # shift the high frequencies to the middle

        # magnitude_spectrum = 20 * np.log(np.abs(centered))
        # spectrum_img = np.abs(magnitude_spectrum).clip(0, 255).astype(np.uint8)
        # cv.imshow('Magnitude Spectrum', spectrum_img)

        # Gaussian mask for keeping only low frequencies (blur borders)
        mask = self.gaussian_mask(img.shape, radius)

        filtered = centered * mask # apply the mask on the magnitude

        # filtered_spectrum = 20 * np.log(np.abs(filtered))
        # rows, cols = filtered_spectrum.shape
        # for x in range(cols):
        #     for y in range(rows):
        #         filtered_spectrum[y, x] = round(filtered_spectrum[y, x]) if filtered_spectrum[y, x] > 0 else 0
        # filtered_img = np.abs(filtered_spectrum).clip(0, 255).astype(np.uint8)
        # cv.imshow('Spectrum with mask applied', filtered_img)

        shifted_back = np.fft.ifftshift(filtered) # shift back to the original frequency positions
        inv_fourier = np.fft.ifft2(shifted_back) # Inverse Fourier transform
        img_back = np.abs(inv_fourier).clip(0, 255).astype(np.uint8)  # convert to a opencv readable format

        return img_back

    # Create a gaussian mask by using the distance to bound the brightness
    def gaussian_mask(self, img_shape, radius):
        mask = np.zeros(img_shape[:2])
        rows, cols = img_shape[:2]
        center = (rows / 2, cols / 2)
        for x in range(cols):
            for y in range(rows):
                mask[y, x] = math.exp(((-self.distance((y, x), center) ** 2) / (2 * (radius ** 2))))
        return mask

    # Calculate the distance between two cartesian points
    def distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
