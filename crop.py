import functions as helper
import crop_layer as cl

import cv2
import numpy as np
import os
import logging
from math import hypot
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import mimetypes


class Crop:

    def __init__(self, config, process_method, logger):
        self.config = config
        self.method = int(process_method)
        self.logger = logger
        if self.method == 4: self.load_nn_model(is_hed=True)
        if self.method == 5: self.load_nn_model(is_hed=False)


    def execute(self, image, mask=False, avg_crop=False):
        '''
        Calls selected method to process image.

        :param image:       image to process
        :param mask:        pre-created mask is used in case 3 - average image mask
        :param avg_crop:    when True, uses different crop method
        :return:            cropped image
        '''
        image_path = image
        image = cv2.imread(image)
        result = image.copy()
        method_name = 'Default'
        dev_mode = bool(self.config['dev_mode'])

        if self.method == 1:
            # Method based on classical edge detection algorithm (operator)
            result = self.classical_edge_detection(image, method=1)
            method_name = f'Classical method ({self.method})'
        elif self.method == 2:
            # Method based on finding vertical lines
            result = self.vertical_lines_detection(image)
            method_name = f'Finding vertical lines ({self.method})'
        elif self.method == 3:
            # Method uses mask from averaged image to crop
            result = self.average_image_mask(image, mask, avg_crop)
            method_name = f'Average mask ({self.method})'
        elif self.method == 4:
            # Method based on HED neural network edge detection
            result = self.neural_edge_detection(image)
            method_name = f'Neural edge detection ({self.method})'
        elif self.method == 5:
            # Method based on finding bounding box using neural network
            result = self.neural_bounding_box(image_path)
            method_name = f'Neural bounding box ({self.method})'
        else:
            # Sobel method by default
            result = self.classical_edge_detection(image, method=1)
            method_name = 'Classical method (1)'

            # Log warnings
            warn = f'Selected method is unavailable. Method automatically switched to default - {method_name}'
            self.logger.warning(warn)
            if self.method in [2, 4, 5]:
                dev_message = 'You\'re trying to run method available from developer\'s mode '
                self.logger.warning(dev_message + f'(method number {self.method}).')

        self.logger.info(f'Selected crop method: {method_name}')
        return result


    def classical_edge_detection(self, image, method=2):
        '''
        Creates edge detection (using selected method), than finds contours
        and bouding box, creates mask according to bounding box and than crops.
        List of supported methods:  1) Sobel operator (default)
                                    2) Laplacian operator
                                    3) Canny edge detection method

        :param image:   image to process
        :param method:  method used to edge detection
        :return:        cropped image
        '''
        edges = self.detect_edges(image, method)
        contours, _ = self.find_contours(edges)
        mask = self.create_mask(image, contours)

        return self.crop_image(image, mask)


    def vertical_lines_detection(self, image):
        '''
        Should be able to detect vertical lines in the image and create bounding
        box according to biggest found vertical line.
        Available in Developer's mode.

        :param image:   image to process
        :return:        cropped image
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        _, bw = cv2.threshold(gray, 133, 255, cv2.THRESH_BINARY)

        # Detect vertical lines
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
        detect_vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                                           vert_kernel, iterations=2)
        contours = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        # Find correct rectangle to crop, and crop it
        max_area = 0

        for c in contours:
            new_area = cv2.contourArea(c)
            if max_area < new_area:
                max_area = new_area
                rect = cv2.boundingRect(c)
                x, y, w, h = rect
                if w*h > 40000:
                    mask = np.zeros(image.shape, np.uint8)
                    mask[y:y+h, x:x+w] = image[y:y+h, x:x+w]
                    cropped_image = image[y:y+h, x:x+w]

        return cropped_image


    def average_image_mask(self, image, mask, is_avg=False):
        '''
        Crops image according to pre-counted mask from average image of dataset.

        :param image:   image from dataset
        :param mask:    pre-created mask from average image
        :param is_avg:  when True, uses different crop method
        :return:        cropped image
        '''
        return self.crop_image(image, mask, is_avg)


    def neural_edge_detection(self, image):
        '''
        Uses neural network based on Holistically-Nested Edge Detection
        to find edges, than finds bounding box and crops.
        Available in Developer's mode.

        :param image:   image to process
        :return:        b&w edged image
        '''
        original_w, original_h = image.shape[:2]
        input_image = image.copy()
        input_image = helper.resize(input_image, scale_percent=35)
        w, h = input_image.shape[:2]

        # Finds edges using HED neural network
        blob = cv2.dnn.blobFromImage(input_image, scalefactor=1.0, size=(w, h),
                                mean=(104, 117, 123), swapRB=False, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()

        out = helper.resize(out[0,0], sizes=(original_h, original_w))
        out_gray = (255 * out).astype(np.uint8)
        test = cv2.cvtColor(out_gray.copy(), cv2.COLOR_GRAY2BGR)
        out_gray = cv2.medianBlur(out_gray, 3)
        _, bw = cv2.threshold(out_gray, 75, 255, cv2.THRESH_BINARY)

        # Connects lines that probably belongs to each other
        #lines = cv2.HoughLinesP(out_gray,
                                #rho=1.,
                                #theta=np.pi/180.,
                                #threshold=100,
                                #minLineLength=10.,
                                #maxLineGap=150.)
        #for line in lines:
            #cv2.line(bw,
                     #(line[0][0], line[0][1]),
                     #(line[0][2], line[0][3]),
                     #color=[255, 255, 255],
                     #thickness=2)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = self.create_mask(image, contours)

        return self.crop_image(image, mask)


    def neural_bounding_box(self, image_path):
        '''
        Uses neural network to find bounding box and crop it.

        :param image_path:  path to image to process
        :return:            cropped image
        '''
        image = cv2.imread(image_path)
        (height, width) = image.shape[:2]

        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = self.model.predict(img)[0]
        (start_x, start_y, end_x, end_y) = preds

        start_x = int(start_x * width)
        start_y = int(start_y * height)
        end_x = int(end_x * width)
        end_y = int(end_y * height)

        w = hypot((start_x - end_x), (start_y - start_y))
        h = hypot((start_x - start_x), (start_y - end_y))
        bbox = int(start_x), int(start_y), int(w), int(h)

        return self.crop_image(image, bbox)


    def detect_edges(self, image, method=1):
        '''
        Finds the edges using method specified in parameter. Used as submethod
        of self.classical_method. This method uses Sobel operator by default.
        List of supported methods:  1) Sobel operator (default)
                                    2) Laplacian operator
                                    3) Canny edge detection method

        :param image:   image to process
        :param method:  method used to edge Detection
        :return:        image with edges
        '''
        low_thresh = 133
        up_thresh = 255
        kernel_size = 5

        image = helper.denoise(image)
        grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(grayscaled, 9)
        _, bw = cv2.threshold(blurred, low_thresh, up_thresh, cv2.THRESH_BINARY)

        # Case to use Sobel operator
        if int(method) == 1:
            sobel_x = cv2.Sobel(bw, cv2.CV_8UC1, 1, 0, ksize=kernel_size)
            sobel_y = cv2.Sobel(bw, cv2.CV_8UC1, 0, 1, ksize=kernel_size)
            #edges = cv2.addWeighted(sobel_y, 0.5, sobel_x, 0.5, 0)
            edges = cv2.bitwise_and(sobel_y, sobel_x)
        # Case to use Laplacian operator
        elif int(method) == 2:
            edges = cv2.Laplacian(grayscaled, cv2.CV_8UC1)
        # Case to use Canny method
        elif int(method) == 3:
            edges = cv2.Canny(grayscaled, low_thresh, up_thresh)
        return edges


    def find_contours(self, edges):
        '''
        Finds contours of image from paramater.

        :param edges:   edges of objects in images
        :return:        original image, founded contours
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.erode(closed, None, iterations=1)
        closed = cv2.dilate(closed, kernel, iterations=3)

        return cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    def create_mask(self, image, contours):
        '''
        Creates mask with margins to crop.

        :param image:       original input image
        :param contours:    contours used to create mask
        :return:            masked image
        '''
        is_found = False
        mask = np.zeros(image.shape, dtype=np.uint8)
        image = cv2.convertScaleAbs(image)

        for contour in contours:
            arc_len = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.1 * arc_len, True)
            if (len(approx) == 4):
                is_found = True
                mask = cv2.drawContours(mask, [approx], -1, (255, 255, 255), -1)
                # masked_image = cv2.bitwise_and(image, mask)
        if is_found:
            grayscaled = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(grayscaled, 133, 255, cv2.THRESH_BINARY)

            return cv2.boundingRect(bw)


    def crop_image(self, image, mask, is_avg=False):
        '''
        Crops an image according to found bounding rectangle.

        :param image:   image to crop
        :param mask:    bounding box or image with mask of area to crop
        :param is_avg:  when True, it uses another crop function
        :return:        cropped image
        '''
        if is_avg:
            return self.crop_rotated(image, mask)
        else:
            x, y, w, h = mask
            return image[y:y+h, x:x+w]


    def crop_rotated(self, image, bbox):
        '''
        Align image, when it's rotated, and crops it.

        :param image:   image to crop
        :param bbox:    rectangle created by cv2 function minAreaRect
        :return:        cropped image
        '''
        # Rotate image
        angle = bbox[2] if bbox[2] == -90 else 0
        rows, columns = image.shape[0], image.shape[1]
        matrix = cv2.getRotationMatrix2D((columns/2, rows/2), angle, 1)
        rotated_img = cv2.warpAffine(image, matrix, (columns, rows))

        # Rotate bounding box
        rotated_bbox = (bbox[0], bbox[1], 0.0)
        box = cv2.boxPoints(rotated_bbox)
        points = np.int0(cv2.transform(np.array([box]), matrix))[0]
        points[points < 0] = 0

        return rotated_img[points[1][1]:points[0][1], points[1][0]:points[2][0]]


    def load_nn_model(self, is_hed=False):
        '''
        Service method, which loads specified model to neural network.

        :param is_hed:  True when we want to load HED caffemodel else False
        '''
        base_path = self.config['nn_base_dir']
        try:
            if is_hed:
                self.logger.info('Loading HED model')
                deploy =  os.path.sep.join([base_path, 'caffe', 'deploy.prototxt'])
                caffemodel = os.path.sep.join([base_path, 'caffe', 'hed.caffemodel'])

                self.net = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
                self.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
                cv2.dnn_registerLayer('Crop', cl.CropLayer)

            else:
                self.logger.info('Loading model')
                base_output = os.path.sep.join([base_path, 'tf'])
                model_path = os.path.sep.join([base_output, 'detector.h5'])

                self.model = load_model(model_path)

        except Exception as ex:
            if bool(self.config['dev_mode']): helper.print_info_message(str(ex), False, 3)
            self.logger.critical(str(ex))
