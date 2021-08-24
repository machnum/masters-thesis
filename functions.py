import cv2
import numpy as np
from datetime import datetime
import yaml
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def print_progress_bar(count, total, suffix=''):
    '''
    Prints progress bar.

    :param count:   current iteration
    :param total:   total iterations
    :param suffix:  label on tthe end
    '''
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '█' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '% Complete', suffix))
    if count == total: sys.stdout.write("\n")
    sys.stdout.flush()


def print_info_message(message, use_time=False, status=1):
    '''
    Prints info message.

    :param messsage:        message to print
    :param use_time:        if True, prints current time when message is printed
    '''
    statuses = {1: 'INFO', 2: 'WARNING', 3: 'ERROR', 4: 'DONE'}

    info_message = f'[{statuses.get(status)}]: ' + message

    if use_time:
        time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        info_message += f' in time {str(time)}...'

    print(info_message)


def resize(image, scale_percent=100, sizes=()):
    '''
    Changes image's size.

    :param image:           image to resize
    :param scale_percent:   percent to scale image
    :sizes:                 width and height which shoud have resized result
    :return:                resized image
    '''
    new_size = ()

    if len(sizes) == 0:
        width = (image.shape[1] * scale_percent) / 100
        height = (image.shape[0] * scale_percent) / 100
        new_size = (int(width), int(height))
    else:
        new_size = sizes

    return cv2.resize(image, new_size)


def denoise(image):
    '''
    Probably denoises image.

    :param image:       image to denoise
    :return:            denoised image
    '''
    morph = image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # take morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)

    # split the gradient image into channels
    channels = np.split(np.asarray(morph), 3, axis=2)
    channel_height, channel_width, _ = channels[0].shape

    # apply Otsu threshold to each channel
    for i in range(0, 3):
        _, channels[i] = cv2.threshold(channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        channels[i] = np.reshape(channels[i], newshape=(channel_height, channel_width, 1))
    channels = np.concatenate((channels[0], channels[1], channels[2]), axis=2)

    return channels


def reorder(points):
    '''
    Reorders biggest contour points.

    :param points:      biggest contour
    :return:            reordered points of biggest contour
    '''
    points = points.reshape((4, 2))
    reordered_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)

    reordered_points[0] = points[np.argmin(add)]
    reordered_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    reordered_points[1] = points[np.argmin(diff)]
    reordered_points[2] = points[np.argmax(diff)]

    return reordered_points


def biggest_contour(contours):
    '''
    Finds biggest contour from list of contours.

    :param contours:    list of all detected contours
    :return:            biggest contour and area of biggest contour
    '''
    biggest = np.array([])
    max_area = 0

    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area


def draw_rectangle(image, biggest, color=(0, 255, 0)):
    '''
    Draws rectangle around biggest contour from parameter.

    :param image:         input image
    :param biggest:       biggest contour
    :param color:         BGR color of rectangle
    :return:              image with drawed rectangle
    '''
    thickness = 3
    cv2.line(image, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), color, thickness)
    cv2.line(image, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), color, thickness)
    cv2.line(image, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), color, thickness)
    cv2.line(image, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), color, thickness)
    return image


def create_avg_images(dataset, config, destination_path, logger):
    '''
    Creates b&w average of images composed from all images of dataset.

    :param dataset:             collection of image's pathnames
    :param config:              loaded config
    :param destination_path:    destination path... for developer's purposes
    :param logger:              logger
    :return:                    pair of b&w image created from each image of dataset
    '''
    avg_left = []
    avg_right = []
    is_size_set = False
    count = 0

    # Initial call to print 0% progress
    print_progress_bar(0, len(dataset), suffix='')

    tmp_dir = str(config['mask_dir_name'])
    path_name = os.path.join(destination_path, tmp_dir)
    if not os.path.exists(path_name): os.mkdir(path_name)
    conflicts = os.path.join(path_name, 'conflicts.txt')
    with open(conflicts, 'a+', encoding='UTF-8') as f:
        f.write('Mask doesn\'t contains following images. These couldn\'t be cropped correctly:\n')

    for path_name in dataset:
        image = cv2.imread(path_name, cv2.IMREAD_GRAYSCALE)
        #bw = original_bw * 1.5
        #bw = bw.astype('uint8')
        #image = cv2.imread(path_name)
        #image = image.astype('uint8')
        #bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
        count += 1
        ignore = False

        if not is_size_set:
            blank_image = 255 * np.ones(bw.shape, np.uint8)
            if count == 1: avg_left = blank_image.copy()
            if count == 2: avg_right = blank_image.copy()
            is_size_set = True if count == 2 else False

        is_conflict_even = bool((count % 2)==0 and avg_right.shape != bw.shape)
        is_conflict_odd = bool((count % 2)!=0 and avg_left.shape != bw.shape)
        if is_conflict_even or is_conflict_odd:
            #bw = cv2.resize(bw, (avg_right.shape[1], avg_right.shape[0]))
            #logger.warning(f'Image ({path_name}) has been resized.')
            ignore = True
            with open(conflicts, 'a+', encoding='UTF-8') as f:
                f.write(f'{path_name}\n')

        if not ignore:
            # naive recognition of left/right page
            if (count % 2) == 0:
                avg_right = cv2.bitwise_and(bw, avg_right)
            else:
                avg_left = cv2.bitwise_and(bw, avg_left)

            print_progress_bar(count, len(dataset), suffix='')
        else:
            message = f'Image ({path_name}) has been ignored during mask creating.'
            logger.warning(message)

    #avg_left = cv2.blur(avg_left, (3,3))
    #avg_right = cv2.blur(avg_right, (3,3))
    #_, avg_left = cv2.threshold(avg_left, 100, 255, cv2.THRESH_BINARY)
    #_, avg_right = cv2.threshold(avg_right, 100, 255, cv2.THRESH_BINARY)

    if config['dev_mode']:
        tmp_dir = str(config['mask_dir_name'])
        path_name = os.path.join(destination_path, tmp_dir)
        path_name_l = os.path.join(path_name, 'avg_image_l.png')
        path_name_r = os.path.join(path_name, 'avg_image_r.png')
        if not os.path.exists(path_name): os.mkdir(path_name)

        cv2.imwrite(path_name_l, avg_left)
        cv2.imwrite(path_name_r, avg_right)

    return avg_left, avg_right


def find_avg_mask(image):
    '''
    Finds mask of averaged image.

    :param image:       input image
    :return:            found bounding rectangle, image with mask
    '''
    blurred = cv2.blur(image, (7,7))
    blurred = cv2.medianBlur(blurred, 7)
    _, bw = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, np.uint8)
    sorted_contours = sorted(cnts, key=cv2.contourArea, reverse=True)

    # bounding box is not tight enough using this option
    # x, y, w, h = bbox = cv2.boundingRect(sorted_contours[0])
    # cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)

    # this option works with tighter bounding box
    bbox = cv2.minAreaRect(sorted_contours[0])
    points = np.int0(cv2.boxPoints(bbox))
    cv2.drawContours(mask, [points], 0, (255, 255, 255), -1)

    return bbox, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def load_file(filename, key, logger):
    '''
    General function to loading YAML files.

    :param filename:    name of file, which should be loaded
    :param key:         key means section of yaml file
    :param logger:      logger
    :return:            boolean, when the file is loaded, returns true
    '''
    dest_file = []

    try:
       with open(filename, 'r', encoding='UTF-8') as yaml_content:
           content = yaml.safe_load(yaml_content)
       dest_file = content[key]
       logger.info(f'File with {key} loaded')

       return True, dest_file

    except IOError:
       message = 'The application is unable to load file'
       print_info_message(message, False, 3)
       logger.critical(message)

       return False, dest_file


def load_config(logger, filename='conf/config.yaml'):
    '''
    Loads configuration file.

    :param logger:      logger
    :param filename:    configuration file, default is config.yaml
    :return:            boolean, when the file is loaded, returns true
    '''
    is_loaded, config = load_file(filename, 'configuration', logger)

    return is_loaded, config


def neural_network_train(config, logger):
    '''
    Function used to train neural network to detect bounding box.

    :param config:      loaded configuration file
    :param logger:      logger
    '''
    try:
        base_path = config['nn_base_dir']
        base_output = os.path.sep.join([base_path, 'tf'])
        images_path = os.path.sep.join([base_path, 'tf', 'train'])
        annots_path = os.path.sep.join([base_path, 'tf', 'bounds.txt'])
        model_path = os.path.sep.join([base_output, 'detector.h5'])
        plot_path = os.path.sep.join([base_output, 'plot.png'])
        test_image_names = os.path.sep.join([base_output, 'test_images.txt'])
        init_learn_rate = 1e-4
        epochs = 25
        batch_size = 32
        data = []
        targets = []
        filenames = []

        if not os.path.exists(base_output): os.mkdir(base_output)
        if not os.path.exists(images_path): os.mkdir(images_path)

        # Load the content of bounds.txt
        message = 'Loading dataset'
        print_info_message(message, True)
        logger.info(message)
        rows = open(annots_path).read().strip().split('\n')

        message = 'Dataset contents'
        print_info_message(message)
        for row in rows:
            row = row.split(',')
            logger.info(f'{message}: {row}')
            print(row)
            (filename, start_x, start_y, end_x, end_y) = row

            image_path = os.path.sep.join([images_path, filename])
            #print_info_message(f'Train image will be stored in: {image_path}')
            logger.info(f'Loaded image: {image_path}')

            image = cv2.imread(image_path)
            (height, width) = image.shape[:2]
            start_x = float(start_x) / width
            start_y = float(start_y) / height
            end_x = float(end_x) / width
            end_y = float(end_y) / height

            # Image preprocessing
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)

            data.append(image)
            targets.append((start_x, start_y, end_x, end_y))
            filenames.append(filename)

        data = np.array(data, dtype='float32') / 255.0
        targets = np.array(targets, dtype='float32')
        split = train_test_split(data, targets, filenames, test_size=0.10, random_state=42)

        (train_images, test_images) = split[:2]
        (train_targets, test_targets) = split[2:4]
        (train_filenames, test_filenames) = split[4:]

        # Test files
        message = 'Saving testing filenames'
        print_info_message(message, True)
        logger.info(f'{message}: {test_image_names}')
        print(test_image_names)
        test_file = open(test_image_names, 'w', encoding='UTF-8')
        test_file.write('\n'.join(test_filenames))

        # Neural network
        vgg = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
        vgg.trainable = False
        flatten = vgg.output
        flatten = Flatten()(flatten)
        bbox_head = Dense(128, activation='relu')(flatten)
        bbox_head = Dense(64, activation='relu')(bbox_head)
        bbox_head = Dense(32, activation='relu')(bbox_head)
        bbox_head = Dense(4, activation='sigmoid')(bbox_head)
        model = Model(inputs=vgg.input, outputs=bbox_head)

        # Optimizer
        opt = Adam(lr=init_learn_rate)
        model.compile(loss='mse', optimizer=opt)
        print(model.summary())
        logger.info(model.summary())

        message = 'Training bounding box regressor'
        print_info_message(message, True)
        logger.info(message)
        H = model.fit(train_images,
                      train_targets,
                      validation_data=(test_images, test_targets),
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1)

        message = 'Saving object detector model'
        print_info_message(message, True)
        logger.info(message)
        model.save(model_path, save_format='h5')

        # Result chart creation
        logger.info('Creating result chart')
        n = epochs
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(np.arange(0, n), H.history['loss'], label='train_loss')
        plt.plot(np.arange(0, n), H.history['val_loss'], label='val_loss')
        plt.title('Bounding Box Regression Loss on Training Set')
        plt.xlabel('Epoch #')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig(plot_path)

    except Exception as ex:
        if bool(config['dev_mode']): print_info_message(str(ex), False, 3)
        logger.critical(str(ex))
