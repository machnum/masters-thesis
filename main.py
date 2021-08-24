from core import *
import functions as helper

import cv2
import numpy as np
import os
from datetime import datetime


def click_event(event, x, y, flags, params):
    '''
    Function used to detect mouse clicks and add point to image.

    :param event:   event detecting which button is pressed
    :param x:       x coordinate of mouse click
    :param y:       y coordinate of mouse click
    :param flags:   ...
    :param params:  another params sent to event
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        global CLICKS
        CLICKS = CLICKS - 1
        coords = f'{str(x)},{str(y)}'
        coords += ',' if CLICKS == 1 else ''
        params[1].write(coords)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 0)
        cv2.putText(params[2], f'{str(x)}, {str(y)}', (x,y), font, 1, color, 2)
        cv2.imshow(params[0], params[2])

        if CLICKS <= 0:
            cv2.destroyAllWindows()
            return


def create_dataset_manually(unprocessed_files, config):
    '''
    Function used to manual creation of dataset to train neural network which
    finds bounding box.

    :param unprocessed_files:   list of paths to images
    :param config:              configuration file
    '''
    try:
        helper.print_info_message('Dataset creation start')

        for file in unprocessed_files:
            global CLICKS
            CLICKS = 2
            error = False

            img = cv2.imread(file, 1)
            img = helper.resize(img, scale_percent=15)
            filename = file.split('\\')[-1]
            nn_base_dir = config['nn_base_dir']
            path = os.path.sep.join([nn_base_dir, 'tf', 'train', filename])
            cv2.imwrite(path, img)
            cv2.imshow(file, img)

            bounds_path = os.path.sep.join([nn_base_dir, 'tf', 'bounds.txt'])
            with open(bounds_path, 'a+', encoding='UTF-8') as bounds:
                bounds.write(filename + ',')

                cv2.setMouseCallback(file, click_event, [file, bounds, img])
                key = cv2.waitKey()

                # Pressed ESC key
                if key == 27:
                    error = True
                    message = 'Dataset creation has ended, but it could be corrupted'
                    helper.print_info_message(message)
                    print('')
                    bounds.write('\n')
                    bounds.close()
                    cv2.destroyAllWindows()
                    break

                bounds.write('\n')
            helper.print_info_message(f'{file} has been saved to dataset')
            cv2.destroyAllWindows()

        message = 'Dataset has been sucessfuly created' if not error else ''
        helper.print_info_message(message)
    except Exception as ex:
        if bool(config['dev_mode']): print_info_message(str(ex), False, 3)
        logger.critical(str(ex))


if __name__ == '__main__':
    CLICKS = 2
    action = 1
    core = Core()

    if core.config_loaded:
        unprocessed_files = core.load_unprocessed_files()
        config = core.config

        if action == 1:
            core.crop(unprocessed_files)
        elif action == 2 and bool(config['dev_mode']):
            create_dataset_manually(unprocessed_files, config)
        elif action == 3 and bool(config['dev_mode']):
            helper.neural_network_train(config, core.logger)
        else:
            helper.print_info_message('Unspecified method', False, 3)
    else:
        helper.print_info_message('Config file has not been loaded', False, 3)
