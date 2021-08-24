from crop import *

import argparse as ap
import textwrap
import cv2
import logging
import os
from os import environ
from PIL import Image
from datetime import datetime


class Core:

    def __init__(self, source_path='', destination_path=''):
        self.source_path = source_path
        self.destination_path = destination_path
        self.config = {}
        self.process_method = 1
        self.logger = self.initialize_logger()
        self.config_loaded = self.start_process()
        self.meth_ranges = (1, 5)
        if bool(self.config['use_ndk_standard']):
            environ['OPENCV_IO_ENABLE_JASPER'] = 'true'


    def start_process(self):
        '''
        Loads config file, parses args and creates source/destination directory.

        :return:    True, when config file loaded
        '''
        is_loaded, self.config = helper.load_config(self.logger)
        if is_loaded:
            self.logger.info('Application started')
            # Parse arguments
            parser = ap.ArgumentParser(
                description='Auto-crop images application used in Olomouc Research Library',
                epilog='Created as master\'s thesis in Dept. of Computer Science Palacky University Olomouc',
                formatter_class=ap.RawTextHelpFormatter)

            parser.add_argument('src', metavar='source', type=str, help='Source path to directory with images')

            help_text = textwrap.dedent('''\
                    Specifies which method should be used to process image.
                    Accepted values:
                        1 ... cropper based on Sobel edge detection,
                        2 ... cropper based on finding vertical lines,
                        3 ... cropper based on averaged image mask (default)
                        4 ... cropper based on edge detection by neural network
                        5 ... cropper based on bounding box by neural network
                    In the other case will be used default cropper.''')
            parser.add_argument('-m', metavar='--method',
                                type=int,
                                action='store',
                                dest='method',
                                help=help_text)

            if bool(self.config['allow_destination_directory']):
                parser.add_argument('dest', metavar='destination',
                                    type=str,
                                    help='Destination path to directory with processed images.')
            else:
                parser.add_argument('-dest', metavar='--destination',
                                    type=str,
                                    action='store',
                                    dest='opt_dest',
                                    help='Destination path to directory with processed images.')

            args = parser.parse_args()

            # Load process method from params
            if args.method is not None:
                self.process_method = int(args.method)
                self.logger.info(f'Process method ({args.method}) has been selected')

            # Load source path name from parameter
            if args.src is not None: self.source_path = args.src

            # Load destination path name from parameter or from config
            allow_dest_dir = bool(self.config['allow_destination_directory'])

            if allow_dest_dir and args.dest is not None:
                self.destination_path = args.dest
            elif args.opt_dest is not None:
                self.destination_path = args.opt_dest
            else:
                self.destination_path = str(self.config['destination_dir_name'])

            self.logger.info(f'Current source path: \"{self.source_path}\"')
            self.logger.info(f'Current destination path: \"{self.destination_path}\"')

            return True
        else:
            log_message = 'Application is unable to load configuration file'
            self.logger.critical(log_message)
            helper.print_info_message(log_message, False, 3)

            return False


    def crop(self, unprocessed):
        '''
        Processes (crop) each file with various methods.

        :param unprocessed: list with unprocessed (non-cropped) files
        '''
        try:
            self.create_destination_directory()

            # Available method range
            method = self.process_method
            in_range = method in range(self.meth_ranges[0], self.meth_ranges[1])
            if self.process_method is not None or in_range:
                count = 0
                processed_count = 0;
                file_label = 'file'
                files_count = len(unprocessed)
                use_time = bool(self.config['use_time'])
                use_ndk_standard = bool(self.config['use_ndk_standard'])

                log_message_state = 'Yes' if use_ndk_standard else 'No'
                self.logger.info(f'Use NDK standard: {log_message_state}')

                mask = False
                avg_crop = False
                c = Crop(self.config, method, self.logger)

                log_message = 'Post-processing has begun'
                self.logger.info(log_message)
                helper.print_info_message(log_message, use_time)

                # Mask is created only in case of process_method 3
                if method == 3:
                    mask_l, mask_r = self.create_avg_mask(unprocessed, use_time)
                    avg_crop = True

                # Initial call to print 0% progress
                helper.print_progress_bar(0, files_count, suffix='')

                for image in unprocessed:
                    count += 1
                    filename = os.path.splitext(os.path.split(image)[1])[0]

                    # Create filename
                    if use_ndk_standard:
                        filename += '.jp2'
                    else:
                        filename += str(self.config['store_extension'])
                        path = os.path.join(self.destination_path, filename)

                    update_label = f' (file: {filename} has been saved)'

                    # Select proper (left/right) mask and run crop
                    if method == 3: mask = mask_r if (count % 2) == 0 else mask_l

                    output = c.execute(image, mask, avg_crop)

                    # Save file
                    if use_ndk_standard:
                        self.save_compressed_file(filename, output)
                    else:
                        cv2.imwrite(path, output)

                    self.logger.info(update_label)
                    processed_count += 1

                    # Update progress bar
                    helper.print_progress_bar(processed_count, files_count, suffix=update_label)


                if processed_count > 1: file_label = 'files'

                # Print final messages
                processed_count = str(processed_count)
                file_label = str(file_label)
                dest_path = str(self.destination_path)
                message = f'{processed_count} {file_label} has been processed '
                message += f'and saved into {dest_path} directory'
                helper.print_info_message(message, False, 4)
                self.logger.info(message)
                log_message = 'Post-processing has ended'
                helper.print_info_message(log_message, use_time)
                self.logger.info(log_message)

            else:
                process_method = str(self.process_method)
                log_message = f'Undefined crop method {process_method}'
                self.logger.warning(log_message + '. Default method will be used.')
                helper.print_info_message(log_message, False, 2)
        except Exception as ex:
            if bool(self.config['dev_mode']): helper.print_info_message(str(ex), False, 3)
            self.logger.critical(str(ex))

    def create_destination_directory(self):
       '''
       Creates directory to save processed images.
       '''
       try:
           if not os.path.exists(self.destination_path):
               os.mkdir(self.destination_path)

           if bool(self.config['use_ndk_standard']):
               archival = os.path.join(self.destination_path, str(self.config['dest_dir_archival']))
               production = os.path.join(self.destination_path, str(self.config['dest_dir_production']))
               if not os.path.exists(archival): os.mkdir(archival)
               if not os.path.exists(production): os.mkdir(production)
       except OSError:
           log_message = 'Creation of the directory failed'
           self.logger.error(log_message)
           helper.print_info_message(log_message, False, 3)


    def load_unprocessed_files(self):
        '''
        Loads content filenames of directory acording to extension.

        :return:    list of filenames with available extensions
        '''
        content = []
        try:
            for ext in self.config['allowed_extensions']:
                for file in os.listdir(self.source_path):
                    if file.endswith(ext):
                        content.append(os.path.join(self.source_path, file))
            if len(content) <= 0:
                log_message = 'File with preffered extension/s has not been found'
                self.logger.warning(log_message)
                helper.print_info_message(log_message, False, 3)

            return content
        except Exception as ex:
            helper.print_info_message(str(ex), False, 3)
            self.logger.critical(str(ex))


    def create_avg_mask(self, dataset, use_time=False):
        '''
        Creates masks based on averaged image.

        :param image_left:  averaged image of all left (odd) book pages
        :param image_right: averaged image of all right (even) book pages
        :param use_time:    use time in terminal info message
        :return:            tuple of coordinates (left, right) of mask
        '''
        log_message = 'Creating mask from images'
        self.logger.info(log_message)
        helper.print_info_message(log_message, use_time)

        image_left, image_right = helper.create_avg_images(dataset,
                                                           self.config,
                                                           self.destination_path,
                                                           self.logger)
        rect_l, mask_l = helper.find_avg_mask(image_left)
        rect_r, mask_r = helper.find_avg_mask(image_right)

        if bool(self.config['save_mask']):
            mask_dir = str(self.config['mask_dir_name'])
            path_name = os.path.join(self.destination_path, mask_dir)
            path_name_l = os.path.join(path_name, 'mask_l.png')
            path_name_r = os.path.join(path_name, 'mask_r.png')

            if not os.path.exists(path_name): os.mkdir(path_name)

            cv2.imwrite(path_name_l, mask_l)
            cv2.imwrite(path_name_r, mask_r)

            message = f'Masks has been created and saved to \'{path_name}\''
            self.logger.info(message)
            helper.print_info_message(message, use_time)
        else:
            message = 'Mask has been created'
            self.logger.info(message)
            helper.print_info_message(message, use_time)

        return rect_l, rect_r


    def save_compressed_file(self, filename, image):
       '''
       Compress image as JPEG2000 according NDK standards.

       :param path:     string representing name of output file
       :param image:    image to commpress and save
       '''
       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       image = Image.fromarray(image)  # convert to PIL image

       # Save archival master copy
       path = os.path.join(str(self.destination_path), str(self.config['dest_dir_archival']), filename)
       image.save(path,
                  'JPEG2000',
                  tile_size=self.config['arch_tile_size'],
                  quality_mode='rates',
                  quality_layers=[int(self.config['arch_quality_layers'])],
                  codeblock_size=self.config['arch_codeblock_size'],
                  precinct_size=self.config['arch_precinct_size'],
                  irreversible=bool(self.config['arch_irreversible']),
                  progression=str(self.config['arch_progression']))

       # Save archival master copy
       path = os.path.join(str(self.destination_path),
                           str(self.config['dest_dir_production']), filename)
       image.save(path,
                  'JPEG2000',
                  tile_size=self.config['prod_tile_size'],
                  quality_mode='rates',
                  quality_layers=[int(self.config['prod_quality_layers'])],
                  codeblock_size=self.config['prod_codeblock_size'],
                  precinct_size=self.config['prod_precinct_size'],
                  irreversible=bool(self.config['prod_irreversible']),
                  progression=str(self.config['prod_progression']))


    def initialize_logger(self, level=logging.INFO):
        '''
        Creates log file and initialize logger.

        :param level:   which level will be saved into log files
        :return:        logger
        '''
        log_format = '[%(asctime)s] [%(levelname)s]:    %(message)s'
        logs_base_dir = 'logs/' + str(datetime.now().strftime('/%Y/%m'))
        log_filename = str(datetime.now().strftime('%d')) + '.log'
        logs_dir = os.path.join(logs_base_dir, log_filename)
        os.makedirs(logs_base_dir, exist_ok=True)

        logger = logging.getLogger('auto-crop')
        hdlr = logging.FileHandler(logs_dir)
        formatter = logging.Formatter(log_format)
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(level)

        return logger
