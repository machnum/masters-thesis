## About

This application is command line tool used to batch processing of images using operation as mainly crop, dewarp and deskew. This application was developed for use in Olomouc Research Library.

## Application structure
```bash
.
├── conf
│   ├── models
│   │   ├── caffe
│   │   │   ├── deploy.prototxt
│   │   │   └── hed.cafffemodel
│   │   └── tf
│   │       └── detector.h5
│   └── config.yaml
├── data
│   └── processed
├── logs
├── core.py
├── crop.py
├── crop_layer.py
├── functions.py
├── main.py
├── modules.txt
└── README.md
```

## Installation
1. Copy whole directory with application to your hard drive.
2. Install Python 3.7.8 and all dependecies included in file ```modules.txt``` located in root directory using following command:
```bash
pip install -r modules.txt
```
3. Create directory called ```logs``` in your copy of application and set its permissions to write.
4. Create directory called ```data``` in your copy of application and set its permissions to write.
5. Check if exists in your application following files and directories:
   * configuration file ```conf/config.yaml```
   * pretrained models as: ```conf/models/caffe/deploy.prototxt```, ```conf/models/caffe/hed.caffemodel``` and ```conf/models/tf/detector.h5```
   * directory ```logs``` in root of your application structure
   * directory ```data``` in root of your application structure and directory ```data/proccessed```

### Dependencies
Install [Python 3.7.8](https://www.python.org/ftp/python/3.7.8/) and following modules:

```python
opencv-python==4.5.2.52
numpy==1.20.3
Pillow==8.2.0
tensorflow==2.0.0
h5py==2.10.0
Keras-Preprocessing==1.1.0
PyYAML==5.4.1
scikit-learn==0.24.2
matplotlib==3.4.2
```


## Usage
In the blocks of code bellow you can find, how to properly run an application.

```bash
# parameters:
# foo/bar = source path (always required)

python main.py foo/bar
```

```bash
# parameters:
# foo/bar = source path (always required)
# -d      = destination path (optional)

python main.py foo/bar -d foo/bar/result
```

```bash
# parameters:
# foo/bar = source path (always required)
# -m      = method type (optional, default = 1)

python main.py foo/bar -m 1
```

```bash
# parameters:
# foo/bar = source path (always required)
# -d      = destination path (optional)
# -m      = method type (optional, default = 1)

python main.py foo/bar -d foo/bar/result -m 1
```

## Contact information to the programmer
email: [libor.machalek01@upol.cz](mailto:libor.machalek01@upol.cz)

## License
GNU/GPL - GNU GENERAL PUBLIC LICENSE
