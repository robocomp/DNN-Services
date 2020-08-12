# Python RetinaNet Object Detection Component

`objectDetectionRetinaNet` component is an object detection component that uses *RetinaNet* archiecture to detection objects from a single RGB image. RetinaNet is an object detection deep neural network that relies on focal loss for dense object detection. For more information, refer to the original paper ([Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)).

## Installation

-   Install dependencies :
```bash
# system packages
apt-get install tk-dev python-tk
# python packages
pip install torch torchvision
pip install pandas
pip install pycocotools
pip install opencv-python
pip install requests
```

-   Download pretrained weights [here](https://drive.google.com/file/d/1yLmjq3JtXi841yXWBxst0coAgR26MNBS/view) and move it to `src/dnn_lib/models`.

## Configuration parameters

Like any other component, *objectDetectionRetinaNet* needs a configuration file to start. In `etc/config`, you can change the ports and other parameters in the configuration file, according to your setting.

## Starting the component

To run `objectDetectionRetinaNet` component, navigate to the component directory :
```bash
cd <objectDetectionRetinaNet's path> 
```

Then compile the component :
```bash
cmake .
make
```

Then run the component :
```bash
python3 src/objectDetectionRetinaNet.py etc/config
```
