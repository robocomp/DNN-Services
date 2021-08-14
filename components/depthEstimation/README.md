# DepthEstimation

This component finds out **Depth Value** from the image feed. **getDepthEstimation** method takes image as input from camera and returns depth values of each pixel in form of numpy matrix with same size as of input image. These depth values help in locating any object w.r.t camera.

## Resolving dependencies

This section assumes the user has already installed the RoboComp core library and pulled Robolab's components according to this [README guide](https://github.com/robocomp/robocomp).

Before using the component, the user needs to install the necessary libraries:
```
pip3 install numpy opencv-python tensorflow==2.3
```

## Configuration parameters
As any other component, *DepthEstimation* needs a configuration file to start. In
```
etc/config
```
you can find an example of a configuration file. We can find there the following lines:
```
# Endpoints for implements interfaces
DepthEstimation.Endpoints=tcp -p 10100

Ice.Warn.Connections=0
Ice.Trace.Network=0
Ice.Trace.Protocol=0
```
After configuring proxies, 

For detection using Transfer Learning, download depth estimation models from [Models](https://drive.google.com/drive/folders/151knPx2eC1ufAO8YoRx9GlPkPCy3-QZ8?usp=sharing), move it to the assets folder. Also, set **self.method = 'mobdepthwithskip'** in `src/specificworker.py`for using *mobdepthwithskip.hdf5"* model or set **self.method = 'mobdepthwithoutskip'** in `src/specificworker.py`for using *mobdepthwithoutskip.hdf5"* model.


## Starting the component
To avoid changing the *config* file in the repository, we can copy it to the component's home directory, so changes will remain untouched by future git pulls:

```
cd ~/robocomp/components/DNN-Services/components/depthEstimation/
```

After editing the config file we can run the component:

```
cmake .
make
python3 src/DepthEstimation.py etc/config-run
```

