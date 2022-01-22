# ROS Wrapper for AirDet

A ROS Node package for AirDet.

## On PC (Simulation)


### Installation

It is necessary to install Detectron2 [requirements](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) in a python *virtual environment* as it requires `Python 3.6` and ROS works with `Python 2.7`

1. Install ROS Melodic following:

   http://wiki.ros.org/melodic/Installation/Ubuntu

2. Install python Virtual Environment

```bash
sudo apt-get install python-pip
sudo pip install virtualenv
mkdir ~/.virtualenvs
sudo pip install virtualenvwrapper
export WORKON_HOME=~/.virtualenvs
echo '. /usr/local/bin/virtualenvwrapper.sh' >> ~/.bashrc 
```

3. Creating Virtual Environment

```bash
mkvirtualenv --python=python3 detectron2_ros
```

4. [Install the dependencies in the virtual environment](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

We use torch==1.5.1 (recommend) and detectron2==0.2

```bash
workon detectron2_ros
pip install torch==1.5.1 torchvision==0.6.1
pip install cython pyyaml
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
python -m pip install detectron2==0.2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.5/index.html
pip install opencv-python
pip install rospkg
pip install fvcore
pip install pillow
```

### Docker

Also, we provide a pre-built docker image here, with the aforementioned libraries installed:

```
docker pull bowenli1024/airdet-ros:v1
```

## On AGX Xavier (Robot Platform)

### Installation

1. Install ROS Melodic following:

   http://wiki.ros.org/melodic/Installation/Ubuntu

2. Install python Virtual Environment

   ```shell
   sudo apt-get install python-pip
   sudo pip install virtualenv
   mkdir ~/.virtualenvs
   sudo pip install virtualenvwrapper
   export WORKON_HOME=~/.virtualenvs
   echo '. /usr/local/bin/virtualenvwrapper.sh' >> ~/.bashrc 
   ```

3. Creating Virtual Environment

   ```
   mkvirtualenv --python=python3 detectron2_ros
   ```

4. Install dependencies

   ```shell
   workon detectron2_ros
   pip install opencv-python
   pip install rospkg
   pip install fvcore
   pip install pillow
   ```

   ##### Install pytorch 1.6.0 following:

   https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048

   first download [whl file](https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl)

   Then run

   ```shell
   sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
   pip3 install Cython
   pip3 install numpy torch-1.6.0-cp36-cp36m-linux_aarch64.whl
   ```

   ##### Install torchvision:

   ```shell
   sudo apt-get install libjpeg-dev zlib1g-dev
   git clone --branch v0.7.0 https://github.com/pytorch/vision torchvision
   cd torchvision
   sudo python3 setup.py install
   ```

   ##### Install detectron2 by building source code

   Download [source code](https://github.com/facebookresearch/detectron2/archive/refs/tags/v0.2.tar.gz)

   Then extract the compressed file and build the library

   ```shell
   python -m pip install -e detectron2
   ```


## Downloading the Package

Clone the package to the ROS workspace using git tools. The folder should be:

AirDet_ROS/

--build/

--devel/

--src/

----airdet_ros/

```bash
git clone https://github.com/Jaraxxus-Me/airdet_ros.git
mv airdet_ros AirDet_ROS/src/
```

## Compilation

##### Attention: DO NOT USE the python virtual environment previously built to compile catking packages.

```bash
cd ./AirDet_ROS
catkin_make
source $HOME/.bashrc
source devel/setup.bash
```

## Running

1. First launch ROScore into a terminal.

2. Next, open a new terminal and use the virtual environment created.

   ```shell
   workon detectron2_ros
   cd AirDet_ROS
   catkin_make
   source devel/setup.bash
   roslaunch airdet_ros airdet_ros.launch
   ```

   you should see rviz window

   ##### Arguments

   The following arguments can be set on the `roslaunch` above.

   - `input`: image topic name
   - `detection_threshold`: threshold to filter the detection results [0, 1]
   - `visualization`: True or False to pubish the result like a image
   - `model`: path to the training model file.

3. Finally, play a bag

   ```
   rosbag play -r 0.5 [record bag].bag
   ```