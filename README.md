# Guideline of PatternTrack

* Note: This README is still actively being updated.

## Installation
To start this project, you need two devices with LiDAR that has the same pattern as iPhone, iPad, or Vision Pro.

To use python script of this repo, install all required libraries by running
```
pip install -r script/requirements.txt
```

## Camera Calibration

The iPhone's IR image is not exposed by API. You may need to install an IR camera on to your device. After firmly attech the IR camera to the device with LiDAR, follow these procedure to calibrate the IR camera to match with iPhone's built-in  RGB camera.

```
cd calibration/
python camera_calibration.py -path <calibration image path>
python align_images.py -path <calibration image path>
```
The example output can be found in `data` folder.


## Template Pattern Generation
To apply PnP and get 3D pose of projection device, we need to know the template pattern of LiDAR emission. To generate pattern, execute following scripts.
```
cd script/
python pattern.py
python gen_sparse_diamond.py
```

## RGB & Depth Streaming
Install the iOS app to stream depth and RGB using `ios/VideoStreaming.xcworkspace`.


## IR Streaming
You may need raspberry pi to get IR video stream. Run below in the raspberry pi. You may also consider running this at startup. Edit `.bashrc` like below:

```
echo PatternTrack IR streaming
sudo python main.py
```


## Quick Start
To start demo, run
```
cd script/
python demo.py
```

Then connect to your laptop from the iPhone by tap `connect` on iOS app.
