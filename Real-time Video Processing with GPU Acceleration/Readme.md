Installations

Optional: TensorFlow with GPU Support (only if you have GPU): Skip if no GPU is available, or if you're unsure.

Install NVIDIA driver:
sudo ubuntu-drivers devices
sudo apt install nvidia-driver-<version>  # Use recommended version

Install CUDA and cuDNN (offline installers may be required if no internet)

Then install TensorFlow GPU:
pip install tensorflow

To test GPU availability:
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

Steps

Step 1: Create a Real-Time Video Processing Script
python3 realtime_gpu_filter.py

Step 2: Compare CPU vs GPU Performance (Optional)

a. To compare, disable the TensorFlow GPU device in code:
tf.config.set_visible_devices([], 'GPU')  # Forces CPU usage

b. Measure and note the Processing Time overlay on video frame

c. Re-enable GPU and compare time differences