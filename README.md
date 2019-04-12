# solution to comma_speed_challenge

https://github.com/commaai/speedchallenge


attempts:


0. all the following methods use a cropped ROI instead of the whole image


1. use two consecutive frames as input of a CNN, do regression, MSE converge at ~30 (network too simple/not enough iteration)


2. use two consecutive frames to calculate dense optical flow first, then use the flow image as input to a CNN, do regression, MSE converge at 15~20 (dense optical flow seems noisy)


3. choose a trapezoidal ROI and map it to rectangle using perspective transform, then use simple sliding window method to calculate the displacement between frames, then fit this disp to velocity, best training MSE ~6.3 with smoothing


4. method 3 suffers from random noise a lot, will try sparse optical flow to replace sliding window