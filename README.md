# solution to comma_speed_challenge

https://github.com/commaai/speedchallenge


attempts:


0. all the following methods use a cropped ROI instead of the whole image


1. use two consecutive frames as input of a CNN, do regression, MSE converge at ~30 (network too simple/not enough iteration)


2. use two consecutive frames to calculate dense optical flow first, then use the flow image as input to a CNN, do regression, MSE converge at 15~20 (dense optical flow seems noisy)


3. choose a trapezoidal ROI and map it to rectangle using perspective transform, then use simple sliding window method to calculate the displacement between frames, then fit this disp to velocity, best training MSE ~6.3 with smoothing


4. replace sliding window method in 3) with SURF feature matching to calculate displacement of consecutive frames, then do linear fitting, best training MSE ~3.3 with smoothing

5. (future work) use the warped trapezoidal ROI for dense optical flow calculation (hopefully less noisy) and do regression using a CNN

result visualization for method 4:
  <br />
![](https://github.com/ZwX1616/comma_speed_challenge_sol/blob/master/v_by_surf/result/output%5B00-00-04--00-00-34%5D.gif)
![](https://github.com/ZwX1616/comma_speed_challenge_sol/blob/master/v_by_surf/result/train_3.png)
