# me469_hw0 - Implementation and Assessment of a Particle Filter for Robot Localization

## Overview
The goal of this project was to use a particle filter to improve localization accuracy for a robot in a test environment. Odometry data containing the commanded linear and angular velocities, distance and heading measurements relative to known landmarks, and ground truth position data for the robot were all provided as part of the data set. The data was collected from robots at the Autonomous Space Robotics Lab at the University of Toronto. Thorough documentation of the robot platform, task and file formats, as well as video, are provided here: http://asrl.utias.utoronto.ca/datasets/mrclam/index.html.

<img src="https://scferro.github.io/assets/particle_filter_5.png"/>

Overall, the particle filter was successfully able to combine the odometry data and the measured position data to dramatically improve the position estimate relative to the pure dead reckoning estimate. As expected, the filtering algorithm still struggled in situations where no measurements were taken. However as long as measurements were taken regularly, the filter was able to keep the position estimate within ~0.25m of the ground truth position. 

## Quickstart
Use `run.py` to generate the graphs comparing the localization performance of dead reckoning model and the particle filter model. 

## Results and Assessment of Filter Performance

When comparing the particle filter output to the output from the pure dead reckoning model, there is a significant difference between the position estimate using the particle filter and the estimate using dead reckoning. Whereas the dead reckoning estimate drifted far away from the correct position and was effectively outputting a random θ, the particle filter estimate generally stayed very close to the ground truth position for the entire run of the robot. However, compared to the dead reckoning estimate, the position data is noticeably noisier. This makes sense intuitively, as the particle filter estimate is a representation of a probability distribution which will shift as new information is fed through measurements. The position estimate jumps around as it responds to new measurement data, while the dead reckoning is purely a function of the input and previous position; there is no uncertainty, and therefore no noise. 

<img src="https://scferro.github.io/assets/particle_filter_1.png"/>
<img src="https://scferro.github.io/assets/particle_filter_2.png"/>

This is further corroborated by the error plot below. The estimates from the particle filter have more noise, but they stay much closer to the ground truth values. The position error for the particle filter estimates is consistently below 0.25 meters, while the dead reckoning data smoothly drifts to 7+ meters of error. 

<img src="https://scferro.github.io/assets/particle_filter_3.png"/>

At several points, the error on the particle filter spikes noticeably. Many of those points are marked in red, indicating that at that time, the robot had not received a measurement for more than one minute. It makes sense that error would increase at those points, as the robot is not taking in any measurement data to correct for its dead reckoning drift. To improve performance, it would be ideal for the robot to constantly be taking in useful positional information that it can use to correct its position estimate. 
