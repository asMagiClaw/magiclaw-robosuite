# magiclaw-robosuite

<!-- ![gallery of_environments](docs/images/gallery.png) -->

[**[Homepage]**](https://magiclaw.gitbook.io/magiclaw-docs) &ensp; [**[Documentations]**](https://magiclaw.gitbook.io/magiclaw-docs)

**magiclaw-robosuite** is a forked version of [**robosuite**](https://github.com/ARISE-Initiative/robosuite). It includes the MagiClaw as a teleoperation device, which allows users to conveniently and intuitively control the 6-DoF end-effector pose and the gripper angle of the robot. More details about the MagiClaw can be found in our [Documentations](https://magiclaw.gitbook.io/magiclaw-docs).

Using the APIs and scripts in this repo, users can collect human demonstrations with the MagiClaw and train robot learning algorithms with the collected data. We provide a [demo script](robosuite/demos/demo_device_control.py) to showcase how to use the MagiClaw to control a robot in simulation. To collect demonstrations, please follow the instructions in [Documentations](https://magiclaw.gitbook.io/magiclaw-docs/documentation/resources/teleoperation-and-imitation-learning-with-robosuite).
