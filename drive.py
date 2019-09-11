#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
import cozmo
import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
import time
from collect_data import add_speed, go_forward, go_backward, turn_left, turn_right, tilt_up, tilt_down
from cozmo.util import degrees
from tensorflow import keras


def drive(robot: cozmo.robot.Robot, model, labels_list, labels_dict, frequency = 10.0):
    period = 1/frequency
    try:
        while True:
            start_time = time.time()
            img = robot.world.latest_image
            if img is not None:
                img = np.array(img.raw_image)
                # currently resizing images to 60 x 80
                img = cv2.resize(img, (80, 60))
                # required for model
                img = img / 255.0
                # need to add extra dimension for model
                img = np.reshape(img, (1, 60, 80, 3))
                predictions = model.predict(img)
                top_prediction = np.argmax(predictions, axis=1)[0]
                actions = labels_list[top_prediction]
                print("predicted action {}".format(actions))
                actions_list = actions.split()
                wheel_speeds = (0.0, 0.0)
                for action in actions_list:
                    if action == "stop":
                        raise SystemExit()
                    if action == "forward":
                        wheel_speeds = add_speed(wheel_speeds, go_forward)
                    elif action == "backward":
                        wheel_speeds = add_speed(wheel_speeds, go_backward)
                    elif action == "left":
                        wheel_speeds = add_speed(wheel_speeds, turn_left)
                    elif action == "right":
                        wheel_speeds = add_speed(wheel_speeds, turn_right)

                robot.drive_wheel_motors(wheel_speeds[0], wheel_speeds[1], l_wheel_acc=500, r_wheel_acc=500)

            end_time = time.time()
            sample_time = start_time - end_time
            if sample_time < period:
                time.sleep(period - sample_time)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
    except SystemExit:
        print("Stop action received")
    finally:
        robot.stop_all_motors()

def main(robot: cozmo.robot.Robot):
    # enable camera
    robot.camera.image_stream_enabled = True 
    # reset pose
    robot.set_head_angle(degrees(0)).wait_for_completed()
    robot.set_lift_height(0).wait_for_completed()
    # load model
    os.chdir("/home/hcrlab/cozmo/explainability/")
    model = keras.models.load_model("model/cozmo_drive_model.h5")
    # load labels dict/list
    with open("model/pickles/label_names.pkl", 'rb') as file:
        labels_dict, labels_list = pickle.load(file)
    drive(robot, model, labels_list, labels_dict)


if __name__ == "__main__":
    cozmo.run_program(main)