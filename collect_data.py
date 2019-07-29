#!/usr/bin/env python3
# TODO: why doesn't head angle always set properly?
import concurrent.futures
import cozmo
import csv
import os
import pygame
import time
from argparse import ArgumentParser
from cozmo.util import degrees
from datetime import datetime


def write_data(robot: cozmo.robot.Robot, append=False, verbose=False):
    def add_speed(wheel_speeds: tuple, addition: tuple):
        return (wheel_speeds[0] + addition[0], wheel_speeds[1] + addition[1])
        
    pygame.init()
    screen = pygame.display.set_mode((100,100))
    
    # all speeds in millimiters per second
    # (left wheel speed, right wheel speed)
    go_forward = (100, 100)
    go_backward = (-100, -100)
    turn_left = (-50, 50)
    turn_right = (50, -50)

    header = ["left wheel speed", "right wheel speed", "gyro x", "gyro y", "gyro z", "pose x", "pose y", "pose z",
        "q0", "q1", "q2", "q3", "angle_z", "origin_id", "is_accurate", "up key", "down key", "left key", "right key", "timestamp"]

    # where jpg filenames should begin. start at 1 so filenames match rows in csv (accounting for header row)
    start_from = 1 
    if append: # if we're appending, need to update starting point
        img_list = [int(filename[:-4]) for filename in os.listdir("img") if filename.endswith(".jpg")]
        if len(img_list) > 0: # it's possible we're appending to a folder without any images
            start_from = max(img_list) + 1
    begin, end = start_from, start_from # we use these to calculate how many new data points are added

    print("Press Control-C to stop")
    with open("data.csv", "a" if append else "x") as file:
        writer = csv.writer(file)
        if not append:
            writer.writerow(header)
        while True: # should this be "while robot.EvtNewCameraImage:"?
            try:
                # drive cozmo
                event = pygame.event.poll()
                if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                    keys = pygame.key.get_pressed()
                    arrows = {"up": keys[pygame.K_UP], "down": keys[pygame.K_DOWN], "left": keys[pygame.K_LEFT], "right": keys[pygame.K_RIGHT]}
                    # calculate wheel speeds
                    wheel_speeds = (0.0, 0.0)
                    # up arrow key
                    if arrows["up"]:
                        wheel_speeds = add_speed(wheel_speeds, go_forward)
                    # down arrow key
                    if arrows["down"]:
                        wheel_speeds = add_speed(wheel_speeds, go_backward)
                    # left arrow key
                    if arrows["left"]:
                        wheel_speeds = add_speed(wheel_speeds, turn_left)
                    # right arrow key
                    if arrows["right"]:
                        wheel_speeds = add_speed(wheel_speeds, turn_right)
                    
                    # send wheel speeds to cozmo
                    robot.drive_wheel_motors(wheel_speeds[0], wheel_speeds[1])
                    print("sending speeds {} and {}".format(wheel_speeds[0], wheel_speeds[1]))

                # save data
                img = robot.world.latest_image
                if img is not None:
                    # save image
                    if verbose:
                        print("Saving image {}".format(end))
                    img.raw_image.save("img/{}.jpg".format(end))
                    # save time/speed/pose data
                    present_time = datetime.now().time()
                    lws = robot.left_wheel_speed.speed_mmps
                    rws = robot.right_wheel_speed.speed_mmps
                    gyro = robot.gyro
                    pose = robot.pose
                    row = [robot.left_wheel_speed.speed_mmps, robot.right_wheel_speed.speed_mmps, gyro.x, gyro.y, gyro.z, datetime.now().time()]
                    row = [robot.pose.position, robot.pose.rotation]
                    row = [lws, rws, gyro.x, gyro.y, gyro.z, pose.position.x, pose.position.y, pose.position.z, pose.rotation.q0, pose.rotation.q1, pose.rotation.q2,
                        pose.rotation.q3, pose.rotation.angle_z.radians, pose.origin_id, pose.is_accurate, arrows["up"], arrows["down"], arrows["left"],
                        arrows["right"], present_time]
                    if verbose:
                        print("Saving data to row {}".format(end))
                    writer.writerow(row)
                    # update count
                    end += 1

                # so we get ~10 images per second
                # could try every 0.25 seconds instead? maybe 0.5?
                time.sleep(0.1) 

        print("Stop command received")
        print("Saved {} new data points".format(end - begin + 1))

        robot.stop_all_motors()
        pygame.quit()


def collect_data(robot: cozmo.robot.Robot):
    global finished
    """Collect LfD data.

    Arguments:
    robot (cozmo.robot.Robot): cozmo robot object
    """
    parser = ArgumentParser(description="Collect LfD data.")
    parser.add_argument("-a", "--append_to", default=None, help="folder to which new data should be appended (default: None)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print extra messages (default: False)")
    parser.add_argument("-c", "--color", action="store_true", help="use color images (default: False)")
    args = parser.parse_args()

    if args.color:
        robot.camera.color_image_enabled = args.color
        time.sleep(0.1) # color takes a bit to kick in
    robot.camera.image_stream_enabled = True 
    append_to = args.append_to
    verbose = args.verbose
    absolute_path = os.path.dirname(os.path.realpath(__file__))

    if append_to:
        folder_name = append_to
        mode = "append"
    else:
        folder_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        os.makedirs("{}/data/{}/img".format(absolute_path, folder_name)) # root folder contains csv, img contains images
        mode = "write"
    
    try:
        os.chdir("{}/data/{}".format(absolute_path, folder_name))
        if verbose:
            print("Saving data in {} mode".format(mode))
        write_data(robot, append=True if append_to else False, verbose=verbose)
        
        print("Data saved to {}".format(os.getcwd()))

    except NotADirectoryError:
        print("Folder does not exist. Was it mistyped?")

    finally:
        print("Exiting")


def main(robot: cozmo.robot.Robot):
    robot.set_head_angle(degrees(0)).wait_for_completed()
    robot.set_lift_height(0).wait_for_completed()
    collect_data(robot)


if __name__ == "__main__":
    # don't need cozmo to drive off the charger
    cozmo.robot.Robot.drive_off_charger_on_connect = False
    cozmo.run_program(main)