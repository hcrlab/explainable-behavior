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


def write_data(robot: cozmo.robot.Robot, frequency = 10.0, append=False, verbose=False):
    def add_speed(wheel_speeds: tuple, addition: tuple):
        return (wheel_speeds[0] + addition[0], wheel_speeds[1] + addition[1])
        
    pygame.init()
    screen = pygame.display.set_mode((100,100))
    
    # wheel speeds in millimiters per second
    # (left wheel speed, right wheel speed)
    go_forward = (100, 100)
    go_backward = (-100, -100)
    turn_left = (-50, 50)
    turn_right = (50, -50)
    # head speeds in radians per second
    tilt_up = 0.3
    tilt_down = -0.3

    header = ["left wheel speed", "right wheel speed", "gyro x", "gyro y", "gyro z", "pose x", "pose y", "pose z",
        "q0", "q1", "q2", "q3", "angle_z", "origin_id", "is_accurate", "wheels: up key", "wheels: down key",
        "wheels: left key", "wheels: right key", "head: up key", "head: down key" "timestamp"]

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
        try:
            period = 1/frequency # default: 0.1 sec
            wheels = {"up": 0, "down": 0, "left": 0, "right": 0}
            head = {"up": 0, "down": 0}
            while True: # should this be "while robot.EvtNewCameraImage:"?
                # start timer
                start_time = time.time()
                # drive cozmo
                event = pygame.event.poll()
                if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                    keys = pygame.key.get_pressed()
                    wheels["up"] = keys[pygame.K_UP]
                    wheels["down"] = keys[pygame.K_DOWN]
                    wheels["left"] = keys[pygame.K_LEFT]
                    wheels["right"] = keys[pygame.K_RIGHT]
                    head["up"] = keys[pygame.K_w]
                    head["down"] = keys[pygame.K_r] # colemak
                    # head["down"] = keys[pygame.K_s] # QWERTY
                    # calculate wheel speeds
                    wheel_speeds = (0.0, 0.0)
                    # up arrow key
                    if wheels["up"]:
                        wheel_speeds = add_speed(wheel_speeds, go_forward)
                    # down arrow key
                    if wheels["down"]:
                        wheel_speeds = add_speed(wheel_speeds, go_backward)
                    # left arrow key
                    if wheels["left"]:
                        wheel_speeds = add_speed(wheel_speeds, turn_left)
                    # right arrow key
                    if wheels["right"]:
                        wheel_speeds = add_speed(wheel_speeds, turn_right)

                    # calculate head speed
                    head_speed = 0
                    if head["up"]:
                        head_speed += tilt_up
                    if head["down"]:
                        head_speed += tilt_down
                    
                    # send wheel speeds to cozmo
                    robot.drive_wheel_motors(wheel_speeds[0], wheel_speeds[1])
                    # send head speed
                    robot.move_head(head_speed)

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
                        pose.rotation.q3, pose.rotation.angle_z.radians, pose.origin_id, pose.is_accurate, wheels["up"], wheels["down"], wheels["left"],
                        wheels["right"], head["up"], head["down"], present_time]
                    if verbose:
                        print("Saving data to row {}".format(end))
                    writer.writerow(row)
                    # update count
                    end += 1

                # end timer
                end_time = time.time()
                # calculate how long to sleep to mantain desired frequency (default 10 Hz)
                sample_time = start_time - end_time
                if sample_time < period:
                    time.sleep(period - sample_time)
        except (KeyboardInterrupt, SystemExit):
            print("Stop command received")
            print("Saved {} new data points".format(end - begin + 1))
        finally:
            robot.stop_all_motors()
            pygame.quit()


def collect_data(robot: cozmo.robot.Robot, frequency = 10.0):
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
        write_data(robot, frequency, append=True if append_to else False, verbose=verbose)

        os.chdir("..")
        custom_name = input("What would you like to call this data folder? (leave empty to use {}){}".format(
            "original name" if append_to else "timestamp", os.linesep))
        while True:
            if len(custom_name) > 0:
                try:
                    os.rename(folder_name, custom_name)
                    break
                except OSError:
                    custom_name = input("That folder already exists. Try another name (leave empty to use {}):{}".format(
                        "original name" if append_to else "timestamp", os.linesep))
            else:
                break
        
        print("Data saved to {}".format(custom_name if len(custom_name) > 0 else folder_name))

    except NotADirectoryError:
        print("Folder does not exist. Was it mistyped?")

    finally:
        print("Exiting")


def main(robot: cozmo.robot.Robot):
    robot.set_head_angle(degrees(0)).wait_for_completed()
    robot.set_lift_height(0).wait_for_completed()
    collect_data(robot, 10)


if __name__ == "__main__":
    # don't need cozmo to drive off the charger
    cozmo.robot.Robot.drive_off_charger_on_connect = False
    cozmo.run_program(main)