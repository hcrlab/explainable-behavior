#!/usr/bin/env python3

# Copyright (c) 2016 Anki, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Tell Cozmo to find a cube, and then drive up to it

This is a test / example usage of the robot.go_to_object call which creates a
GoToObject action, that can be used to drive within a given distance of an
object (e.g. a LightCube).
'''

import asyncio
import time

import cozmo
from cozmo.util import degrees, distance_mm

def calc_cube_dist(robot, cube):
    return ((cube.pose.position.x - robot.pose.position.x)**2 + (cube.pose.position.y - robot.pose.position.y)**2 + (cube.pose.position.z - robot.pose.position.z)**2)**0.5

def go_to_cube(robot, cube):
    # Drive to 70mm away from the cube (much closer and Cozmo
    # will likely hit the cube) and then stop.
    print("going to cube")
    action = robot.go_to_object(cube, distance_mm(70.0))
    action.wait_for_completed()
    print("Completed action: result = %s" % action)


def main(robot: cozmo.robot.Robot, dist_threshold = 100):
    while True:
        # check if there's a cube currently in front of cozmo
        try:
            cube = robot.world.wait_for_observed_light_cube(timeout=2)
            cube_dist = calc_cube_dist(robot, cube)
            # drive to cube if it's far enough away, otherwise just stay put
            if cube_dist > dist_threshold:
                go_to_cube(robot, cube)
        # if no cube in front of cozmo, look around for one
        except asyncio.TimeoutError:
            print("No cube in view. Searching...")
            look_around = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
            try:
                print("waiting for cube...")
                cube = robot.world.wait_for_observed_light_cube(timeout=30)
                print("Found cube")
            except asyncio.TimeoutError:
                print("Didn't find a cube. Exiting.")
                cube = None # need to assign some value to cube for upcoming "if cube"
            finally:
                # whether we find it or not, we want to stop the behavior
                look_around.stop()
                time.sleep(1) # removing this seems to cause cozmo to try to both go to the cube
                # and look around simultaneously... not good

            if cube:
                go_to_cube(robot, cube)
            else:
                break

if __name__ == "__main__":
    cozmo.run_program(main)
