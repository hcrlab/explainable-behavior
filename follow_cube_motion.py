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

def handle_object_moving_stopped(evt, **kw):
    global cube_moved
    cube_moved = True

def main(robot: cozmo.robot.Robot):
    global cube_moved
    cube_moved = True
    robot.add_event_handler(cozmo.objects.EvtObjectMovingStopped, handle_object_moving_stopped)

    while True:
        if cube_moved:
            print("looking around in place")
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
                # Drive to 70mm away from the cube (much closer and Cozmo
                # will likely hit the cube) and then stop.
                print("going to cube")
                action = robot.go_to_object(cube, distance_mm(70.0))
                action.wait_for_completed()
                print("Completed action: result = %s" % action)
            else:
                break

        cube_moved = False

if __name__ == "__main__":
    cozmo.run_program(main)
