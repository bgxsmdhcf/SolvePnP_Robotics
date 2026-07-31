# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import isaacsim.core.api.tasks as tasks
import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.storage.native import get_assets_root_path


class PickPlace(tasks.PickPlace):
    def __init__(
        self,
        name: str = "ur10e_pick_place",
        cube_initial_position: Optional[np.ndarray] = None,
        #cube_initial_position: Optional[np.ndarray] = np.array([0.4, 0.4, 0.0]),
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = np.array([0.0515, 0.0515, 0.0515]),
    ) -> None:
        tasks.PickPlace.__init__(
            self,
            name=name,
            cube_initial_position=cube_initial_position,
            cube_initial_orientation=cube_initial_orientation,
            target_position=target_position,
            cube_size=cube_size,
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise Exception("Could not find Isaac Sim assets folder")
        asset_path = (
            assets_root_path + "/Isaac/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd"
            #"/home/nnd/elfin_robot/elfin_description/urdf/elfin10_l/elfin10_l/elfin10_l_modify.usd"
            # "/home/nnd/elfin_robot/elfin_description/elfin10_l_gripper/elfin10_l_gripper.usd"
        )
        add_reference_to_stage(usd_path=asset_path, prim_path="/ur")
        # add_reference_to_stage(usd_path=asset_path, prim_path="/elfin10_l")

        # define the gripper
        gripper = ParallelGripper(
            # We chose the following values while inspecting the articulation
            end_effector_prim_path="/ur/ee_link/robotiq_arg2f_base_link",
            # end_effector_prim_path="/elfin10_l/ee_link_robotiq_arg2f_base_link",
            joint_prim_names=["finger_joint"],
            joint_opened_positions=np.array([0]),
            joint_closed_positions=np.array([40]),
            action_deltas=np.array([-40]),
            use_mimic_joints=True,
        )
        # define the manipulator
        manipulator = SingleManipulator(
            prim_path="/ur",
            name="ur10_robot",
            end_effector_prim_path="/ur/ee_link/robotiq_arg2f_base_link",
            # prim_path="/elfin10_l",
            # name="elfin10_l_robot",
            # end_effector_prim_path="/elfin10_l/ee_link_robotiq_arg2f_base_link",
            gripper=gripper,
        )
        return manipulator
