# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
'''
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})
import numpy as np
from controller.pick_place import PickPlaceController
from isaacsim.core.api import World
from tasks.pick_place import PickPlace
#import omni

my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=20 / 200)

target_position = np.array([-0.3, 0.6, 0])
target_position[2] = 0.0515 / 2.0
#my_task = PickPlace(name="ur10e_pick_place", target_position=target_position, cube_size=np.array([0.1, 0.0515, 0.1]))
my_task = PickPlace(name="ur10e_pick_place", cube_initial_position=np.array([0.6, 0.6, 0.5]), target_position=target_position, cube_size=np.array([0.1, 0.0515, 0.1]))
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("ur10e_pick_place").get_params()
ur10e_name = task_params["robot_name"]["value"]
my_ur10e = my_world.scene.get_object(ur10e_name)
# initialize the controller

my_controller = PickPlaceController(name="controller", robot_articulation=my_ur10e, gripper=my_ur10e.gripper)
task_params = my_world.get_task("ur10e_pick_place").get_params()
articulation_controller = my_ur10e.get_articulation_controller()

reset_needed = False
task_completed = False

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False
            my_controller.reset()
            task_completed = False
        if my_world.current_time_step_index == 0:
            my_controller.reset()

        observations = my_world.get_observations()
        #vertical_orientation = np.array([0.0, 0.0, 1.0, 0.0])
        #cube_name = task_params["cube_name"]["value"]
        #cube_orientation = observations[cube_name]["orientation"]
        # forward the observation values to the controller to get the actions
        actions = my_controller.forward(
            picking_position=observations[task_params["cube_name"]["value"]]["position"],
            placing_position=observations[task_params["cube_name"]["value"]]["target_position"],
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            # This offset needs tuning as well
            end_effector_offset=np.array([0, 0, 0.19]),
            #end_effector_offset=np.array([0, 0, 0.18]),
            #end_effector_offset=np.array([0, 0, 0.30]),
            #end_effector_orientation=cube_orientation,
            #end_effector_orientation=target_downward_orientation,
            #end_effector_orientation=np.array([0.0, 0.0, 1.0, 0.0])
        )
        if my_controller.is_done() and not task_completed:
            print("done picking and placing")
            task_completed = True
        articulation_controller.apply_action(actions)

    if my_world.is_stopped():
        reset_needed = True


simulation_app.close()
'''


from isaacsim import SimulationApp

# 1. 首先启动 SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from controller.pick_place import PickPlaceController
from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.core.prims import RigidPrim  # 如果物块有刚体物理属性，用 RigidPrim；否则用 XFormPrim

# 2. 打开你现有的完整 USD 场景文件
# 请将下方的路径替换为你实际的 USD 文件绝对路径
usd_scene_path = "/home/nnd/ur_usd/official_configured_usd/Collected_ur_gripper/ur_gripper_scene.usd" 
open_stage(usd_path=usd_scene_path)

# 3. 初始化 World（此时 Stage 中已经加载了你的所有模型）
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=20 / 200)

# 4. 绑定场景中【已经存在】的机械臂与夹爪
# 注意：请务必去 Stage 树中核对你的机械臂和夹爪的实际 Prim 路径以及关节名称
gripper = ParallelGripper(
    end_effector_prim_path="/World/ur_gripper/ee_link/robotiq_arg2f_base_link",  # 替换为实际末端路径
    joint_prim_names=["finger_joint"],                                   # 替换为实际夹爪关节名
    joint_opened_positions=np.array([0]),
    joint_closed_positions=np.array([40]),
    action_deltas=np.array([-40]),
    use_mimic_joints=True,
)

my_robot = SingleManipulator(
    prim_path="/World/ur_gripper",                                              # 替换为实际机器人根路径
    name="ur10_robot",
    end_effector_prim_path="/World/ur_gripper/ee_link/robotiq_arg2f_base_link",
    gripper=gripper,
)
my_world.scene.add(my_robot)

# 5. 绑定场景中【已经存在】的待抓取物块
# 请在 Stage 树中查看该物块的绝对路径（例如 /World/Cube 或 /World/Visual/Cube）
cube_prim_path = "/World/Cube" 
my_cube = RigidPrim(prim_paths_expr=cube_prim_path, name="target_cube")
my_world.scene.add(my_cube)

# 6. 设置你想让它放置的目标位置（Target Position）
# 因为没有了 Task 自动创建视觉目标，这里可以直接指定一个坐标
target_position = np.array([-0.3, 0.6, 0.09]) 

# 7. 初始化控制器
my_controller = PickPlaceController(name="controller", robot_articulation=my_robot, gripper=my_robot.gripper)
articulation_controller = my_robot.get_articulation_controller()

# 充置世界与控制器
my_world.reset()
my_controller.reset()

reset_needed = False
task_completed = False

# from scipy.spatial.transform import Rotation as R  # 导入 scipy 用于处理旋转

# # ==================== 1. 在主循环外部定义抓取朝向 ====================
# # 机械臂垂直向下抓取时，通常 Roll=180, Pitch=0。
# # 我们通过调整 Yaw（围绕 Z 轴旋转）来控制它是抓长边还是短边。
# # 如果运行后发现方向反了，可以把 Yaw 从 90 改为 0 或 -90 进行微调。
# target_euler = [180, 0, 90]  # 依次为 [Roll, Pitch, Yaw] (单位：度)

# q_xyzw = R.from_euler('xyz', target_euler, degrees=True).as_quat()
# # 将 Scipy 的 [x, y, z, w] 转换为 Isaac Sim 要求的 [w, x, y, z]
# custom_orientation = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

# 主循环
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False
            my_controller.reset()
            task_completed = False
        if my_world.current_time_step_index == 0:
            my_controller.reset()

        # 8. 直接从绑定的物体中实时获取位置和关节状态，不再读取 Task Observations
        positions, _ = my_cube.get_world_poses()
        cube_position = positions[0]
        current_joints = my_robot.get_joint_positions()

        # 将状态传入控制器计算 Action
        actions = my_controller.forward(
            picking_position=cube_position,
            placing_position=target_position,
            current_joint_positions=current_joints,
            end_effector_offset=np.array([0, 0, 0.18]),  # 根据你的实际夹爪长度微调该偏置
            #end_effector_orientation=custom_orientation,
        )
        
        if my_controller.is_done() and not task_completed:
            print("done picking and placing")
            task_completed = True
            
        articulation_controller.apply_action(actions)

    if my_world.is_stopped():
        reset_needed = True

simulation_app.close()
