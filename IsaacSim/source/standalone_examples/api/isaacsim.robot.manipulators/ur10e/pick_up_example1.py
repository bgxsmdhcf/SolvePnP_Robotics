from isaacsim import SimulationApp

# 1. 首先启动 SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from scipy.spatial.transform import Rotation as R  # 导入 scipy 用于处理旋转
from controller.pick_place import PickPlaceController
from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.core.prims import RigidPrim, XFormPrim 

# 2. 打开你现有的完整 USD 场景文件
usd_scene_path = "/home/nnd/ur_usd/official_configured_usd/Collected_ur_gripper/ur_gripper_scene.usd" 
open_stage(usd_path=usd_scene_path)

# 3. 初始化 World
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=20 / 200)

# 4. 绑定场景中已有的机械臂与夹爪
gripper = ParallelGripper(
    end_effector_prim_path="/World/ur_gripper/ee_link/robotiq_arg2f_base_link",
    joint_prim_names=["finger_joint"],
    joint_opened_positions=np.array([0]),
    joint_closed_positions=np.array([40]),
    action_deltas=np.array([-40]),
    use_mimic_joints=True,
)

my_robot = SingleManipulator(
    prim_path="/World/ur_gripper",
    name="ur10_robot",
    end_effector_prim_path="/World/ur_gripper/ee_link/robotiq_arg2f_base_link",
    gripper=gripper,
)
my_world.scene.add(my_robot)

# 5. 绑定场景中已有的待抓取物块与相机
cube_prim_path = "/World/Cube" 
my_cube = RigidPrim(prim_paths_expr=cube_prim_path, name="target_cube")
my_world.scene.add(my_cube)

# 实例化相机 Prim 以便获取它在世界坐标系下的实时位姿
camera_prim = XFormPrim(prim_paths_expr="/World/Camera") 

# 6. 设置放置的目标位置
target_position = np.array([-0.3, 0.6, 0.09]) 

# 7. 初始化控制器
my_controller = PickPlaceController(name="controller", robot_articulation=my_robot, gripper=my_robot.gripper)
articulation_controller = my_robot.get_articulation_controller()

# 重置世界与控制器
my_world.reset()
my_controller.reset()

# ==================== 【核心：基于 PnP 结果计算位姿】 ====================
# 填入你运行 official_verify.py 得到的 OpenCV 坐标系下的数据
t_cv = np.array([-0.15435726,  0.25767091,  1.81198095])
R_cv = np.array([
    [ 0.8538611,   0.52049565, -0.00234552],
    [ 0.46908119, -0.77145416, -0.4299085 ],
    [-0.22557497,  0.3659819,  -0.90286941]
])

# Step 1: OpenCV 相机系 转换为 Isaac Sim 相机系
t_cam_isaac = np.array([t_cv[0], -t_cv[1], -t_cv[2]])
R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
R_cam_isaac = R_flip @ R_cv

# Step 2: 获取相机在世界坐标系下的位姿，转换为世界坐标
cam_world_pos, cam_world_quat = camera_prim.get_world_poses()  # quat 格式为 [w, x, y, z]
R_cam_to_world = R.from_quat([cam_world_quat[0][1], cam_world_quat[0][2], cam_world_quat[0][3], cam_world_quat[0][0]]).as_matrix()

cube_world_pos = R_cam_to_world @ t_cam_isaac + cam_world_pos[0]
R_cube_world = R_cam_to_world @ R_cam_isaac

# Step 3: 获取机械臂 Base 在世界坐标系下的位姿，计算相对 Base 的位姿
robot_world_pos, robot_world_quat = my_robot.get_world_pose()
R_robot_to_world = R.from_quat([robot_world_quat[1], robot_world_quat[2], robot_world_quat[3], robot_world_quat[0]]).as_matrix()

# 最终计算出：物块相对于机械臂 Base 的位姿 (用户所需信息)
cube_base_pos = R_robot_to_world.T @ (cube_world_pos - robot_world_pos)
R_cube_base = R_robot_to_world.T @ R_cube_world

print("\n" + "="*20 + " 相对 Base 位姿计算结果 " + "="*20)
print(f"物块相对机械臂 Base 的平移向量 (t_base):\n{cube_base_pos}")
print(f"物块相对机械臂 Base 的旋转矩阵 (R_base):\n{R_cube_base}")

# Step 4: 自动计算自适应抓取朝向（解决短边下手问题）
# 提取物块在世界系下的 Euler 角 (XYZ 顺序)
cube_euler = R.from_matrix(R_cube_world).as_euler('xyz', degrees=True)
cube_yaw = cube_euler[2]

# 机械臂向下抓取基准为 Roll=180, Pitch=0。我们将 Yaw 与物块的 Yaw 同步。
# 如果发现依旧抓长边，请将下面的 `cube_yaw + 90` 改为 `cube_yaw` 或 `cube_yaw - 90`
target_yaw = cube_yaw + 90 
target_gripper_euler = [180, 0, target_yaw]

q_xyzw = R.from_euler('xyz', target_gripper_euler, degrees=True).as_quat()
custom_orientation = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]) # 转为 [w, x, y, z]
print(f"自适应短边抓取朝向四元数: {custom_orientation}")
print("="*60 + "\n")
# ========================================================================

reset_needed = False
task_completed = False

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

        current_joints = my_robot.get_joint_positions()

        # 解决上个回合提到的放物体挤压问题：手动为放置目标点加上 0.19m 的高度预量
        #adjusted_placing_position = target_position + np.array([0, 0, 0.09])

        # 将视觉解算得到的物理世界坐标 cube_world_pos 喂给控制器
        actions = my_controller.forward(
            picking_position=cube_world_pos,            # 核心：使用 PnP 计算出的世界坐标
            placing_position=target_position,  # 核心：防止释放挤压的修正坐标
            current_joint_positions=current_joints,
            end_effector_offset=np.array([0, 0, 0.18]),  # 保持 0.18 悬空闭合预量
            end_effector_orientation=custom_orientation, # 核心：传入自适应短边的朝向
        )
        
        if my_controller.is_done() and not task_completed:
            print("done picking and placing")
            task_completed = True
            
        articulation_controller.apply_action(actions)

    if my_world.is_stopped():
        reset_needed = True

simulation_app.close()