#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, JointConstraint
from shape_msgs.msg import SolidPrimitive

class PickAndPlaceClient(Node):
    def __init__(self):
        super().__init__('elfin_pick_and_place_client')
        # Action 和 Service 客户端
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self._cartesian_client = self.create_client(GetCartesianPath, 'compute_cartesian_path')
        self._execute_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory')
        
        # 基础坐标系配置
        self.BASE_FRAME = "elfin10_l_elfin_base"
        self.EE_LINK = "elfin10_l_elfin_link6"
        # 第三轴与第四轴关节名称
        self.JOINT_3_NAME = "elfin_joint3"
        self.JOINT_4_NAME = "elfin_joint4"

    def wait_for_server(self):
        self.get_logger().info('正在等待 MoveGroup Action 服务器连接...')
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            return False
            
        self.get_logger().info('正在等待 compute_cartesian_path 服务连接...')
        if not self._cartesian_client.wait_for_service(timeout_sec=10.0):
            return False
            
        self.get_logger().info('正在等待 execute_trajectory Action 服务器连接...')
        if not self._execute_client.wait_for_server(timeout_sec=10.0):
            return False
            
        return True

    # ==================== 1. 自由空间运动 (OMPL 自由规划) ====================
    def move_arm_to_pose(self, target_x, target_y, target_z, qx, qy, qz, qw, max_retries=3):
        """用于远距离大范围移动（如 Step 1、Step 5）"""
        for attempt in range(1, max_retries + 1):
            goal_msg = MoveGroup.Goal()
            goal_msg.request.group_name = "elfin_arm"
            goal_msg.request.num_planning_attempts = 200
            goal_msg.request.allowed_planning_time = 5.0
            goal_msg.request.max_velocity_scaling_factor = 0.5
            goal_msg.request.max_acceleration_scaling_factor = 0.5
            goal_msg.request.start_state.is_diff = True

            constraint = Constraints()

            # --- 位置约束 ---
            pos_con = PositionConstraint()
            pos_con.header.frame_id = self.BASE_FRAME
            pos_con.link_name = self.EE_LINK
            
            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [0.002, 0.002, 0.002]
            
            box_pose = Pose()
            box_pose.position.x = target_x
            box_pose.position.y = target_y
            box_pose.position.z = target_z
            box_pose.orientation.w = 1.0
            
            pos_con.constraint_region.primitives.append(box)
            pos_con.constraint_region.primitive_poses.append(box_pose)
            pos_con.weight = 1.0

            # --- 姿态约束 ---
            ori_con = OrientationConstraint()
            ori_con.header.frame_id = self.BASE_FRAME
            ori_con.link_name = self.EE_LINK
            ori_con.orientation.x = qx
            ori_con.orientation.y = qy
            ori_con.orientation.z = qz
            ori_con.orientation.w = qw
            ori_con.absolute_x_axis_tolerance = 0.05
            ori_con.absolute_y_axis_tolerance = 0.05
            ori_con.absolute_z_axis_tolerance = 0.05
            ori_con.weight = 1.0

            # --- 关节角度正值约束 ---
            joint3_con = JointConstraint()
            joint3_con.joint_name = self.JOINT_3_NAME
            joint3_con.position = 1.5708
            joint3_con.tolerance_below = 1.5708
            joint3_con.tolerance_above = 1.5708
            joint3_con.weight = 1.0

            joint4_con = JointConstraint()
            joint4_con.joint_name = self.JOINT_4_NAME
            joint4_con.position = 1.5708
            joint4_con.tolerance_below = 1.5708
            joint4_con.tolerance_above = 0.0
            joint4_con.weight = 1.0

            constraint.position_constraints.append(pos_con)
            constraint.orientation_constraints.append(ori_con)
            constraint.joint_constraints.append(joint3_con)
            constraint.joint_constraints.append(joint4_con)

            goal_msg.request.goal_constraints.append(constraint)
            goal_msg.planning_options.plan_only = False

            send_future = self._action_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, send_future)
            goal_handle = send_future.result()

            if not goal_handle.accepted:
                self.get_logger().warn(f'⚠️ [尝试 {attempt}/{max_retries}] 机械臂 Goal 被 MoveIt 拒绝，1s 后重试...')
                time.sleep(1)
                continue

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result().result
            
            if result.error_code.val == 1:
                return True
            else:
                self.get_logger().warn(
                    f'⚠️ [尝试 {attempt}/{max_retries}] 机械臂规划/执行失败，MoveIt 错误码: {result.error_code.val}，1s 后重试...'
                )
                time.sleep(1)

        self.get_logger().error(f'❌ 机械臂自由空间运动在重试 {max_retries} 次后仍失败！')
        return False

    # ==================== 2. 笛卡尔直线插补运动 (用于上升/下降) ====================
    def move_arm_cartesian_to_pose(self, target_x, target_y, target_z, qx, qy, qz, qw, max_retries=3):
        """
        利用笛卡尔直线插补 (GetCartesianPath) 从当前姿态沿直线规划移动到目标姿态
        可完全防止上升/下降过程中发生关节跳变和肘部下凹
        """
        target_pose = Pose()
        target_pose.position.x = float(target_x)
        target_pose.position.y = float(target_y)
        target_pose.position.z = float(target_z)
        target_pose.orientation.x = float(qx)
        target_pose.orientation.y = float(qy)
        target_pose.orientation.z = float(qz)
        target_pose.orientation.w = float(qw)

        waypoints = [target_pose]

        for attempt in range(1, max_retries + 1):
            req = GetCartesianPath.Request()
            req.header.frame_id = self.BASE_FRAME
            req.start_state.is_diff = True
            req.group_name = "elfin_arm"
            req.link_name = self.EE_LINK
            req.waypoints = waypoints
            req.max_step = 0.001          # 1cm 插补步长
            req.jump_threshold = 0.0     # 0.0 表示禁用跳变阈值检测
            req.avoid_collisions = False

            future = self._cartesian_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            res = future.result()

            if res is None:
                self.get_logger().warn(f'⚠️ [尝试 {attempt}/{max_retries}] 笛卡尔路径计算服务无响应，1s 后重试...')
                time.sleep(1.0)
                continue

            # fraction 记录直线计算完成度，1.0 表示 100% 成功生成直线轨迹
            if res.fraction < 0.95:
                self.get_logger().warn(
                    f'⚠️ [尝试 {attempt}/{max_retries}] 笛卡尔直线路径无法完整插补 (完成度: {res.fraction * 100:.1f}%)，可能遇到自碰撞，1s 后重试...'
                )
                time.sleep(1.0)
                continue

            # 发送给 execute_trajectory Action 服务器执行
            exec_goal = ExecuteTrajectory.Goal()
            exec_goal.trajectory = res.trajectory

            send_future = self._execute_client.send_goal_async(exec_goal)
            rclpy.spin_until_future_complete(self, send_future)
            goal_handle = send_future.result()

            if not goal_handle.accepted:
                self.get_logger().warn(f'⚠️ [尝试 {attempt}/{max_retries}] 笛卡尔轨迹 Goal 被拒绝，1s 后重试...')
                time.sleep(1.0)
                continue

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            exec_result = result_future.result().result

            if exec_result.error_code.val == 1:
                return True
            else:
                self.get_logger().warn(
                    f'⚠️ [尝试 {attempt}/{max_retries}] 笛卡尔轨迹执行失败，MoveIt 错误码: {exec_result.error_code.val}，1s 后重试...'
                )
                time.sleep(1.0)

        self.get_logger().error(f'❌ 笛卡尔直线运动在重试 {max_retries} 次后仍失败！')
        return False

    # ==================== 3. 多轴夹爪控制 (elfin_gripper) ====================
    def control_gripper(self, joint_positions_dict, wait_sec=1.0, max_retries=3):
        for attempt in range(1, max_retries + 1):
            goal_msg = MoveGroup.Goal()
            goal_msg.request.group_name = "elfin_gripper"
            goal_msg.request.num_planning_attempts = 20
            goal_msg.request.allowed_planning_time = 5.0
            goal_msg.request.start_state.is_diff = True

            constraint = Constraints()

            for joint_name, target_pos in joint_positions_dict.items():
                joint_con = JointConstraint()
                joint_con.joint_name = joint_name
                joint_con.position = float(target_pos)
                joint_con.tolerance_above = 0.01
                joint_con.tolerance_below = 0.01
                joint_con.weight = 1.0
                constraint.joint_constraints.append(joint_con)

            goal_msg.request.goal_constraints.append(constraint)
            goal_msg.planning_options.plan_only = False

            send_future = self._action_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, send_future)
            goal_handle = send_future.result()

            if not goal_handle.accepted:
                self.get_logger().warn(f'⚠️ [尝试 {attempt}/{max_retries}] 夹爪 Goal 被 MoveIt 拒绝，0.5s 后重试...')
                time.sleep(0.5)
                continue

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result().result
            
            if result.error_code.val == 1:
                self.get_logger().info('✔ 夹爪指令已顺利成功执行！')
                if wait_sec > 0:
                    self.get_logger().info(f'⏳ 等待 {wait_sec}s 以确保物理状态完全到位...')
                    time.sleep(wait_sec)
                return True
            else:
                self.get_logger().warn(
                    f'⚠️ [尝试 {attempt}/{max_retries}] 夹爪规划失败，MoveIt 错误码: {result.error_code.val} (可能触发了碰撞保护)，0.5s 后重试...'
                )
                time.sleep(0.5)

        self.get_logger().error(f'❌ 夹爪控制在重试 {max_retries} 次后仍失败！')
        return False


def main(args=None):
    rclpy.init(args=args)
    client = PickAndPlaceClient()

    if not client.wait_for_server():
        client.get_logger().error('❌ 无法连接到 Action / Service 服务器，退出程序。')
        return

    # ==================== 1. SRDF 中定义的夹爪 6 轴状态 ====================
    GRIPPER_OPEN = {
        "finger_joint": 0.0,
        "left_inner_finger_joint": 0.0,
        "left_inner_knuckle_joint": 0.0,
        "right_inner_finger_joint": 0.0,
        "right_inner_knuckle_joint": 0.0,
        "right_outer_knuckle_joint": 0.0
    }

    GRIPPER_CLOSE = {
        "finger_joint": 0.5,
        "left_inner_finger_joint": 0.5,
        "left_inner_knuckle_joint": -0.5,
        "right_inner_finger_joint": 0.5,
        "right_inner_knuckle_joint": -0.5,
        "right_outer_knuckle_joint": -0.5
    }

    # ==================== 2. 坐标与参数定义 ====================
    raw_cube_x = -0.67717286       
    raw_cube_y = 0.05104384
    raw_cube_z = 0.54833898

    # 四元数
    qx, qy, qz, qw = 0.5, 0.5, 0.5, -0.5

    # 抓取点上方悬空位置
    hover_x = raw_cube_x
    hover_y = raw_cube_y - 0.6
    hover_z = raw_cube_z

    # 下降深度 10cm
    DOWN_OFFSET_Y = 0.10

    # 放置点坐标
    place_hover_x = 0.865196 
    place_hover_y = 0.063129 - 0.6
    place_hover_z = 0.2

    client.get_logger().info('🚀 开始执行抓取与放置全流程任务...')
    client.control_gripper(GRIPPER_OPEN)
    
    # ---------------- Step 1: 移动到物块正上方 (自由规划) ----------------
    client.get_logger().info('📍 [Step 1/8] 正在移动到物块正上方...')
    if not client.move_arm_to_pose(hover_x, hover_y, hover_z, qx, qy, qz, qw):
        client.get_logger().error('Step 1 重试耗尽，终止任务')
        return

    # ---------------- Step 2: 往下移动 10cm (直线插补) ----------------
    client.get_logger().info('⬇️ [Step 2/8] 正在【直线向下】移动 10cm 到抓取高度...')
    grasp_y = hover_y + DOWN_OFFSET_Y
    if not client.move_arm_cartesian_to_pose(hover_x, grasp_y, hover_z, qx, qy, qz, qw):
        client.get_logger().error('Step 2 重试耗尽，终止任务')
        return

    # ---------------- Step 3: 控制夹爪关闭 ----------------
    client.get_logger().info('✊ [Step 3/8] 正在关闭 6 轴夹爪 (close)...')
    if not client.control_gripper(GRIPPER_CLOSE):
        client.get_logger().error('Step 3 夹爪关闭失败，终止任务')
        return

    # ---------------- Step 4: 往上移动 10cm (直线插补) ----------------
    client.get_logger().info('⬆️ [Step 4/8] 正在【直线向上】抬升 10cm 回到悬空高度...')
    if not client.move_arm_cartesian_to_pose(hover_x, hover_y, hover_z, qx, qy, qz, qw):
        client.get_logger().error('Step 4 重试耗尽，终止任务')
        return

    # ---------------- Step 5: 水平移动到放置点上方 (自由规划) ----------------
    client.get_logger().info('➡️ [Step 5/8] 正在水平移动到放置点上方...')
    if not client.move_arm_to_pose(place_hover_x, place_hover_y, place_hover_z, qx, qy, qz, qw):
        client.get_logger().error('Step 5 重试耗尽，终止任务')
        return

    # ---------------- Step 6: 往下移动 9cm (直线插补) ----------------
    client.get_logger().info('⬇️ [Step 6/8] 正在【直线向下】移动 9cm 到放置高度...')
    place_y = place_hover_y + 0.09
    if not client.move_arm_cartesian_to_pose(place_hover_x, place_y, place_hover_z, qx, qy, qz, qw):
        client.get_logger().error('Step 6 重试耗尽，终止任务')
        return

    # ---------------- Step 7: 打开夹爪 ----------------
    client.get_logger().info('✋ [Step 7/8] 正在打开 6 轴夹爪 (open)...')
    if not client.control_gripper(GRIPPER_OPEN, wait_sec=0.5):
        client.get_logger().error('Step 7 夹爪打开失败，终止任务')
        return

    # ---------------- Step 8: 往上移动回到准备位置 (直线插补) ----------------
    client.get_logger().info('⬆️ [Step 8/8] 正在【直线向上】抬升 9cm 回到准备位置...')
    if not client.move_arm_cartesian_to_pose(place_hover_x, place_hover_y, place_hover_z, qx, qy, qz, qw):
        client.get_logger().error('Step 8 重试耗尽，终止任务')
        return

    client.get_logger().info('🎉 🎉 🎉 【全流程成功】物块抓取与放置流程已顺利完成！')
    
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()