#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, JointConstraint
from shape_msgs.msg import SolidPrimitive

class PickAndPlaceClient(Node):
    def __init__(self):
        super().__init__('elfin_pick_and_place_client')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        
        # 基础坐标系配置
        self.BASE_FRAME = "elfin10_l_elfin_base"
        self.EE_LINK = "elfin10_l_elfin_link6"
        # 第三轴关节名称（如果 URDF 中带有前缀，请根据实际调整，例如 "elfin10_l_elfin_joint3"）
        self.JOINT_2_NAME = "elfin_joint2"
        self.JOINT_3_NAME = "elfin_joint3"
        self.JOINT_4_NAME = "elfin_joint4"

    def wait_for_server(self):
        self.get_logger().info('正在等待 MoveGroup Action 服务器连接...')
        return self._action_client.wait_for_server(timeout_sec=10.0)

    # ==================== 1. 机械臂末端 6D 姿态控制 (含重试与关节约束) ====================
    def move_arm_to_pose(self, target_x, target_y, target_z, qx, qy, qz, qw, max_retries=3):
        """
        max_retries: 规划/执行失败时的最大重试次数
        """
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
            #box.dimensions = [0.015, 0.015, 0.015]
            
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
            ori_con.absolute_x_axis_tolerance = 0.005
            ori_con.absolute_y_axis_tolerance = 0.005
            ori_con.absolute_z_axis_tolerance = 0.005
            ori_con.weight = 1.0

            joint2_con = JointConstraint()
            joint2_con.joint_name = self.JOINT_2_NAME
            joint2_con.position = 0.0       # 中心目标为 pi/2 (rad)
            joint2_con.tolerance_below = 1.5708 # 下限: 1.5708 - 1.5708 = 0.0 (禁止负角度)
            joint2_con.tolerance_above = 1.5708 # 上限: 1.5708 + 1.5708 = 3.1416 (pi)
            joint2_con.weight = 1.0

            # --- 第三轴正角度约束 (限制在 [0.0, 3.1416] 弧度之间) ---
            joint3_con = JointConstraint()
            joint3_con.joint_name = self.JOINT_3_NAME
            joint3_con.position = 1.5708       # 中心目标为 pi/2 (rad)
            joint3_con.tolerance_below = 1.5708 # 下限: 1.5708 - 1.5708 = 0.0 (禁止负角度)
            joint3_con.tolerance_above = 1.5708 # 上限: 1.5708 + 1.5708 = 3.1416 (pi)
            joint3_con.weight = 1.0

            joint4_con = JointConstraint()
            joint4_con.joint_name = self.JOINT_4_NAME
            joint4_con.position = 1.5708       # 中心目标为 pi/2 (rad)
            joint4_con.tolerance_below = 1.5708 # 下限: 1.5708 - 1.5708 = 0.0 (禁止负角度)
            joint4_con.tolerance_above = 0.0 # 上限: 1.5708 + 1.5708 = 3.1416 (pi)
            joint4_con.weight = 1.0

            constraint.position_constraints.append(pos_con)
            constraint.orientation_constraints.append(ori_con)
            constraint.joint_constraints.append(joint2_con)
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

        self.get_logger().error(f'❌ 机械臂运动在重试 {max_retries} 次后仍失败！')
        return False

    # ==================== 2. 多轴夹爪控制 (elfin_gripper) ====================
    def control_gripper(self, joint_positions_dict, wait_sec=1.0, max_retries=3):
        """
        joint_positions_dict: {'joint_name': value, ...}
        wait_sec: 夹爪动作完成后等待物理到达的秒数
        max_retries: 规划失败时的最大重试次数 (针对物理碰撞导致的临时失败)
        """
        for attempt in range(1, max_retries + 1):
            goal_msg = MoveGroup.Goal()
            goal_msg.request.group_name = "elfin_gripper"
            goal_msg.request.num_planning_attempts = 20
            goal_msg.request.allowed_planning_time = 5.0
            goal_msg.request.start_state.is_diff = True

            constraint = Constraints()

            # 遍历字典，建立 JointConstraint (容差调宽至 0.01 提高放宽率)
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
        client.get_logger().error('❌ 无法连接到 MoveGroup Action 服务器，退出程序。')
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
    # raw_cube_x = -0.67717286       
    # raw_cube_y = 0.05104384
    # raw_cube_z = 0.54833898

    raw_cube_x = -0.34940088      
    raw_cube_y = 0.05849476
    raw_cube_z = 0.92097903

    # 四元数 (抓短边朝向)
    qx, qy, qz, qw = 0.5, 0.5, 0.5, -0.5

    # 抓取点上方悬空位置
    hover_x = raw_cube_x
    hover_y = raw_cube_y - 0.6
    hover_z = raw_cube_z

    # 下降深度 10cm
    DOWN_OFFSET_Y = 0.10

    # 放置点坐标
    place_hover_x = -0.865196 
    place_hover_y = 0.063129 - 0.6
    place_hover_z = -0.2

    # place_hover_x = raw_cube_x + 0.3
    # place_hover_y = hover_y
    # place_hover_z = raw_cube_z

    client.get_logger().info('🚀 开始执行抓取与放置全流程任务...')
    client.control_gripper(GRIPPER_OPEN)
    
    # ---------------- Step 1: 移动到物块正上方 ----------------
    client.get_logger().info('📍 [Step 1/8] 正在移动到物块正上方...')
    if not client.move_arm_to_pose(hover_x, hover_y, hover_z, qx, qy, qz, qw):
        client.get_logger().error('Step 1 重试耗尽，终止任务')
        return

    # ---------------- Step 2: 往下移动 10cm ----------------
    client.get_logger().info('⬇️ [Step 2/8] 正在往下移动 10cm 到抓取高度...')
    grasp_y = hover_y + DOWN_OFFSET_Y
    if not client.move_arm_to_pose(hover_x, grasp_y, hover_z, qx, qy, qz, qw):
        client.get_logger().error('Step 2 重试耗尽，终止任务')
        return

    # ---------------- Step 3: 控制夹爪关闭 (close) ----------------
    client.get_logger().info('✊ [Step 3/8] 正在关闭 6 轴夹爪 (close)...')
    if not client.control_gripper(GRIPPER_CLOSE):
        client.get_logger().error('Step 3 夹爪关闭失败，终止任务')
        return

    # ---------------- Step 4: 往上移动 10cm ----------------
    client.get_logger().info('⬆️ [Step 4/8] 正在往上抬升 10cm 回到悬空高度...')
    if not client.move_arm_to_pose(hover_x, hover_y, hover_z, qx, qy, qz, qw):
        client.get_logger().error('Step 4 重试耗尽，终止任务')
        return

    # ---------------- Step 5: 水平移动到放置点上方 ----------------
    client.get_logger().info('➡️ [Step 5/8] 正在水平移动到放置点上方...')
    if not client.move_arm_to_pose(place_hover_x, place_hover_y, place_hover_z, qx, qy, qz, qw):
        client.get_logger().error('Step 5 重试耗尽，终止任务')
        return

    # ---------------- Step 6: 往下移动 9cm ----------------
    client.get_logger().info('⬇️ [Step 6/8] 正在往下移动 9cm 到放置高度...')
    place_y = place_hover_y + 0.09
    if not client.move_arm_to_pose(place_hover_x, place_y, place_hover_z, qx, qy, qz, qw):
        client.get_logger().error('Step 6 重试耗尽，终止任务')
        return

    # ---------------- Step 7: 打开夹爪 (open) ----------------
    client.get_logger().info('✋ [Step 7/8] 正在打开 6 轴夹爪 (open)...')
    if not client.control_gripper(GRIPPER_OPEN, wait_sec=0.5):
        client.get_logger().error('Step 7 夹爪打开失败，终止任务')
        return

    # ---------------- Step 8: 往上移动回到准备位置 ----------------
    client.get_logger().info('⬆️ [Step 8/8] 正在往上抬升 9cm 回到准备位置...')
    if not client.move_arm_to_pose(place_hover_x, place_hover_y, place_hover_z, qx, qy, qz, qw):
        client.get_logger().error('Step 8 重试耗尽，终止任务')
        return

    client.get_logger().info('🎉 🎉 🎉 【全流程成功】物块抓取与放置流程已顺利完成！')
    
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()