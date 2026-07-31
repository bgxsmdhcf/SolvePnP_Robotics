#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive

class MoveItPoseClient(Node):
    def __init__(self):
        super().__init__('elfin_moveit_pose_client')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')

    def send_goal(self, target_x, target_y, target_z, qx, qy, qz, qw):
        self.get_logger().info('正在等待 MoveGroup Action 服务器连接...')
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('❌ 无法连接到 MoveGroup！请确保后台已运行 demo.launch.py')
            return

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "elfin_arm"
        goal_msg.request.num_planning_attempts = 20
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.3
        goal_msg.request.max_acceleration_scaling_factor = 0.3

        # 使用当前状态作为起始点
        goal_msg.request.start_state.is_diff = True

        BASE_FRAME = "elfin10_l_elfin_base"
        EE_LINK = "elfin10_l_elfin_link6"

        # ==================== 构建标准 6D Pose 目标约束 ====================
        constraint = Constraints()

        # 1. 位置约束 (使用 2mm 精确立方体包围盒)
        pos_con = PositionConstraint()
        pos_con.header.frame_id = BASE_FRAME
        pos_con.link_name = EE_LINK
        
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.002, 0.002, 0.002]  # 2mm 容差，与 RViz 保持一致
        
        box_pose = Pose()
        box_pose.position.x = target_x
        box_pose.position.y = target_y
        box_pose.position.z = target_z
        box_pose.orientation.w = 1.0  # 包围盒自身的姿态为单位四元数
        
        pos_con.constraint_region.primitives.append(box)
        pos_con.constraint_region.primitive_poses.append(box_pose)
        pos_con.weight = 1.0

        # 2. 姿态约束 (严格对齐目标朝向)
        ori_con = OrientationConstraint()
        ori_con.header.frame_id = BASE_FRAME
        ori_con.link_name = EE_LINK
        ori_con.orientation.x = qx
        ori_con.orientation.y = qy
        ori_con.orientation.z = qz
        ori_con.orientation.w = qw
        
        # 允许微小容差 (约 2.8 度)，给 IK 解算留出姿态微调空间
        ori_con.absolute_x_axis_tolerance = 0.05
        ori_con.absolute_y_axis_tolerance = 0.05
        ori_con.absolute_z_axis_tolerance = 0.05
        ori_con.weight = 1.0

        constraint.position_constraints.append(pos_con)
        constraint.orientation_constraints.append(ori_con)
        goal_msg.request.goal_constraints.append(constraint)

        # 允许 MoveIt 自动规划并直接执行
        goal_msg.planning_options.plan_only = False

        self.get_logger().info(f'🎯 Link6 目标位置: X={target_x:.3f}, Y={target_y:.3f}, Z={target_z:.3f}')
        self.get_logger().info('正在发送目标至 MoveIt 进行 IK 解算与轨迹规划...')
        
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ Goal 被 MoveIt 拒绝！说明该目标点超出了机械臂臂展极限或处于盲区，请在 Isaac Sim 中将物块放近一些。')
            return

        self.get_logger().info('✔ Goal 已接收，正在生成轨迹并驱动 Isaac Sim 执行...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == 1:
            self.get_logger().info('🎉 【运动成功】机械臂姿态正常，已平稳停留在物块上方！')
        else:
            self.get_logger().error(f'❌ 规划/执行失败，MoveIt 错误码: {result.error_code.val}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    client = MoveItPoseClient()

    # ==================== 测试参数设置 ====================
    # 物块 X = 0.865196382m, Y = 0.0631294238m, Z = 0.0m，moveit中的坐标轴y与z跟isaacsim是调换的，moveit中的y对应isaacsim中的z，moveit中的z对应isaacsim中的y
    raw_cube_x = 0.865196       
    raw_cube_y = 0.063129
    raw_cube_z = 0.0001

    GRIPPER_LENGTH = 0.18    # 夹爪长度 (18cm)
    HOVER_HEIGHT = 0.15      # 悬空高度 (15cm)

    target_x = raw_cube_x
    target_y = raw_cube_y - 0.6 #moveit中y轴向下是正方向，所以这里用减是为了让机械臂的末端执行器到达物块上方
    #target_z = raw_cube_z + GRIPPER_LENGTH + HOVER_HEIGHT  # 计算出的 Link6 目标 Z 轴高度
    target_z = raw_cube_z

    # 四元数 [x, y, z, w]
    #qx, qy, qz, qw = 0.707106781, 0.707106781, 0.0, 0.0
    #qx, qy, qz, qw = 0.0, 0.707106781, 0.707106781, 0.0
    qx, qy, qz, qw = 0.5, 0.5, 0.5, -0.5
    # ======================================================

    client.send_goal(target_x, target_y, target_z, qx, qy, qz, qw)
    rclpy.spin(client)

if __name__ == '__main__':
    main()