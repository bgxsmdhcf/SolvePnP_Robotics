#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

class ViewportVerifierNode(Node):
    def __init__(self):
        super().__init__('viewport_verifier_node')
        self.bridge = CvBridge()
        
        # 缓存最新收到的 2D 关键点坐标
        self.current_keypoints = None
        
        # 1. 订阅计算出来的 2D 关键点话题
        self.points_sub = self.create_subscription(
            Float64MultiArray,
            '/cube_2d_keypoints',
            self.points_callback,
            10
        )
        
        # 2. 订阅仿真器相机发出来的原生 RGB 图像话题
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.get_logger().info('==================================================')
        self.get_logger().info('🎉 Sim2Real 2D投影像素级验证节点启动成功！')
        self.get_logger().info('   正在等待仿真数据流，请确保 Isaac Sim 处于 Play 状态...')
        self.get_logger().info('==================================================')

    def points_callback(self, msg):
        # 收到坐标数据（应该是包含 8 个 float 的一维数组）
        if len(msg.data) == 8:
            self.current_keypoints = np.array(msg.data).reshape(-1, 2)

    def image_callback(self, msg):
        try:
            # 将 ROS 2 图像消息转换为 OpenCV 的 BGR 格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像解码失败: {str(e)}')
            return

        # 创建一个画布副本用于绘制叠加信息
        canvas = cv_image.copy()
        
        # 检查当前帧是否有可用的坐标数据
        if self.current_keypoints is not None:
            # 遍历这 4 个投影出来的点，并把它们画在图像上
            for i, pt in enumerate(self.current_keypoints):
                u, v = int(pt[0]), int(pt[1])
                
                # 在计算出的像素位置画一个显眼的红色实心圆（半径 6 像素）
                cv2.circle(canvas, (u, v), 6, (0, 0, 255), -1)
                
                # 给每个角点标上编号，方便排查顺序是否错位
                cv2.putText(canvas, f"P{i+1}", (u + 10, v - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 在画面左上角提示当前状态
            cv2.putText(canvas, "Status: LOCK 4/4 KEYPOINTS", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(canvas, "Status: WAITING FOR DATA...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # 实时弹窗展现对齐效果
        cv2.imshow("Sim2Real Pixel-Level Verification Panel", canvas)
        
        # 刷新 OpenCV 窗口缓冲区（必须保留，否则画面会卡死白屏）
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ViewportVerifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()