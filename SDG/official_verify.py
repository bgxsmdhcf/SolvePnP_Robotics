import json
import cv2
import numpy as np


def verify_solve_pnp(json_path):
    # 1. 读取 JSON 数据
    with open(json_path, "r") as f:
        data = json.load(f)

    camera_data = data["camera_data"]
    obj_data = data["objects"][0]

    # 2. 构建相机内参矩阵 K
    intrinsics = camera_data["intrinsics"]
    camera_matrix = np.array(
        [
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    # Isaac Sim 默认是理想针孔相机，无畸变
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # 3. 构建物块本地局部坐标系（Object Local Frame）下的 3D 关键点
    # 根据 keypoint_order: ["Center", "LDB", "LDF", "LUB", "LUF", "RDB", "RDF", "RUB", "RUF"]
    size = obj_data["size"]
    x, y, z = size[0] / 2, size[1] / 2, size[2] / 2

    object_points = np.array(
        [
            [0, 0, 0],  # Center
            [-x, -y, -z],  # LDB (Left, Down, Back)
            [-x, -y, z],  # LDF (Left, Down, Front)
            [-x, y, -z],  # LUB (Left, Up, Back)
            [-x, y, z],  # LUF (Left, Up, Front)
            [x, -y, -z],  # RDB (Right, Down, Back)
            [x, -y, z],  # RDF (Right, Down, Front)
            [x, y, -z],  # RUB (Right, Up, Back)
            [x, y, z],  # RUF (Right, Up, Front)
        ],
        dtype=np.float32,
    )

    # 4. 获取图像上的 2D 投影关键点
    image_points = np.array(
        obj_data["cuboid_keypoints_projected"], dtype=np.float32
    )

    # 5. 运行 cv2.solvePnP 解算位姿 (Estimated Pose)
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    R_est, _ = cv2.Rodrigues(rvec)  # 将旋转向量转换为旋转矩阵

    # 6. 从 JSON 中提取并转换出 OpenCV 相机系下的真实位姿 (Ground Truth Pose)
    # Isaac Cam -> OpenCV Cam: X_cv = X_isaac, Y_cv = -Y_isaac, Z_cv = -Z_isaac
    pts_isaac_cam = np.array(obj_data["cuboid_keypoints_camera_frame"])
    pts_cv_cam = np.zeros_like(pts_isaac_cam)
    pts_cv_cam[:, 0] = pts_isaac_cam[:, 0]
    pts_cv_cam[:, 1] = -pts_isaac_cam[:, 1]
    pts_cv_cam[:, 2] = -pts_isaac_cam[:, 2]

    # 真实平移向量：即 OpenCV 相机系下中心点(Center, index 0)的坐标
    t_gt = pts_cv_cam[0].reshape(3, 1)

    # 真实旋转矩阵：通过关键点相对方向计算出物体局部坐标轴在 OpenCV 相机系下的朝向向量
    # X轴方向 (Right - Left) 组合平均
    vX = (
        (pts_cv_cam[5] - pts_cv_cam[1])
        + (pts_cv_cam[6] - pts_cv_cam[2])
        + (pts_cv_cam[7] - pts_cv_cam[3])
        + (pts_cv_cam[8] - pts_cv_cam[4])
    )
    vX /= np.linalg.norm(vX)

    # Y轴方向 (Up - Down) 组合平均
    vY = (
        (pts_cv_cam[3] - pts_cv_cam[1])
        + (pts_cv_cam[4] - pts_cv_cam[2])
        + (pts_cv_cam[7] - pts_cv_cam[5])
        + (pts_cv_cam[8] - pts_cv_cam[6])
    )
    vY /= np.linalg.norm(vY)

    # Z轴方向 (Front - Back) 组合平均
    vZ = (
        (pts_cv_cam[2] - pts_cv_cam[1])
        + (pts_cv_cam[4] - pts_cv_cam[3])
        + (pts_cv_cam[6] - pts_cv_cam[5])
        + (pts_cv_cam[8] - pts_cv_cam[7])
    )
    vZ /= np.linalg.norm(vZ)

    R_gt = np.column_stack((vX, vY, vZ))

    # 7. 打印对比结果与误差
    print("=================== 位姿验证结果 ===================")
    print(f"解算是否成功: {success}\n")

    print("【平移向量 t (Translation) 对比】单位: 米")
    print(f"solvePnP 估计值:\n{tvec.flatten()}")
    print(f"JSON 几何真实值:\n{t_gt.flatten()}")
    t_error = np.linalg.norm(tvec - t_gt)
    print(f"--> 平移绝对误差: {t_error:.6f} 米\n")

    print("【旋转矩阵 R (Rotation Matrix) 对比】")
    print(f"solvePnP 估计值:\n{R_est}")
    print(f"JSON 几何真实值:\n{R_gt}")

    # 计算旋转角度误差 (轴角距离)
    R_diff = np.dot(R_est, R_gt.T)
    angle_error = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
    print(f"--> 旋转角度误差: {np.degrees(angle_error):.4f} 度")
    print("====================================================")


# 运行验证（确保 000000.json 在当前目录下或指定正确路径）
verify_solve_pnp("./_out_pose_writer/000000.json")