import os
import sys
import math

# ====================================================================
# 1. 启动 Isaac Sim Standalone 仿真环境
# ====================================================================
from isaacsim import SimulationApp

# 设置 headless=False 确保本地渲染硬件上下文完全建立
simulation_app = SimulationApp({"headless": False}) 

import numpy as np
import omni.usd
import omni.graph.core as og
import omni.replicator.core as rep
from pxr import Sdf, Usd, Gf, UsdGeom  # 标准 Pixar 原生几何库

from isaacsim.core.utils import extensions
from isaacsim.core.api import SimulationContext

# 启动 ROS 2 桥接扩展
extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# 加载包含预建相机 Action Graph 的场景
omni.usd.get_context().open_stage("/home/nnd/elfin_robot/elfin_description/urdf/elfin10_l/elfin10_l/elfin10_l.usd")
simulation_context = SimulationContext(stage_units_in_meters=1.0)

# ====================================================================
# 2. 获取场景关键图元节点
# ====================================================================
CUBE_PATH = "/World/cube/Cube"
CAMERA_PRIM_PATH = "/Camera"  

stage = omni.usd.get_context().get_stage()
cube_prim = stage.GetPrimAtPath(CUBE_PATH)
camera_prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)

full_x = 0.07535
full_y = 0.15454
full_z = 0.20482
half_x = full_x / 2.0
half_y = full_y / 2.0

LOCAL_CORNERS = np.array([
    [ half_x,  half_y, full_z, 1.0],
    [-half_x,  half_y, full_z, 1.0],
    [-half_x, -half_y, full_z, 1.0],
    [ half_x, -half_y, full_z, 1.0]
])

# ====================================================================
# 3. 动态构建坐标发布者（使用脚本独占路径）
# ====================================================================
DYNAMIC_GRAPH_PATH = "/World/ScriptKeypointsGraph_v2"

print("[Sim2Real] 正在安全追加特征点发布 Action Graph...")
og.Controller.edit(
    {"graph_path": DYNAMIC_GRAPH_PATH, "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("ROS2Publisher", "isaacsim.ros2.bridge.ROS2Publisher"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "ROS2Publisher.inputs:execIn"),
            ("ROS2Context.outputs:context", "ROS2Publisher.inputs:context"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("ROS2Publisher.inputs:messagePackage", "std_msgs"),
            ("ROS2Publisher.inputs:messageSubfolder", "msg"),
            ("ROS2Publisher.inputs:messageName", "Float64MultiArray"),
            ("ROS2Publisher.inputs:topicName", "/cube_2d_keypoints"),
        ],
    },
)

for _ in range(10):
    simulation_app.update()

data_attr = og.Controller.attribute(f"{DYNAMIC_GRAPH_PATH}/ROS2Publisher.inputs:data")

# 禁止 Replicator 自动接管生命周期
rep.orchestrator.set_capture_on_play(False)

# ====================================================================
# 🌟 核心破局：绕过 UI 视口，在底层直接对该相机硬激活一个 Render Product
# ====================================================================
print(f"[Sim2Real] 🔗 正在底层强制为物理相机 {CAMERA_PRIM_PATH} 创建激活的渲染通道(Render Product)...")
# 这一步直接强行打开了 C++ 渲染流水线的阀门，不需要任何 Viewport Window 配合！
prod = rep.create.render_product(CAMERA_PRIM_PATH, resolution=(1280, 720))

# 启动仿真
simulation_context.initialize_physics()
simulation_context.play()

# 渲染管线强行预热推流
print("[Sim2Real] 正在疯狂向 Hydra 引擎推流以强制注册图像话题...")
for _ in range(30):
    simulation_context.step(render=True)

print("[Sim2Real] 🚀 底层硬激活完成！请在新终端中执行 'ros2 topic list' 查看图像话题。")

# ====================================================================
# 4. 主物理步长循环
# ====================================================================
try:
    while True:
        if not simulation_app.is_running():
            break
            
        simulation_context.step(render=True)
        
        if not cube_prim.IsValid() or not camera_prim.IsValid():
            continue

        CALC_WIDTH = 1280.0
        CALC_HEIGHT = 720.0
        time_code = Usd.TimeCode.Default()
        
        # 1. 提取相机的原生位姿 View 矩阵
        usd_cam = UsdGeom.Camera(camera_prim)
        cam_xform = usd_cam.ComputeLocalToWorldTransform(time_code)
        world_to_cam_matrix = np.array(cam_xform.GetInverse()).T

        # 2. 动态解算相机标准内参
        focal_length = usd_cam.GetFocalLengthAttr().Get()            
        horiz_aperture = usd_cam.GetHorizontalApertureAttr().Get()    
        vert_aperture = usd_cam.GetVerticalApertureAttr().Get()      
        
        fov_h = 2.0 * math.atan(horiz_aperture / (2.0 * focal_length))
        fov_v = 2.0 * math.atan(vert_aperture / (2.0 * focal_length))
        
        fx = CALC_WIDTH / (2.0 * math.tan(fov_h / 2.0))
        fy = CALC_HEIGHT / (2.0 * math.tan(fov_v / 2.0))
        cx = CALC_WIDTH / 2.0
        cy = CALC_HEIGHT / 2.0

        # 3. 计算物体的实时世界位姿并转到相机空间
        cube_xformable = UsdGeom.Xformable(cube_prim)
        cube_world_matrix = cube_xformable.ComputeLocalToWorldTransform(time_code)
        cube_world_transform = np.array(cube_world_matrix).T

        world_corners = LOCAL_CORNERS @ cube_world_transform
        camera_corners_usd = world_corners @ world_to_cam_matrix

        temp_points = []
        valid_count = 0

        # 4. 投影计算
        for i in range(4):
            x_cam, y_cam, z_cam, _ = camera_corners_usd[i]
            z_depth = abs(z_cam)
            
            if z_depth > 0.01:
                u_raw = cx + (x_cam * fx) / z_depth
                v_raw = cy + (y_cam * fy) / z_depth  

                if u_raw > CALC_WIDTH * 1.5 or v_raw > CALC_HEIGHT * 1.5:
                    u = u_raw / 2.0
                    v = v_raw / 2.0
                else:
                    u = u_raw
                    v = v_raw
                
                if 0 <= u <= CALC_WIDTH and 0 <= v <= CALC_HEIGHT:
                    valid_count += 1
                
                temp_points.extend([float(u), float(v)])

        if valid_count == 4 and len(temp_points) == 8:
            projected_2d_points = temp_points
        else:
            projected_2d_points = []

        # 5. 发布连续内存段数据
        if len(projected_2d_points) == 8 and data_attr.is_valid():
            og_data = np.array(projected_2d_points, dtype=np.float64)
            og.Controller.set(data_attr, og_data)

except Exception as e:
    print(f"\n[🚨 关键错误捕捉] 主循环内部报错:")
    import traceback
    traceback.print_exc()

finally:
    print("\n[Sim2Real] 正在请求执行程序安全退出 (Safe Exit)...")
    try:
        # 手动销毁硬激活的渲染流水线句柄
        if 'prod' in locals():
            prod.destroy()
            
        simulation_context.stop()
        if stage.GetPrimAtPath(DYNAMIC_GRAPH_PATH).IsValid():
            print(f"[Sim2Real] 正在主动注销动态追加的 Graph 管道: {DYNAMIC_GRAPH_PATH}")
            stage.RemovePrim(DYNAMIC_GRAPH_PATH)
        
        for _ in range(5):
            simulation_app.update()
            
    except Exception as cleanup_err:
        print(f"[Sim2Real] 垃圾回收阶段遇到次级扰动: {cleanup_err}")
    
    simulation_app.close()
    sys.exit(0)