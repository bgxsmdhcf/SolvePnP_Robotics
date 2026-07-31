import os

from isaacsim import SimulationApp

# 如果你的场景不需要UI，可以把 headless 改为 True
simulation_app = SimulationApp(launch_config={"headless": True})

import carb.settings
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.utils.semantics import add_labels
from omni.replicator.core import Writer
from pxr import Sdf, UsdGeom


# Create a custom writer to access the annotator data
class MyWriter(Writer):
    def __init__(self, camera_params: bool = True, bounding_box_3d: bool = True):
        self.data_structure = "renderProduct"
        self.annotators = []
        if camera_params:
            self.annotators.append(rep.annotators.get("camera_params"))
        if bounding_box_3d:
            self.annotators.append(rep.annotators.get("bounding_box_3d"))
        self._frame_id = 0

    def write(self, data):
        print(f"[MyWriter][{self._frame_id}] data:{data}")
        self._frame_id += 1


# Register the writer for use
rep.writers.register_writer(MyWriter)


def run_example():
    # -----------------------------------------------------------------
    # 【修改1】替换为你自己的 USD 文件绝对路径
    # -----------------------------------------------------------------
    usd_path = "/home/nnd/elfin_robot/elfin_description/urdf/elfin10_l/elfin10_l/elfin10_l.usd" 
    #usd_path = "/home/nnd/ur_usd/official_configured_usd/Collected_ur_gripper/ur_gripper_scene.usd" 
    
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"找不到指定的USD文件: {usd_path}")
        
    # 打开你原本的 USD 场景，而不是新建空白 Stage
    omni.usd.get_context().open_stage(usd_path)
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    stage = omni.usd.get_context().get_stage()
    
    # -----------------------------------------------------------------
    # 【修改2】指定你场景中已有的物体和相机路径
    # 请根据你在 Isaac Sim 层次结构（Stage）中看到的实际路径进行修改
    # -----------------------------------------------------------------
    #cube_path = "/World/cube/Cube"         # 场景中物块的路径
    cube_path = "/World/Cube"
    camera_path = "/Camera"     # 场景中相机的路径
    #camera_path = "/World/Camera"

    # 确保路径在场景中真实存在
    cube_prim = stage.GetPrimAtPath(cube_path)
    camera_prim = stage.GetPrimAtPath(camera_path)
    
    if not cube_prim.IsValid():
        print(f"[Warning] 未能在路径 {cube_path} 找到物块，请检查路径！")
    else:
        # 为现有的物块添加语义标签，用于 3D 边界框和 Pose 识别
        add_labels(cube_prim, labels=["MyCube"], instance_name="class")

    if not camera_prim.IsValid():
        raise ValueError(f"在场景中未找到相机，请检查路径是否正确: {camera_path}")

    # -----------------------------------------------------------------
    # 【修改3】只保留你指定的相机视角，移除内置的默认透视视角 (Perspective)
    # 这样 Replicator 就只会按照你预设的相机进行输出，不会乱切视角
    # -----------------------------------------------------------------
    rp_cam = rep.create.render_product(camera_path, (1024, 1024), name="my_custom_view")

    # 使用 Annotator 获取单帧数据（直接挂载到你的相机上）
    rgb_annotator_cam = rep.annotators.get("rgb")
    rgb_annotator_cam.attach(rp_cam)

    # 自定义 Writer 初始化与挂载
    custom_writer = rep.writers.get("MyWriter")
    custom_writer.initialize(camera_params=True, bounding_box_3d=True)
    custom_writer.attach([rp_cam])

    # Pose Writer 初始化与挂载（输出到磁盘）
    pose_writer = rep.WriterRegistry.get("PoseWriter")
    out_dir = os.path.join(os.getcwd(), "_out_pose_writer")
    print(f"Output directory: {out_dir}")
    pose_writer.initialize(output_dir=out_dir, write_debug_images=True)
    pose_writer.attach([rp_cam])

    # Trigger a data capture request
    for i in range(3):
        print(f"Step {i}")
        rep.orchestrator.step()

        # Get the data from the annotator
        rgb_data_cam = rgb_annotator_cam.get_data()
        print(f"[Annotator][Cam][{i}] rgb_data_cam shape: {rgb_data_cam.shape}")

    # Detach and clean up resources
    pose_writer.detach()
    custom_writer.detach()
    rgb_annotator_cam.detach()
    rp_cam.destroy()

    # Wait for the data to be written to disk
    rep.orchestrator.wait_until_complete()


run_example()

# Let the simulation run until it is manually closed
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()