"""
date: 2024.08.27
author: Seongyong Kim
collaborator: Seungwoo Baek
description: office environment class
"""
import numpy as np
from isaacgym import gymapi
from isaacgym.torch_utils import *
import random
import math

class EnvBase:
    def __init__(self, asset_root, gym, sim, device):
        self.asset_root = asset_root
        self.gym = gym
        self.sim = sim
        self.device = device

    def set_robot_asset(self):
        raise NotImplementedError("Should implement set_robot_asset")

    def set_interior_asset(self):
        raise NotImplementedError("Should implement set_interior_asset")
    
    def set_object_asset(self):
        raise NotImplementedError("Should implement set_object_asset")
    
    def create_robot(self, env, num_env):
        raise NotImplementedError("Should implement create_robot")

    def create_assets(self):
        raise NotImplementedError("Should implement create_assets")
    
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def generate_position(x_lim,y_lim,min_distance,positions):
    while True:
        x = random.uniform(x_lim[0], x_lim[1])
        y = random.uniform(y_lim[0], y_lim[1])
        new_position = [x,y]

        if all(distance(new_position, pos) >= min_distance for pos in positions):
            return new_position

def get_object_positions(num_objects, x_lim, y_lim, distance=0.1):
    min_distance = distance

    positions = []
    for i in range(num_objects):
        print("Calculating object position...(%d/%d)"%(i+1,num_objects))
        new_pos = generate_position(x_lim, y_lim, min_distance, positions)
        positions.append(new_pos)
    positions = np.array(positions)
    print("Object positions are generated")

    return positions[:,0], positions[:,1]

class Env1(EnvBase):
    def set_robot_asset(self):
        # config UR5e asset
        print("UR5e is not ready")

    def set_wall_type(self, wall_type):
        if wall_type == "ivory":
            self.wall_color = gymapi.Vec3(220/255, 205/255, 152/255)
        elif wall_type == "white":
            self.wall_color = gymapi.Vec3(1, 1, 1)
        elif wall_type == "green":
            self.wall_color = gymapi.Vec3(24/255, 70/255, 50/255)

    def set_interior_asset(self):
        simple_options = gymapi.AssetOptions()
        simple_options.fix_base_link = True

        gravity_options = gymapi.AssetOptions()

        complex_options = gymapi.AssetOptions()
        complex_options.fix_base_link = True
        complex_options.vhacd_enabled = True
        complex_options.vhacd_params = gymapi.VhacdParams()
        complex_options.vhacd_params.resolution = 30000
        complex_options.vhacd_params.max_convex_hulls = 512
        complex_options.vhacd_params.convex_hull_approximation = False

        floor_file = "urdf/floor/wood2.urdf"
        self.floor_asset = self.gym.load_asset(self.sim, self.asset_root, floor_file, simple_options)
        self.floor_pose1 = gymapi.Transform(p=gymapi.Vec3(0.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose2 = gymapi.Transform(p=gymapi.Vec3(1.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose3 = gymapi.Transform(p=gymapi.Vec3(2.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose4 = gymapi.Transform(p=gymapi.Vec3(0.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose5 = gymapi.Transform(p=gymapi.Vec3(1.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose6 = gymapi.Transform(p=gymapi.Vec3(2.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose7 = gymapi.Transform(p=gymapi.Vec3(0.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose8 = gymapi.Transform(p=gymapi.Vec3(1.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose9 = gymapi.Transform(p=gymapi.Vec3(2.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose10 = gymapi.Transform(p=gymapi.Vec3(0.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose11 = gymapi.Transform(p=gymapi.Vec3(1.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose12 = gymapi.Transform(p=gymapi.Vec3(2.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))

        self.wall_asset_1 = self.gym.create_box(self.sim,0.1,4,3.0,simple_options)
        self.wall1_pose = gymapi.Transform(p=gymapi.Vec3(0.0,1,1.5))
        self.wall3_pose = gymapi.Transform(p=gymapi.Vec3(3.0,1,1.5))
        self.wall_asset_2 = self.gym.create_box(self.sim,3.0,0.1,3.0,simple_options)
        self.wall2_pose = gymapi.Transform(p=gymapi.Vec3(1.5,3.0,1.5))

        desk_file = "urdf/office/Desk/model.urdf"
        self.desk_asset = self.gym.load_asset(self.sim, self.asset_root, desk_file, simple_options)
        self.desk_pose = gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi),p=gymapi.Vec3(2.7-0.125,2.43-0.11,0.0))
        self.desk_pose2 = gymapi.Transform(p=gymapi.Vec3(0.42,2.43-0.11,0.0))

        sorter_pink_file ="urdf/office/Poppin_File_Sorter_Pink/model.urdf"
        self.sorter_pink_asset = self.gym.load_asset(self.sim, self.asset_root,sorter_pink_file, simple_options)
        self.sorter_pink_pose = gymapi.Transform(p=gymapi.Vec3(2.85,2.7,0.825))

        laptop_file ="urdf/office/Travel_Mate_P_series_Notebook/model.urdf"
        self.laptop_asset = self.gym.load_asset(self.sim, self.asset_root,laptop_file, simple_options)
        self.laptop_pose =gymapi.Transform(p=gymapi.Vec3(2.7-0.125,2.43-0.21,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))
        self.laptop_pose2 = gymapi.Transform(p=gymapi.Vec3(0.25,2.0,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        black_mouse_file ="urdf/office/Razer_Taipan_Black_Ambidextrous_Gaming_Mouse/model.urdf"
        self.black_mouse_asset = self.gym.load_asset(self.sim, self.asset_root,black_mouse_file, simple_options)
        self.black_mouse_pose = gymapi.Transform(p=gymapi.Vec3(2.7-0.125,2.43-0.45,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        shelf_file = "urdf/office/shelf/model.urdf"
        self.shelf_asset = self.gym.load_asset(self.sim, self.asset_root, shelf_file, complex_options)
        self.shelf_pose = gymapi.Transform(p=gymapi.Vec3(0.358,2.794,0.825))

        drawer_file = "urdf/office/Drawer/model.urdf"
        self.drawer_asset = self.gym.load_asset(self.sim, self.asset_root, drawer_file, simple_options)
        self.drawer_pose = gymapi.Transform(p=gymapi.Vec3(0.288,1.484,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        chair_file = "urdf/office/OfficeChairGrey/model.urdf"
        self.chair_asset = self.gym.load_asset(self.sim, self.asset_root, chair_file, simple_options)
        self.chair_pose = gymapi.Transform(p=gymapi.Vec3(2.2,2.3,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        pad_file = "urdf/office/Razer_Goliathus_Control_Edition_Small_Soft_Gaming_Mouse_Mat/model.urdf"
        self.pad_asset = self.gym.load_asset(self.sim, self.asset_root, pad_file, simple_options)
        self.pad_pose = gymapi.Transform(p=gymapi.Vec3(0.55,2.43,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        keyboard_file = "urdf/office/Kanex_MultiSync_Wireless_Keyboard/model.urdf"
        self.keyboard_asset = self.gym.load_asset(self.sim, self.asset_root, keyboard_file, gravity_options)
        self.keyboard_pose = gymapi.Transform(p=gymapi.Vec3(0.55,2.0,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        power_strip_file = "urdf/office/18_Power_strip/model.urdf"
        self.power_strip_asset = self.gym.load_asset(self.sim, self.asset_root, power_strip_file, gravity_options)
        self.power_strip_pose = gymapi.Transform(p=gymapi.Vec3(0.1,1.48,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        xylitol_file = "urdf/office/Xyli_Pure_Xylitol/model.urdf"
        self.xylitol_asset = self.gym.load_asset(self.sim, self.asset_root, xylitol_file, gravity_options)
        self.xylitol_pose = gymapi.Transform(p=gymapi.Vec3(0.23,1.6,0.625),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        germanium_file = "urdf/office/Germanium_GE132/model.urdf"
        self.germanium_asset = self.gym.load_asset(self.sim, self.asset_root, germanium_file, simple_options)
        self.germanium_pose = gymapi.Transform(p=gymapi.Vec3(0.21,1.43,0.625),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        cdg_file = "urdf/office/DIM_CDG/model.urdf"
        self.cdg_asset = self.gym.load_asset(self.sim, self.asset_root, cdg_file, simple_options)
        self.cdg_pose = gymapi.Transform(p=gymapi.Vec3(0.21,1.5,0.625),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        stapler_file = "urdf/office/07_stapler_0/model.urdf"
        self.stapler_asset = self.gym.load_asset(self.sim, self.asset_root, stapler_file, simple_options)
        self.stapler_pose = gymapi.Transform(p=gymapi.Vec3(0.35,1.55,0.625),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/6))

        cabinet_file = "urdf/office/MetalCabinet/model.urdf"
        self.cabinet_asset = self.gym.load_asset(self.sim, self.asset_root, cabinet_file, simple_options)
        self.cabinet_pose = gymapi.Transform(p=gymapi.Vec3(2.725,1.28,0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        p1, p2 =get_object_positions(1, [0.15,0.25], [2.65,2.85])

        alarm_file = "urdf/office/Crosley_Alarm_Clock_Vintage_Metal/model.urdf"
        self.alarm_asset = self.gym.load_asset(self.sim, self.asset_root, alarm_file, simple_options)
        self.alarm_pose = gymapi.Transform(p=gymapi.Vec3(p1,p2,0.86))

        p1, p2 =get_object_positions(2, [0.41,0.58], [2.65,2.85])

        panda_figure = "urdf/office/Android_Figure_Panda/model.urdf"
        self.panda_asset = self.gym.load_asset(self.sim, self.asset_root, panda_figure, simple_options)
        self.panda_pose = gymapi.Transform(p=gymapi.Vec3(p1[0],p2[0],0.86))

        orange_figure = "urdf/office/Android_Figure_Orange/model.urdf"
        self.orange_asset = self.gym.load_asset(self.sim, self.asset_root, orange_figure, simple_options)
        self.orange_pose = gymapi.Transform(p=gymapi.Vec3(p1[1],p2[1],0.86))
        
    def set_object_asset(self):
        print("object asset is set")

    def create_robot(self, env, num_env):
        print("creating robot")

    def create_assets(self, env, num_env):
        self.gym.create_actor(env, self.floor_asset, self.floor_pose1, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose2, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose3, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose4, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose5, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose6, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose7, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose8, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose9, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose10, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose11, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose12, "floor", num_env, 0)
        
        wall1_handle = self.gym.create_actor(env, self.wall_asset_1, self.wall1_pose, "wall1", num_env, 0)
        wall2_handle = self.gym.create_actor(env, self.wall_asset_2, self.wall2_pose, "wall2", num_env, 0)
        wall3_handle = self.gym.create_actor(env, self.wall_asset_1, self.wall3_pose, "wall3", num_env, 0)
        self.gym.set_rigid_body_color(env, wall1_handle, 0, gymapi.MESH_VISUAL, self.wall_color)
        self.gym.set_rigid_body_color(env, wall2_handle, 0, gymapi.MESH_VISUAL, self.wall_color)
        self.gym.set_rigid_body_color(env, wall3_handle, 0, gymapi.MESH_VISUAL, self.wall_color)
        
        desk_handle = self.gym.create_actor(env, self.desk_asset, self.desk_pose, "desk", num_env, 0)
        desk2_handle = self.gym.create_actor(env, self.desk_asset, self.desk_pose2, "desk2", num_env, 0)
        self.gym.set_actor_scale(env, desk_handle, 1.5)
        self.gym.set_actor_scale(env, desk2_handle, 1.5)
        sorter_pink_handle = self.gym.create_actor(env, self.sorter_pink_asset, self.sorter_pink_pose, "sorter_pink", num_env, 0)
        laptop_handle = self.gym.create_actor(env, self.laptop_asset, self.laptop_pose, "laptop", num_env, 0)
        black_mouse_handle = self.gym.create_actor(env, self.black_mouse_asset, self.black_mouse_pose, "black_mouse", num_env, 0)
        shelf_handle = self.gym.create_actor(env, self.shelf_asset, self.shelf_pose, "shelf", num_env, 0)
        self.gym.set_rigid_body_color(env,shelf_handle,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.1,0.1,0.1))
        self.gym.set_actor_scale(env, shelf_handle, 0.8)
        drawer_handle = self.gym.create_actor(env, self.drawer_asset, self.drawer_pose, "drawer", num_env, 0)
        chair_handle = self.gym.create_actor(env, self.chair_asset, self.chair_pose, "chair", num_env, 0)
        pad_handle = self.gym.create_actor(env, self.pad_asset, self.pad_pose, "pad", num_env, 0)
        keyboard_handle = self.gym.create_actor(env, self.keyboard_asset, self.keyboard_pose, "keyboard", num_env, 0)
        power_strip_handle = self.gym.create_actor(env, self.power_strip_asset, self.power_strip_pose, "power_strip", num_env, 0)
        laptop2_handle = self.gym.create_actor(env, self.laptop_asset, self.laptop_pose2, "laptop2", num_env, 0)
        xylitol_handle = self.gym.create_actor(env, self.xylitol_asset, self.xylitol_pose, "xylitol", num_env, 0)
        germanium_handle = self.gym.create_actor(env, self.germanium_asset, self.germanium_pose, "germanium", num_env, 0)
        cdg_handle = self.gym.create_actor(env, self.cdg_asset, self.cdg_pose, "cdg", num_env, 0)
        stapler_handle = self.gym.create_actor(env, self.stapler_asset, self.stapler_pose, "stapler", num_env, 0)
        cabinet_handle = self.gym.create_actor(env, self.cabinet_asset, self.cabinet_pose, "cabinet", num_env, 0)
        alarm_handle = self.gym.create_actor(env, self.alarm_asset, self.alarm_pose, "alarm", num_env, 0)
        panda_handle = self.gym.create_actor(env, self.panda_asset, self.panda_pose, "panda", num_env, 0)
        orange_handle = self.gym.create_actor(env, self.orange_asset, self.orange_pose, "orange", num_env, 0)


class Env2(EnvBase):
    def set_robot_asset(self):
        print("UR5e is not ready")

    def set_wall_type(self, wall_type):
        if wall_type == "ivory":
            self.wall_color = gymapi.Vec3(220/255, 205/255, 152/255)
        elif wall_type == "white":
            self.wall_color = gymapi.Vec3(1, 1, 1)
        elif wall_type == "green":
            self.wall_color = gymapi.Vec3(24/255, 70/255, 50/255)

    def set_interior_asset(self):
        simple_options = gymapi.AssetOptions()
        simple_options.fix_base_link = True

        complex_options = gymapi.AssetOptions()
        complex_options.fix_base_link = True
        complex_options.vhacd_enabled = True
        complex_options.vhacd_params = gymapi.VhacdParams()
        complex_options.vhacd_params.resolution = 30000
        complex_options.vhacd_params.max_convex_hulls = 512
        complex_options.vhacd_params.convex_hull_approximation = False

        floor_file = "urdf/floor/wood3.urdf"
        self.floor_asset = self.gym.load_asset(self.sim, self.asset_root, floor_file, simple_options)
        self.floor_pose1 = gymapi.Transform(p=gymapi.Vec3(0.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose2 = gymapi.Transform(p=gymapi.Vec3(1.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose3 = gymapi.Transform(p=gymapi.Vec3(2.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose4 = gymapi.Transform(p=gymapi.Vec3(0.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose5 = gymapi.Transform(p=gymapi.Vec3(1.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose6 = gymapi.Transform(p=gymapi.Vec3(2.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose7 = gymapi.Transform(p=gymapi.Vec3(0.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose8 = gymapi.Transform(p=gymapi.Vec3(1.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose9 = gymapi.Transform(p=gymapi.Vec3(2.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose10 = gymapi.Transform(p=gymapi.Vec3(0.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose11 = gymapi.Transform(p=gymapi.Vec3(1.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose12 = gymapi.Transform(p=gymapi.Vec3(2.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))

        self.wall_asset_1 = self.gym.create_box(self.sim,0.1,4.0,3.0,simple_options)
        self.wall1_pose = gymapi.Transform(p=gymapi.Vec3(0.0,1.0,1.5))
        self.wall3_pose = gymapi.Transform(p=gymapi.Vec3(3.0,1.0,1.5))
        self.wall_asset_2 = self.gym.create_box(self.sim,3.0,0.1,3.0,simple_options)
        self.wall2_pose = gymapi.Transform(p=gymapi.Vec3(1.5,3.0,1.5))

        desk_file = "urdf/office/Desk/model_dark_marble.urdf"
        self.desk_asset = self.gym.load_asset(self.sim, self.asset_root, desk_file, simple_options)
        self.desk_pose = gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi),p=gymapi.Vec3(2.7-0.125,2.43-0.71,0.0))
        self.desk_pose2 = gymapi.Transform(p=gymapi.Vec3(0.42,2.43-0.71,0.0))

        sorter_blue_file ="urdf/office/Poppin_File_Sorter_Blue/model.urdf"
        self.sorter_blue_asset = self.gym.load_asset(self.sim, self.asset_root,sorter_blue_file, simple_options)
        self.sorter_blue_pose = gymapi.Transform(p=gymapi.Vec3(0.15,2.2,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        sorter_pink_file ="urdf/office/Poppin_File_Sorter_Pink/model.urdf"
        self.sorter_pink_asset = self.gym.load_asset(self.sim, self.asset_root,sorter_pink_file, simple_options)
        self.sorter_pink_pose = gymapi.Transform(p=gymapi.Vec3(2.85,2.2,0.825))

        laptop_file ="urdf/office/Lenovo_Yoga_2_11/model.urdf"
        self.laptop_asset = self.gym.load_asset(self.sim, self.asset_root,laptop_file, simple_options)
        self.laptop_pose = gymapi.Transform(p=gymapi.Vec3(0.375,2.43-0.61,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        laptop2_file ="urdf/office/Travel_Mate_P_series_Notebook/model.urdf"
        self.laptop2_asset = self.gym.load_asset(self.sim, self.asset_root,laptop2_file, simple_options)
        self.laptop2_pose =gymapi.Transform(p=gymapi.Vec3(2.7-0.125,2.43-0.71,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        white_mouse_file ="urdf/office/Razer_Taipan_White_Ambidextrous_Gaming_Mouse/model.urdf"
        self.white_mouse_asset = self.gym.load_asset(self.sim, self.asset_root,white_mouse_file, simple_options)
        self.white_mouse_pose = gymapi.Transform(p=gymapi.Vec3(0.375,2.03,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        black_mouse_file ="urdf/office/Razer_Taipan_Black_Ambidextrous_Gaming_Mouse/model.urdf"
        self.black_mouse_asset = self.gym.load_asset(self.sim, self.asset_root,black_mouse_file, simple_options)
        self.black_mouse_pose = gymapi.Transform(p=gymapi.Vec3(2.7-0.125,2.43-0.92,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))
        
        chair_file = "urdf/office/OfficeChairGrey/model.urdf"
        self.chair_asset = self.gym.load_asset(self.sim, self.asset_root, chair_file, simple_options)
        self.chair_pose = gymapi.Transform(p=gymapi.Vec3(2.2,1.8,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        cabinet_file = "urdf/office/WhiteCabinet/model.urdf"
        self.cabinet_asset = self.gym.load_asset(self.sim, self.asset_root, cabinet_file, simple_options)
        self.cabinet_pose = gymapi.Transform(p=gymapi.Vec3(0.275,0.638,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))
        self.cabinet_pose2 = gymapi.Transform(p=gymapi.Vec3(2.725,0.638,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        book_shelf_file = "urdf/office/17_Book_shelf/model.urdf"
        self.book_shelf_asset = self.gym.load_asset(self.sim, self.asset_root, book_shelf_file, simple_options)
        self.book_shelf_pose = gymapi.Transform(p=gymapi.Vec3(0.21,1.16,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        mug_file = "urdf/office/Room_Essentials_Mug_White_Yellow/model.urdf"
        self.mug_asset = self.gym.load_asset(self.sim, self.asset_root, mug_file, simple_options)
        self.mug_pose = gymapi.Transform(p=gymapi.Vec3(0.34,1.45,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/4))

        pencil_sharpener_file = "urdf/office/03_Pencil_sharpener/model.urdf"
        self.pencil_sharpener_asset = self.gym.load_asset(self.sim, self.asset_root, pencil_sharpener_file, simple_options)
        self.pencil_sharpener_pose = gymapi.Transform(p=gymapi.Vec3(0.1,1.05,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        flower_pot_file = "urdf/office/20_Flower_pot/model.urdf"
        self.flower_pot_asset = self.gym.load_asset(self.sim, self.asset_root, flower_pot_file, simple_options)
        self.flower_pot_pose = gymapi.Transform(p=gymapi.Vec3(0.12,0.3,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))
        self.flower_pot_pose2 = gymapi.Transform(p=gymapi.Vec3(0.12,0.4,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))
        self.flower_pot_pose3 = gymapi.Transform(p=gymapi.Vec3(0.12,0.5,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))
        
        camera_file = "urdf/office/24_Camera/model.urdf"
        self.camera_asset = self.gym.load_asset(self.sim, self.asset_root, camera_file, simple_options)
        self.camera_pose = gymapi.Transform(p=gymapi.Vec3(0.12,0.9,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))
        self.camera_pose2 = gymapi.Transform(p=gymapi.Vec3(1.265,2.755,1.635))

        ink_black_file = "urdf/office/Canon_Pixma_Ink_Cartridge_8/model.urdf"
        self.ink_black_asset = self.gym.load_asset(self.sim, self.asset_root, ink_black_file, simple_options)
        self.ink_black_pose = gymapi.Transform(p=gymapi.Vec3(0.10,0.65,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        ink_green_file = "urdf/office/Canon_Pixma_Ink_Cartridge_8_Green/model.urdf"
        self.ink_green_asset = self.gym.load_asset(self.sim, self.asset_root, ink_green_file, simple_options)
        self.ink_green_pose = gymapi.Transform(p=gymapi.Vec3(0.10,0.75,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        ink_red_file = "urdf/office/Canon_Pixma_Ink_Cartridge_8_Red/model.urdf"
        self.ink_red_asset = self.gym.load_asset(self.sim, self.asset_root, ink_red_file, simple_options)
        self.ink_red_pose = gymapi.Transform(p=gymapi.Vec3(0.15,0.65,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        shelf_file = "urdf/office/shelf/model.urdf"
        self.shelf_asset = self.gym.load_asset(self.sim, self.asset_root, shelf_file, complex_options)
        self.shelf_pose = gymapi.Transform(p=gymapi.Vec3(1.115,2.755,1.225))
        self.shelf_pose2 = gymapi.Transform(p=gymapi.Vec3(1.885,2.755,1.225))

        p1, p2 =get_object_positions(5, [2.55,2.9], [0.238,1.038])

        mario_file = "urdf/office/Nintendo_Mario_Action_Figure/model.urdf"
        self.mario_asset = self.gym.load_asset(self.sim, self.asset_root, mario_file, simple_options)
        self.mario_pose = gymapi.Transform(p=gymapi.Vec3(p1[0],p2[0],1.1))
        
        yoshi_file = "urdf/office/Nintendo_Yoshi_Action_Figure/model.urdf"
        self.yoshi_asset = self.gym.load_asset(self.sim, self.asset_root, yoshi_file, simple_options)
        self.yoshi_pose = gymapi.Transform(p=gymapi.Vec3(p1[1],p2[1],1.1),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        spider_file = "urdf/office/SpiderMan/model.urdf"
        self.spider_asset = self.gym.load_asset(self.sim, self.asset_root, spider_file, simple_options)
        self.spider_pose = gymapi.Transform(p=gymapi.Vec3(p1[2],p2[2],1.1),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        transmormer_file = "urdf/office/Transformers/model.urdf"
        self.transformer_asset = self.gym.load_asset(self.sim, self.asset_root, transmormer_file, simple_options)
        self.transformer_pose = gymapi.Transform(p=gymapi.Vec3(p1[3],p2[3],1.1),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        rocket_file = "urdf/office/Rocket/model.urdf"
        self.rocket_asset = self.gym.load_asset(self.sim, self.asset_root, rocket_file, simple_options)
        self.rocket_pose = gymapi.Transform(p=gymapi.Vec3(p1[4],p2[4],1.1),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi))

        p1, p2 =get_object_positions(3, [0.815,1.065], [2.61,2.9])

        xylitol_file = "urdf/office/Xyli_Pure_Xylitol/model.urdf"
        self.xylitol_asset = self.gym.load_asset(self.sim, self.asset_root, xylitol_file, simple_options)
        self.xylitol_pose = gymapi.Transform(p=gymapi.Vec3(p1[0],p2[0],1.27))

        germanium_file = "urdf/office/Germanium_GE132/model.urdf"
        self.germanium_asset = self.gym.load_asset(self.sim, self.asset_root, germanium_file, simple_options)
        self.germanium_pose = gymapi.Transform(p=gymapi.Vec3(p1[1],p2[1],1.27))

        cdg_file = "urdf/office/DIM_CDG/model.urdf"
        self.cdg_asset = self.gym.load_asset(self.sim, self.asset_root, cdg_file, simple_options)
        self.cdg_pose = gymapi.Transform(p=gymapi.Vec3(p1[2],p2[2],1.27))

    def set_object_asset(self):
        print("object asset is set")
    
    def create_robot(self, env, num_env):
        print("creating robot")

    def create_assets(self, env, num_env):
        self.gym.create_actor(env, self.floor_asset, self.floor_pose1, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose2, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose3, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose4, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose5, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose6, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose7, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose8, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose9, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose10, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose11, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose12, "floor", num_env, 0)
        
        wall1_handle = self.gym.create_actor(env, self.wall_asset_1, self.wall1_pose, "wall1", num_env, 0)
        wall2_handle = self.gym.create_actor(env, self.wall_asset_2, self.wall2_pose, "wall2", num_env, 0)
        wall3_handle = self.gym.create_actor(env, self.wall_asset_1, self.wall3_pose, "wall3", num_env, 0)
        self.gym.set_rigid_body_color(env, wall1_handle, 0, gymapi.MESH_VISUAL, self.wall_color)
        self.gym.set_rigid_body_color(env, wall2_handle, 0, gymapi.MESH_VISUAL, self.wall_color)
        self.gym.set_rigid_body_color(env, wall3_handle, 0, gymapi.MESH_VISUAL, self.wall_color)
        
        desk_handle = self.gym.create_actor(env, self.desk_asset, self.desk_pose, "desk", num_env, 0)
        desk2_handle = self.gym.create_actor(env, self.desk_asset, self.desk_pose2, "desk2", num_env, 0)
        self.gym.set_actor_scale(env, desk_handle, 1.5)
        self.gym.set_actor_scale(env, desk2_handle, 1.5)

        sorter_blue_handle = self.gym.create_actor(env, self.sorter_blue_asset, self.sorter_blue_pose, "sorter_blue", num_env, 0)
        sorter_pink_handle = self.gym.create_actor(env, self.sorter_pink_asset, self.sorter_pink_pose, "sorter_pink", num_env, 0)
        laptop_handle = self.gym.create_actor(env, self.laptop_asset, self.laptop_pose, "laptop", num_env, 0)
        laptop2_handle = self.gym.create_actor(env, self.laptop2_asset, self.laptop2_pose, "laptop2", num_env, 0)
        white_mouse_handle = self.gym.create_actor(env, self.white_mouse_asset, self.white_mouse_pose, "white_mouse", num_env, 0)
        black_mouse_handle = self.gym.create_actor(env, self.black_mouse_asset, self.black_mouse_pose, "black_mouse", num_env, 0)
        chair_handle = self.gym.create_actor(env, self.chair_asset, self.chair_pose, "chair", num_env, 0)
        cabinet_handle = self.gym.create_actor(env, self.cabinet_asset, self.cabinet_pose, "cabinet", num_env, 0)
        cabinet2_handle = self.gym.create_actor(env, self.cabinet_asset, self.cabinet_pose2, "cabinet2", num_env, 0)
        book_shelf_handle = self.gym.create_actor(env, self.book_shelf_asset, self.book_shelf_pose, "book_shelf", num_env, 0)
        mug_handle = self.gym.create_actor(env, self.mug_asset, self.mug_pose, "mug", num_env, 0)
        pencil_sharpener_handle = self.gym.create_actor(env, self.pencil_sharpener_asset, self.pencil_sharpener_pose,"pencil_sharpener", num_env, 0)
        flower_pot_handle = self.gym.create_actor(env, self.flower_pot_asset, self.flower_pot_pose,"flower_pot", num_env, 0)
        flower_pot_handle2 = self.gym.create_actor(env, self.flower_pot_asset, self.flower_pot_pose2,"flower_pot", num_env, 0)
        flower_pot_handle3 = self.gym.create_actor(env, self.flower_pot_asset, self.flower_pot_pose3,"flower_pot", num_env, 0)
        camera_handle = self.gym.create_actor(env, self.camera_asset, self.camera_pose, "camera", num_env, 0)
        camera_handle2 = self.gym.create_actor(env, self.camera_asset, self.camera_pose2, "camera2", num_env, 0)
        ink_black_handle = self.gym.create_actor(env, self.ink_black_asset, self.ink_black_pose, "blackink", num_env, 0)
        ink_green_handle = self.gym.create_actor(env, self.ink_green_asset, self.ink_green_pose, "greenink", num_env, 0)
        ink_red_handle = self.gym.create_actor(env, self.ink_red_asset, self.ink_red_pose, "redink", num_env, 0)
        shelf_handle = self.gym.create_actor(env, self.shelf_asset, self.shelf_pose, "shelf", num_env, 0)
        shelf_handle2 = self.gym.create_actor(env, self.shelf_asset, self.shelf_pose2, "shelf2", num_env, 0)
        #spider_handle = self.gym.create_actor(env, self.spider_asset, self.spider_pose, "spider", num_env, 0)
        #self.gym.set_actor_scale(env, spider_handle, 0.5)
        yoshi_handle = self.gym.create_actor(env, self.yoshi_asset, self.yoshi_pose, "yoshi", num_env, 0)
        mario_handle = self.gym.create_actor(env, self.mario_asset, self.mario_pose, "mario", num_env, 0)
        transformer_handle = self.gym.create_actor(env, self.transformer_asset, self.transformer_pose, "transformer", num_env, 0)
        self.gym.set_actor_scale(env, transformer_handle, 0.5)
        rocket_handle = self.gym.create_actor(env, self.rocket_asset, self.rocket_pose, "rocket", num_env, 0)
        xylitol_handle = self.gym.create_actor(env, self.xylitol_asset, self.xylitol_pose, "xylitol", num_env, 0)
        germanium_handle = self.gym.create_actor(env, self.germanium_asset, self.germanium_pose, "germanium", num_env, 0)
        cdg_handle = self.gym.create_actor(env, self.cdg_asset, self.cdg_pose, "cdg", num_env, 0)

class Env3(EnvBase):
    def set_robot_asset(self):
        print("UR5e is not ready")

    def set_wall_type(self, wall_type):
        if wall_type == "ivory":
            self.wall_color = gymapi.Vec3(220/255, 205/255, 152/255)
        elif wall_type == "white":
            self.wall_color = gymapi.Vec3(1, 1, 1)
        elif wall_type == "green":
            self.wall_color = gymapi.Vec3(24/255, 70/255, 50/255)

    def set_interior_asset(self):
        simple_options = gymapi.AssetOptions()
        simple_options.fix_base_link = True

        gravity_options = gymapi.AssetOptions()

        complex_options = gymapi.AssetOptions()
        complex_options.fix_base_link = True
        complex_options.vhacd_enabled = True
        complex_options.vhacd_params = gymapi.VhacdParams()
        complex_options.vhacd_params.resolution = 30000
        complex_options.vhacd_params.max_convex_hulls = 512
        complex_options.vhacd_params.convex_hull_approximation = False

        floor_file = "urdf/floor/wood2.urdf"
        self.floor_asset = self.gym.load_asset(self.sim, self.asset_root, floor_file, simple_options)
        self.floor_pose1 = gymapi.Transform(p=gymapi.Vec3(0.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose2 = gymapi.Transform(p=gymapi.Vec3(1.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose3 = gymapi.Transform(p=gymapi.Vec3(2.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose4 = gymapi.Transform(p=gymapi.Vec3(0.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose5 = gymapi.Transform(p=gymapi.Vec3(1.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose6 = gymapi.Transform(p=gymapi.Vec3(2.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose7 = gymapi.Transform(p=gymapi.Vec3(0.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose8 = gymapi.Transform(p=gymapi.Vec3(1.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose9 = gymapi.Transform(p=gymapi.Vec3(2.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose10 = gymapi.Transform(p=gymapi.Vec3(0.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose11 = gymapi.Transform(p=gymapi.Vec3(1.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose12 = gymapi.Transform(p=gymapi.Vec3(2.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))

        self.wall_asset_1 = self.gym.create_box(self.sim,0.1,3.95,3.0,simple_options)
        self.wall1_pose = gymapi.Transform(p=gymapi.Vec3(0.0,0.975,1.5))
        self.wall3_pose = gymapi.Transform(p=gymapi.Vec3(3.0,0.975,1.5))
        self.wall_asset_2 = self.gym.create_box(self.sim,3.0,0.1,3.0,simple_options)
        self.wall2_pose = gymapi.Transform(p=gymapi.Vec3(1.5,3.0,1.5))

        desk_file = "urdf/office/Desk/model_bright_marble.urdf"
        self.desk_asset = self.gym.load_asset(self.sim, self.asset_root, desk_file, simple_options)
        self.desk_pose = gymapi.Transform(p=gymapi.Vec3(0.68,2.575,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        sorter_pink_file ="urdf/office/Poppin_File_Sorter_Pink/model.urdf"
        self.sorter_pink_asset = self.gym.load_asset(self.sim, self.asset_root,sorter_pink_file, simple_options)
        self.sorter_pink_pose = gymapi.Transform(p=gymapi.Vec3(2.85,1.9,0.825))

        laptop_file ="urdf/office/Travel_Mate_P_series_Notebook/model.urdf"
        self.laptop_asset = self.gym.load_asset(self.sim, self.asset_root,laptop_file, gravity_options)
        self.laptop_pose =gymapi.Transform(p=gymapi.Vec3(2.7-0.125,2.43-1.01,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        black_mouse_file ="urdf/office/Razer_Taipan_Black_Ambidextrous_Gaming_Mouse/model.urdf"
        self.black_mouse_asset = self.gym.load_asset(self.sim, self.asset_root,black_mouse_file, gravity_options)
        self.black_mouse_pose = gymapi.Transform(p=gymapi.Vec3(2.7-0.125,2.43-1.27,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        cabinet_file = "urdf/office/WhiteCabinet/model.urdf"
        self.cabinet_asset = self.gym.load_asset(self.sim, self.asset_root, cabinet_file, simple_options)
        self.cabinet_pose = gymapi.Transform(p=gymapi.Vec3(2.725,2.5,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        chair_file = "urdf/office/OfficeChairGrey/model.urdf"
        self.chair_asset = self.gym.load_asset(self.sim, self.asset_root, chair_file, simple_options)
        self.chair_pose = gymapi.Transform(p=gymapi.Vec3(2.2,1.4,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        trashbin_file = "urdf/office/TrashBin/model.urdf"
        self.trashbin_asset = self.gym.load_asset(self.sim, self.asset_root, trashbin_file, complex_options)
        self.trashbin_pose = gymapi.Transform(p=gymapi.Vec3(1.42,2.525,0.0))

        book_shelf_file = "urdf/office/17_Book_shelf/model.urdf"
        self.book_shelf_asset = self.gym.load_asset(self.sim, self.asset_root, book_shelf_file, simple_options)
        self.book_shelf_pose = gymapi.Transform(p=gymapi.Vec3(0.11,2.78,0.825))

        osilloscope_file = "urdf/office/00_Oscilloscope/model.urdf"
        self.osilloscope_asset = self.gym.load_asset(self.sim, self.asset_root, osilloscope_file, gravity_options)
        self.osilloscope_pose = gymapi.Transform(p=gymapi.Vec3(0.4,2.8,0.875),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/6))

        camera_file = "urdf/office/24_Camera/model.urdf"
        self.camera_asset = self.gym.load_asset(self.sim, self.asset_root, camera_file, gravity_options)
        self.camera_pose = gymapi.Transform(p=gymapi.Vec3(0.75,2.5,0.875))

        lens_file = "urdf/office/Nikon_1_AW1_w11275mm_Lens_Silver/model.urdf"
        self.lens_asset = self.gym.load_asset(self.sim, self.asset_root, lens_file, gravity_options)
        self.lens_pose = gymapi.Transform(p=gymapi.Vec3(0.6,2.4,0.875))

        driver_file = "urdf/office/043_phillips_screwdriver/model.urdf"
        self.driver_asset = self.gym.load_asset(self.sim, self.asset_root, driver_file, gravity_options)
        self.driver_pose = gymapi.Transform(p=gymapi.Vec3(0.9,2.45,0.875),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/6))

        desk2_file = "urdf/office/AdjTable/model.urdf"
        self.desk2_asset = self.gym.load_asset(self.sim, self.asset_root, desk2_file, simple_options)
        self.desk2_pose = gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(2.59,1.25,0.0))

        p1, p2 =get_object_positions(3, [0.7,1.2], [2.65,2.85])

        ink_black_file = "urdf/office/Canon_Pixma_Ink_Cartridge_8/model.urdf"
        self.ink_black_asset = self.gym.load_asset(self.sim, self.asset_root, ink_black_file, gravity_options)
        self.ink_black_pose = gymapi.Transform(p=gymapi.Vec3(p1[0],p2[0],0.828),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        ink_green_file = "urdf/office/Canon_Pixma_Ink_Cartridge_8_Green/model.urdf"
        self.ink_green_asset = self.gym.load_asset(self.sim, self.asset_root, ink_green_file, gravity_options)
        self.ink_green_pose = gymapi.Transform(p=gymapi.Vec3(p1[1],p2[1],0.828),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 2*np.pi/3))

        ink_red_file = "urdf/office/Canon_Pixma_Ink_Cartridge_8_Red/model.urdf"
        self.ink_red_asset = self.gym.load_asset(self.sim, self.asset_root, ink_red_file, gravity_options)
        self.ink_red_pose = gymapi.Transform(p=gymapi.Vec3(p1[2],p2[2],0.828),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/3))

        radiator_file = "urdf/office/Radiator/model.urdf"
        self.radiator_asset = self.gym.load_asset(self.sim, self.asset_root, radiator_file, simple_options)
        self.radiator_pose = gymapi.Transform(p=gymapi.Vec3(0.05,1.0,0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), np.pi/2))

    def set_object_asset(self):
        print("object asset is set")
    
    def create_robot(self, env, num_env):
        print("creating robot")

    def create_assets(self, env, num_env):
        self.gym.create_actor(env, self.floor_asset, self.floor_pose1, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose2, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose3, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose4, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose5, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose6, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose7, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose8, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose9, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose10, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose11, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose12, "floor", num_env, 0)

        wall1_handle = self.gym.create_actor(env, self.wall_asset_1, self.wall1_pose, "wall1", num_env, 0)
        wall2_handle = self.gym.create_actor(env, self.wall_asset_2, self.wall2_pose, "wall2", num_env, 0)
        wall3_handle = self.gym.create_actor(env, self.wall_asset_1, self.wall3_pose, "wall3", num_env, 0)
        self.gym.set_rigid_body_color(env, wall1_handle, 0, gymapi.MESH_VISUAL, self.wall_color)
        self.gym.set_rigid_body_color(env, wall2_handle, 0, gymapi.MESH_VISUAL, self.wall_color)
        self.gym.set_rigid_body_color(env, wall3_handle, 0, gymapi.MESH_VISUAL, self.wall_color)
        
        desk_handle = self.gym.create_actor(env, self.desk_asset, self.desk_pose, "desk", num_env, 0)
        desk2_handle = self.gym.create_actor(env, self.desk2_asset, self.desk2_pose, "desk2", num_env, 0)
        self.gym.set_actor_scale(env, desk_handle, 1.5)

        sorter_pink_handle = self.gym.create_actor(env, self.sorter_pink_asset, self.sorter_pink_pose, "sorter_pink", num_env, 0)
        laptop_handle = self.gym.create_actor(env, self.laptop_asset, self.laptop_pose, "laptop", num_env, 0)
        black_mouse_handle = self.gym.create_actor(env, self.black_mouse_asset, self.black_mouse_pose, "black_mouse", num_env, 0)

        cabinet_handle = self.gym.create_actor(env, self.cabinet_asset, self.cabinet_pose, "cabinet", num_env, 0)
        chair_handle = self.gym.create_actor(env, self.chair_asset, self.chair_pose, "chair", num_env, 0)
        trashbin_handle = self.gym.create_actor(env, self.trashbin_asset, self.trashbin_pose, "trashbin", num_env, 0)
        book_shelf_handle = self.gym.create_actor(env, self.book_shelf_asset, self.book_shelf_pose, "bookshelf", num_env, 0)
        osilloscope_handle = self.gym.create_actor(env, self.osilloscope_asset, self.osilloscope_pose, "osilloscope", num_env, 0)
        self.gym.set_actor_scale(env, osilloscope_handle, 5)
        camera_handle = self.gym.create_actor(env, self.camera_asset, self.camera_pose, "camera", num_env, 0)
        lens_handle = self.gym.create_actor(env, self.lens_asset, self.lens_pose, "lens", num_env, 0)
        driver_handle = self.gym.create_actor(env, self.driver_asset, self.driver_pose, "driver", num_env, 0)
        ink_black_handle = self.gym.create_actor(env, self.ink_black_asset, self.ink_black_pose, "blackink", num_env, 0)
        ink_green_handle = self.gym.create_actor(env, self.ink_green_asset, self.ink_green_pose, "greenink", num_env, 0)
        ink_red_handle = self.gym.create_actor(env, self.ink_red_asset, self.ink_red_pose, "redink", num_env, 0)
        radiator_handle = self.gym.create_actor(env, self.radiator_asset, self.radiator_pose, "radiator", num_env, 0)

class Env4(EnvBase):
    def set_robot_asset(self):
        print("UR5e is not ready")

    def set_wall_type(self, wall_type):
        if wall_type == "ivory":
            self.wall_color = gymapi.Vec3(220/255, 205/255, 152/255)
        elif wall_type == "white":
            self.wall_color = gymapi.Vec3(1, 1, 1)
        elif wall_type == "green":
            self.wall_color = gymapi.Vec3(24/255, 70/255, 50/255)

    def set_interior_asset(self):
        simple_options = gymapi.AssetOptions()
        simple_options.fix_base_link = True

        gravity_options = gymapi.AssetOptions()
        gravity_options.fix_base_link = False

        complex_options = gymapi.AssetOptions()
        complex_options.fix_base_link = True
        complex_options.vhacd_enabled = True
        complex_options.vhacd_params = gymapi.VhacdParams()
        complex_options.vhacd_params.resolution = 30000
        complex_options.vhacd_params.max_convex_hulls = 512
        complex_options.vhacd_params.convex_hull_approximation = False

        floor_file = "urdf/floor/wood3.urdf"
        self.floor_asset = self.gym.load_asset(self.sim, self.asset_root, floor_file, simple_options)
        self.floor_pose1 = gymapi.Transform(p=gymapi.Vec3(0.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose2 = gymapi.Transform(p=gymapi.Vec3(1.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose3 = gymapi.Transform(p=gymapi.Vec3(2.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose4 = gymapi.Transform(p=gymapi.Vec3(3.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose5 = gymapi.Transform(p=gymapi.Vec3(3.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose6 = gymapi.Transform(p=gymapi.Vec3(3.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose7 = gymapi.Transform(p=gymapi.Vec3(0.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose8 = gymapi.Transform(p=gymapi.Vec3(1.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose9 = gymapi.Transform(p=gymapi.Vec3(2.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose10 = gymapi.Transform(p=gymapi.Vec3(0.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose11 = gymapi.Transform(p=gymapi.Vec3(1.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose12 = gymapi.Transform(p=gymapi.Vec3(2.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose13 = gymapi.Transform(p=gymapi.Vec3(0.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose14 = gymapi.Transform(p=gymapi.Vec3(1.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose15 = gymapi.Transform(p=gymapi.Vec3(2.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose16 = gymapi.Transform(p=gymapi.Vec3(3.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))

        self.wall_asset_1 = self.gym.create_box(self.sim,0.1,3.95,3.0,simple_options)
        self.wall1_pose = gymapi.Transform(p=gymapi.Vec3(0.0,0.975,1.5))
        self.wall3_pose = gymapi.Transform(p=gymapi.Vec3(4.0,0.975,1.5))
        self.wall_asset_2 = self.gym.create_box(self.sim,4.0,0.1,3.0,simple_options)
        self.wall2_pose = gymapi.Transform(p=gymapi.Vec3(2.0,3.0,1.5))

        desk_file = "urdf/office/SmallCubicle/model.urdf"
        self.desk_asset = self.gym.load_asset(self.sim, self.asset_root, desk_file, complex_options)
        self.desk_pose = gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.525,1.775,0.0))
        self.desk_pose2 = gymapi.Transform(p=gymapi.Vec3(3.525,0.425,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        shelf_file = "urdf/office/shelf/model.urdf"
        self.shelf_asset = self.gym.load_asset(self.sim, self.asset_root, shelf_file, complex_options)
        self.shelf_pose = gymapi.Transform(p=gymapi.Vec3(2.0,3.0-0.27,1.08))

        cabinet_file = "urdf/office/WhiteCabinet/model.urdf"
        self.cabinet_asset = self.gym.load_asset(self.sim, self.asset_root, cabinet_file, simple_options)
        self.cabinet_pose = gymapi.Transform(p=gymapi.Vec3(2.0,3.0-0.275,0.0))
        self.cabinet_pose2 = gymapi.Transform(p=gymapi.Vec3(1.1,3.0-0.275,0.0))

        chair_file = "urdf/office/OfficeChairBlack/model.urdf"
        self.chair_asset = self.gym.load_asset(self.sim, self.asset_root, chair_file, simple_options)
        self.chair_pose = gymapi.Transform(p=gymapi.Vec3(3.0,0.45,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        laptop_file ="urdf/office/Travel_Mate_P_series_Notebook/model.urdf"
        self.laptop_asset = self.gym.load_asset(self.sim, self.asset_root,laptop_file, simple_options)
        self.laptop_pose =gymapi.Transform(p=gymapi.Vec3(3.6,0.45,0.745),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))
        
        black_mouse_file ="urdf/office/Razer_Taipan_Black_Ambidextrous_Gaming_Mouse/model.urdf"
        self.black_mouse_asset = self.gym.load_asset(self.sim, self.asset_root,black_mouse_file, simple_options)
        self.black_mouse_pose = gymapi.Transform(p=gymapi.Vec3(3.6,0.24,0.745),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))
        
        cup_file = "urdf/office/PlasticCup/model.urdf"
        self.cup_asset = self.gym.load_asset(self.sim, self.asset_root, cup_file, simple_options)
        self.cup_pose = gymapi.Transform(p=gymapi.Vec3(3.65,0.75,0.82),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))
        self.cup_pose2 = gymapi.Transform(p=gymapi.Vec3(3.8,1.45,0.82),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        power_strip_file = "urdf/office/18_Power_strip/model.urdf"
        self.power_strip_asset = self.gym.load_asset(self.sim, self.asset_root, power_strip_file, simple_options)
        self.power_strip_pose = gymapi.Transform(p=gymapi.Vec3(3.85,2.2,0.76),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        pencil_sharpener_file = "urdf/office/03_Pencil_sharpener/model.urdf"
        self.pencil_sharpener_asset = self.gym.load_asset(self.sim, self.asset_root, pencil_sharpener_file, simple_options)
        self.pencil_sharpener_pose = gymapi.Transform(p=gymapi.Vec3(3.8,1.35,0.76),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        coffeemaker_file = "urdf/office/Coffeemaker/model.urdf"
        self.coffeemaker_asset = self.gym.load_asset(self.sim, self.asset_root, coffeemaker_file, simple_options)
        self.coffeemaker_pose = gymapi.Transform(p=gymapi.Vec3(1.4,3.0-0.15,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 1*np.pi/8))

        coffee_can_file = "urdf/office/Nescafe_Instant_Coffee/model.urdf"
        self.coffee_can_asset = self.gym.load_asset(self.sim, self.asset_root, coffee_can_file, simple_options)
        self.coffee_can_pose = gymapi.Transform(p=gymapi.Vec3(1.7,3.0-0.4,1.125))

        coffe_can_file2 = "urdf/office/Coffee_can/model.urdf"
        self.coffee_can_asset2 = self.gym.load_asset(self.sim, self.asset_root, coffe_can_file2, simple_options)
        self.coffee_can_pose2 = gymapi.Transform(p=gymapi.Vec3(1.85,3.0-0.4,1.125),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        ink_cartridge_file1 = "urdf/office/Office_Depot_HP_61Tricolor_Ink_Cartridge/model.urdf"
        self.ink_cartridge_color_asset = self.gym.load_asset(self.sim, self.asset_root, ink_cartridge_file1, simple_options)
        self.ink_cartridge_color_pose1 = gymapi.Transform(p=gymapi.Vec3(2.265,3.0-0.3,1.125))
        self.ink_cartridge_color_pose2 = gymapi.Transform(p=gymapi.Vec3(2.265,3.0-0.35,1.125))
        self.ink_cartridge_color_pose3 = gymapi.Transform(p=gymapi.Vec3(2.265,3.0-0.4,1.125))

        ink_cartridge_file2 = "urdf/office/Office_Depot_HP_71_Remanufactured_Ink_Cartridge_Black/model.urdf"
        self.ink_cartridge_black_asset = self.gym.load_asset(self.sim, self.asset_root, ink_cartridge_file2, simple_options)
        self.ink_cartridge_black_pose1 = gymapi.Transform(p=gymapi.Vec3(2.35,3.0-0.3,1.125))
        self.ink_cartridge_black_pose2 = gymapi.Transform(p=gymapi.Vec3(2.35,3.0-0.35,1.125))
        self.ink_cartridge_black_pose3 = gymapi.Transform(p=gymapi.Vec3(2.35,3.0-0.4,1.125))

        mario_file = "urdf/office/Nintendo_Mario_Action_Figure/model.urdf"
        self.mario_asset = self.gym.load_asset(self.sim, self.asset_root, mario_file, simple_options)
        self.mario_pose = gymapi.Transform(p=gymapi.Vec3(1.75,3.0-0.42,1.542))
        
        yoshi_file = "urdf/office/Nintendo_Yoshi_Action_Figure/model.urdf"
        self.yoshi_asset = self.gym.load_asset(self.sim, self.asset_root, yoshi_file, simple_options)
        self.yoshi_pose = gymapi.Transform(p=gymapi.Vec3(1.85,3.0-0.42,1.542))

        cabinet2_file = "urdf/office/MetalCabinet/model.urdf"
        self.cabinet2_asset = self.gym.load_asset(self.sim, self.asset_root, cabinet2_file, simple_options)
        self.cabinet2_pose = gymapi.Transform(p=gymapi.Vec3(0.275,1.55,0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))
        self.cabinet2_pose2 = gymapi.Transform(p=gymapi.Vec3(0.275,1.55-0.81,0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        p1, p2 =get_object_positions(5, [3.2,3.65], [1.5,2.25], 0.15)

        tape0_file = "urdf/office/Tape0/model.urdf"
        self.tape0_asset = self.gym.load_asset(self.sim, self.asset_root, tape0_file, gravity_options)
        self.tape0_pose = gymapi.Transform(p=gymapi.Vec3(p1[0],p2[0],0.76))

        tape1_file = "urdf/office/Tape1/model.urdf"
        self.tape1_asset = self.gym.load_asset(self.sim, self.asset_root, tape1_file, gravity_options)
        self.tape1_pose = gymapi.Transform(p=gymapi.Vec3(p1[1],p2[1],0.76))

        tape2_file = "urdf/office/Tape2/model.urdf"
        self.tape2_asset = self.gym.load_asset(self.sim, self.asset_root, tape2_file, gravity_options)
        self.tape2_pose = gymapi.Transform(p=gymapi.Vec3(p1[2],p2[2],0.76))

        stapler_file = "urdf/office/07_stapler_0/model.urdf"
        self.stapler_asset = self.gym.load_asset(self.sim, self.asset_root, stapler_file, gravity_options)
        self.stapler_pose = gymapi.Transform(p=gymapi.Vec3(p1[3],p2[3],0.76),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/6))

        marker_file = "urdf/office/040_large_marker/model.urdf"
        self.marker_asset = self.gym.load_asset(self.sim, self.asset_root, marker_file, gravity_options)
        self.marker_pose = gymapi.Transform(p=gymapi.Vec3(p1[4],p2[4],0.76),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/6))

    def set_object_asset(self):
        print("object asset is set")
    
    def create_robot(self, env, num_env):
        print("creating robot")

    def create_assets(self, env, num_env):
        self.gym.create_actor(env, self.floor_asset, self.floor_pose1, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose2, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose3, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose4, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose5, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose6, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose7, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose8, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose9, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose10, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose11, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose12, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose13, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose14, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose15, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose16, "floor", num_env, 0)

        wall1_handle = self.gym.create_actor(env, self.wall_asset_1, self.wall1_pose, "wall1", num_env, 0)
        wall2_handle = self.gym.create_actor(env, self.wall_asset_2, self.wall2_pose, "wall2", num_env, 0)
        wall3_handle = self.gym.create_actor(env, self.wall_asset_1, self.wall3_pose, "wall3", num_env, 0)
        self.gym.set_rigid_body_color(env, wall1_handle, 0, gymapi.MESH_VISUAL, self.wall_color)
        self.gym.set_rigid_body_color(env, wall2_handle, 0, gymapi.MESH_VISUAL, self.wall_color)
        self.gym.set_rigid_body_color(env, wall3_handle, 0, gymapi.MESH_VISUAL, self.wall_color)
        
        desk_handle = self.gym.create_actor(env, self.desk_asset, self.desk_pose, "desk", num_env, 0)
        desk2_handle = self.gym.create_actor(env, self.desk_asset, self.desk_pose2, "desk2", num_env, 0)
        
        cabinet_handle = self.gym.create_actor(env, self.cabinet_asset,self.cabinet_pose, "cabinet", num_env, 0)
        shelf_handle = self.gym.create_actor(env, self.shelf_asset,self.shelf_pose, "shelf", num_env, 0)
        cabinet2_handle = self.gym.create_actor(env, self.cabinet_asset,self.cabinet_pose2, "cabinet", num_env, 0)
        self.gym.set_actor_scale(env, shelf_handle, 1.15)
        chair_handle = self.gym.create_actor(env, self.chair_asset, self.chair_pose, "chair", num_env, 0)
        laptop_handle = self.gym.create_actor(env, self.laptop_asset, self.laptop_pose, "laptop", num_env, 0)
        black_mouse_handle = self.gym.create_actor(env, self.black_mouse_asset, self.black_mouse_pose, "black_mouse", num_env, 0)
        cup_handle = self.gym.create_actor(env, self.cup_asset, self.cup_pose, "cup", num_env, 0)
        cup_handle2 = self.gym.create_actor(env, self.cup_asset, self.cup_pose2, "cup2", num_env, 0)
        power_strip_handle = self.gym.create_actor(env, self.power_strip_asset, self.power_strip_pose, "power", num_env, 0)
        stapler_handle = self.gym.create_actor(env, self.stapler_asset, self.stapler_pose, "stapler", num_env, 0)
        pencil_sharpener_handle = self.gym.create_actor(env, self.pencil_sharpener_asset, self.pencil_sharpener_pose, "pencil_sharpner", num_env, 0)
        marker_handle = self.gym.create_actor(env, self.marker_asset, self.marker_pose, "marker", num_env, 0)
        coffeemaker_handle = self.gym.create_actor(env, self.coffeemaker_asset, self.coffeemaker_pose, "coffeemaker", num_env, 0)
        coffee_can_handle = self.gym.create_actor(env, self.coffee_can_asset, self.coffee_can_pose, "coffee_can", num_env, 0)
        coffee_can_handle2 = self.gym.create_actor(env, self.coffee_can_asset2, self.coffee_can_pose2, "coffee_can2", num_env, 0)
        ink_cartridge_color_handle = self.gym.create_actor(env, self.ink_cartridge_color_asset, self.ink_cartridge_color_pose1, "color_cartridge", num_env, 0)
        ink_cartridge_color_handle2 = self.gym.create_actor(env, self.ink_cartridge_color_asset, self.ink_cartridge_color_pose2, "color_cartridge2", num_env, 0)
        ink_cartridge_color_handle3 = self.gym.create_actor(env, self.ink_cartridge_color_asset, self.ink_cartridge_color_pose3, "color_cartridge3", num_env, 0)
        ink_cartridge_black_handle = self.gym.create_actor(env, self.ink_cartridge_black_asset, self.ink_cartridge_black_pose1, "ink_cartridge1", num_env, 0)
        ink_cartridge_black_handle2 = self.gym.create_actor(env, self.ink_cartridge_black_asset, self.ink_cartridge_black_pose2, "ink_cartridge2", num_env, 0)
        ink_cartridge_black_handle3 = self.gym.create_actor(env, self.ink_cartridge_black_asset, self.ink_cartridge_black_pose3, "ink_cartridge3", num_env, 0)
        mario_handle = self.gym.create_actor(env, self.mario_asset, self.mario_pose, "mario", num_env, 0)
        yoshi_handle = self.gym.create_actor(env, self.yoshi_asset, self.yoshi_pose, "yoshi", num_env, 0)
        cabinet3_handle = self.gym.create_actor(env, self.cabinet2_asset, self.cabinet2_pose, "cabinet3", num_env, 0)
        cabinet4_handle = self.gym.create_actor(env, self.cabinet2_asset, self.cabinet2_pose2, "cabinet4", num_env, 0)
        tape0_handle = self.gym.create_actor(env, self.tape0_asset, self.tape0_pose, "tape0", num_env, 0)
        tape1_handle = self.gym.create_actor(env, self.tape1_asset, self.tape1_pose, "tape1", num_env, 0)
        tape2_handle = self.gym.create_actor(env, self.tape2_asset, self.tape2_pose, "tape2", num_env, 0)

class Env5(EnvBase):
    def set_robot_asset(self):
        print("UR5e is not ready")

    def set_wall_type(self, wall_type):
        if wall_type == "ivory":
            self.wall_color = gymapi.Vec3(220/255, 205/255, 152/255)
        elif wall_type == "white":
            self.wall_color = gymapi.Vec3(1, 1, 1)
        elif wall_type == "green":
            self.wall_color = gymapi.Vec3(24/255, 70/255, 50/255)

    def set_interior_asset(self):
        simple_options = gymapi.AssetOptions()
        simple_options.fix_base_link = True

        gravity_options = gymapi.AssetOptions()

        complex_options = gymapi.AssetOptions()
        complex_options.fix_base_link = True
        complex_options.vhacd_enabled = True
        complex_options.vhacd_params = gymapi.VhacdParams()
        complex_options.vhacd_params.resolution = 30000
        complex_options.vhacd_params.max_convex_hulls = 512
        complex_options.vhacd_params.convex_hull_approximation = False

        floor_file = "urdf/floor/wood2.urdf"
        self.floor_asset = self.gym.load_asset(self.sim, self.asset_root, floor_file, simple_options)
        self.floor_pose1 = gymapi.Transform(p=gymapi.Vec3(0.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose2 = gymapi.Transform(p=gymapi.Vec3(1.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose3 = gymapi.Transform(p=gymapi.Vec3(2.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose4 = gymapi.Transform(p=gymapi.Vec3(0.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose5 = gymapi.Transform(p=gymapi.Vec3(1.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose6 = gymapi.Transform(p=gymapi.Vec3(2.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose7 = gymapi.Transform(p=gymapi.Vec3(0.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose8 = gymapi.Transform(p=gymapi.Vec3(1.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose9 = gymapi.Transform(p=gymapi.Vec3(2.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose10 = gymapi.Transform(p=gymapi.Vec3(0.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose11 = gymapi.Transform(p=gymapi.Vec3(1.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose12 = gymapi.Transform(p=gymapi.Vec3(2.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))

        self.wall_asset_1 = self.gym.create_box(self.sim,0.1,3.95,3.0,simple_options)
        self.wall1_pose = gymapi.Transform(p=gymapi.Vec3(0.0,0.975,1.5))
        self.wall3_pose = gymapi.Transform(p=gymapi.Vec3(3.0,0.975,1.5))
        self.wall_asset_2 = self.gym.create_box(self.sim,3.0,0.1,3.0,simple_options)
        self.wall2_pose = gymapi.Transform(p=gymapi.Vec3(1.5,3.0,1.5))

        desk_file = "urdf/office/AdjTable/model.urdf"
        self.desk_asset = self.gym.load_asset(self.sim, self.asset_root, desk_file, simple_options)
        self.desk_pose = gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(2.59,2.23,0.0))
        self.desk_pose2 = gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(2.59,0.79,0.0))
        
        cabinet_file = "urdf/office/MetalCabinet/model.urdf"
        self.cabinet_asset = self.gym.load_asset(self.sim, self.asset_root, cabinet_file, simple_options)
        self.cabinet_pose = gymapi.Transform(p=gymapi.Vec3(0.275,2.55,0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))
        self.cabinet_pose2 = gymapi.Transform(p=gymapi.Vec3(0.275,2.55-0.81,0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        chair_file = "urdf/office/OfficeChairBlue/model.urdf"
        self.chair_asset = self.gym.load_asset(self.sim, self.asset_root, chair_file, simple_options)
        self.chair_pose = gymapi.Transform(p=gymapi.Vec3(2.0,2.25,0.0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        desk_caddy_file = "urdf/office/Markings_Desk_Caddy/model.urdf"
        self.desk_caddy_asset = self.gym.load_asset(self.sim, self.asset_root, desk_caddy_file, complex_options)
        self.desk_caddy_pose = gymapi.Transform(p=gymapi.Vec3(2.85,2.71,0.72))
        self.desk_caddy_pose2 = gymapi.Transform(p=gymapi.Vec3(2.85,1.27,0.72))

        laptop_file ="urdf/office/Travel_Mate_P_series_Notebook/model.urdf"
        self.laptop_asset = self.gym.load_asset(self.sim, self.asset_root,laptop_file, simple_options)
        self.laptop_pose =gymapi.Transform(p=gymapi.Vec3(2.7-0.125,2.43-0.21,0.72),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        white_mouse_file ="urdf/office/Razer_Taipan_White_Ambidextrous_Gaming_Mouse/model.urdf"
        self.white_mouse_asset = self.gym.load_asset(self.sim, self.asset_root,white_mouse_file, simple_options)
        self.white_mouse_pose = gymapi.Transform(p=gymapi.Vec3(2.7-0.125,2.43-0.42,0.72),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        p1, p2 =get_object_positions(5, [2.4,2.8], [0.2,0.8], 0.15)

        wrench_file = "urdf/office/042_adjustable_wrench/model.urdf"
        self.wrench_asset = self.gym.load_asset(self.sim, self.asset_root, wrench_file, gravity_options)
        self.wrench_pose = gymapi.Transform(p=gymapi.Vec3(p1[0],p2[0],0.74),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        driver_file = "urdf/office/043_phillips_screwdriver/model.urdf"
        self.driver_asset = self.gym.load_asset(self.sim, self.asset_root, driver_file, gravity_options)
        self.driver_pose = gymapi.Transform(p=gymapi.Vec3(p1[1],p2[1],0.74),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))
        self.driver_pose2 = gymapi.Transform(p=gymapi.Vec3(2.85,1.17,0.85),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), -np.pi/2)*gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/6))

        flat_driver_file = "urdf/office/044_flat_screwdriver/model.urdf"
        self.flat_driver_asset = self.gym.load_asset(self.sim, self.asset_root, flat_driver_file, gravity_options)
        self.flat_driver_pose = gymapi.Transform(p=gymapi.Vec3(p1[2],p2[2],0.74),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))
        self.flat_driver_pose2 = gymapi.Transform(p=gymapi.Vec3(2.85,1.13,0.85),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), -np.pi/2)*gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/6))

        hammer_file = "urdf/office/Cole_Hardware_Hammer_Black/model.urdf"
        self.hammer_asset = self.gym.load_asset(self.sim, self.asset_root, hammer_file, gravity_options)
        self.hammer_pose = gymapi.Transform(p=gymapi.Vec3(p1[3],p2[3],0.74),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), -np.pi/2))

        clamp_file = "urdf/office/050_medium_clamp/model.urdf"
        self.clamp_asset = self.gym.load_asset(self.sim, self.asset_root, clamp_file, gravity_options)
        self.clamp_pose = gymapi.Transform(p=gymapi.Vec3(p1[4],p2[4],0.74))

        toolbox_file = "urdf/office/Toolbox/model.urdf"
        self.toolbox_asset = self.gym.load_asset(self.sim, self.asset_root, toolbox_file, simple_options)
        self.toolbox_pose = gymapi.Transform(p=gymapi.Vec3(2.7,1.2,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), np.pi/2))

        storage_file = "urdf/office/Curver_Storage_Bin_Black_Small/model.urdf"
        self.storage_asset = self.gym.load_asset(self.sim, self.asset_root, storage_file, gravity_options)
        self.storage_pose = gymapi.Transform(p=gymapi.Vec3(2.55,1.35,0.73))

    def set_object_asset(self):
        print("object asset is set")
    
    def create_robot(self, env, num_env):
        print("creating robot")

    def create_assets(self, env, num_env):
        self.gym.create_actor(env, self.floor_asset, self.floor_pose1, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose2, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose3, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose4, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose5, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose6, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose7, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose8, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose9, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose10, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose11, "floor", num_env, 0)
        self.gym.create_actor(env, self.floor_asset, self.floor_pose12, "floor", num_env, 0)

        wall1_handle = self.gym.create_actor(env, self.wall_asset_1, self.wall1_pose, "wall1", num_env, 0)
        wall2_handle = self.gym.create_actor(env, self.wall_asset_2, self.wall2_pose, "wall2", num_env, 0)
        wall3_handle = self.gym.create_actor(env, self.wall_asset_1, self.wall3_pose, "wall3", num_env, 0)
        self.gym.set_rigid_body_color(env, wall1_handle, 0, gymapi.MESH_VISUAL, self.wall_color)
        self.gym.set_rigid_body_color(env, wall2_handle, 0, gymapi.MESH_VISUAL, self.wall_color)
        self.gym.set_rigid_body_color(env, wall3_handle, 0, gymapi.MESH_VISUAL, self.wall_color)

        desk_handle = self.gym.create_actor(env, self.desk_asset, self.desk_pose, "desk", num_env, 0)
        desk2_handle = self.gym.create_actor(env, self.desk_asset, self.desk_pose2, "desk2", num_env, 0)
        self.gym.set_actor_scale(env, desk_handle, 0.9)
        self.gym.set_actor_scale(env, desk2_handle, 0.9)
        cabinet_handle = self.gym.create_actor(env, self.cabinet_asset, self.cabinet_pose, "cabinet", num_env, 0)
        cabinet_handle2 = self.gym.create_actor(env, self.cabinet_asset, self.cabinet_pose2, "cabinet2", num_env, 0)
        chair_handle = self.gym.create_actor(env, self.chair_asset, self.chair_pose, "chair", num_env, 0)
        desk_caddy_handle = self.gym.create_actor(env, self.desk_caddy_asset, self.desk_caddy_pose, "desk", num_env, 0)
        self.gym.set_actor_scale(env,desk_caddy_handle,1.5)
        desk_caddy_handle2 = self.gym.create_actor(env, self.desk_caddy_asset, self.desk_caddy_pose2, "desk", num_env, 0)
        self.gym.set_actor_scale(env,desk_caddy_handle2,1.5)
        laptop_handle = self.gym.create_actor(env, self.laptop_asset, self.laptop_pose, "laptop", num_env, 0)
        white_mouse_handle = self.gym.create_actor(env, self.white_mouse_asset, self.white_mouse_pose, "white_mouse", num_env, 0)
        wrench_handle = self.gym.create_actor(env,self.wrench_asset, self.wrench_pose, "wrench", num_env, 0)
        driver_handle = self.gym.create_actor(env,self.driver_asset, self.driver_pose, "driver", num_env, 0)
        driver_handle2 = self.gym.create_actor(env,self.driver_asset, self.driver_pose2, "driver2", num_env, 0)
        flat_driver_handle = self.gym.create_actor(env, self.flat_driver_asset, self.flat_driver_pose, "flat_driver", num_env, 0)
        flat_driver_handle2 = self.gym.create_actor(env, self.flat_driver_asset, self.flat_driver_pose2, "flat_driver2", num_env, 0)
        hammer_handle = self.gym.create_actor(env, self.hammer_asset, self.hammer_pose, "hammer", num_env, 0)
        clamp_handle = self.gym.create_actor(env, self.clamp_asset, self.clamp_pose, "clamp", num_env, 0)
        toolbox_handle = self.gym.create_actor(env, self.toolbox_asset, self.toolbox_pose, "toolbox", num_env, 0)
        storage_handle = self.gym.create_actor(env, self.storage_asset, self.storage_pose, "storage", num_env, 0)

class OfficeEnvManager:
    def __init__(self, asset_root, gym, sim, device):
        self.envs = {
            "env1": Env1(asset_root, gym, sim, device),
            "env2": Env2(asset_root, gym, sim, device),
            "env3": Env3(asset_root, gym, sim, device),
            "env4": Env4(asset_root, gym, sim, device),
            "env5": Env5(asset_root, gym, sim, device),
        }
        self.current_env = None

    def set_env(self, env_name):
        if env_name in self.envs:
            self.current_env = self.envs[env_name]
            print(f"Environment is set to {env_name}")
        else:
            print(f"Environment {env_name} is not supported")

    def set_current_robot_asset(self):
        self.current_env.set_robot_asset()

    def set_current_interior_asset(self):
        self.current_env.set_interior_asset()
    
    def set_current_object_asset(self):
        self.current_env.set_object_asset()

    def create_current_robot(self, env, num_env):
        self.current_env.create_robot(env, num_env)

    def create_current_assets(self, env, num_env):
        self.current_env.create_assets(env, num_env)

    def set_wall_type(self, wall_type):
        print(f"Wall type is set to {wall_type}")
        self.current_env.set_wall_type(wall_type)  