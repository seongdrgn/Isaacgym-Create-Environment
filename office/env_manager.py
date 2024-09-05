"""
date: 2024.08.27
author: Seongyong Kim
description: kitchen environment class
"""
import numpy as np
from isaacgym import gymapi
from isaacgym.torch_utils import *
import random

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

def get_object_positions(num_objects, x_lim, y_lim):
    x_min_diff = 0.01
    y_min_diff = 0.05

    x_values = []
    while len(x_values) < num_objects:
        x = random.uniform(x_lim[0], x_lim[1])
        if all(abs(x - x_val) >= x_min_diff for x_val in x_values):
            x_values.append(x)

    y_values = []
    while len(y_values) < num_objects:
        y = random.uniform(y_lim[0], y_lim[1])
        if all(abs(y - y_val) >= y_min_diff for y_val in y_values):
            y_values.append(y)
    
    return x_values, y_values

class Env1(EnvBase):
    def set_robot_asset(self):
        # config UR5e asset
        print("Setting")

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

        desk_file = "urdf/kitchen/Desk/model.urdf"
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
        self.black_mouse_pose = gymapi.Transform(p=gymapi.Vec3(2.7-0.125,2.43-0.42,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        shelf_file = "urdf/office/shelf/model.urdf"
        self.shelf_asset = self.gym.load_asset(self.sim, self.asset_root, shelf_file, simple_options)
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
        self.keyboard_asset = self.gym.load_asset(self.sim, self.asset_root, keyboard_file, simple_options)
        self.keyboard_pose = gymapi.Transform(p=gymapi.Vec3(0.55,2.0,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        power_strip_file = "urdf/office/18_Power_strip/model.urdf"
        self.power_strip_asset = self.gym.load_asset(self.sim, self.asset_root, power_strip_file, simple_options)
        self.power_strip_pose = gymapi.Transform(p=gymapi.Vec3(0.08,1.48,0.625),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        xylitol_file = "urdf/office/Xyli_Pure_Xylitol/model.urdf"
        self.xylitol_asset = self.gym.load_asset(self.sim, self.asset_root, xylitol_file, simple_options)
        self.xylitol_pose = gymapi.Transform(p=gymapi.Vec3(0.2,1.6,0.625),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        germanium_file = "urdf/office/Germanium_GE132/model.urdf"
        self.germanium_asset = self.gym.load_asset(self.sim, self.asset_root, germanium_file, simple_options)
        self.germanium_pose = gymapi.Transform(p=gymapi.Vec3(0.18,1.43,0.625),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        cdg_file = "urdf/office/DIM_CDG/model.urdf"
        self.cdg_asset = self.gym.load_asset(self.sim, self.asset_root, cdg_file, simple_options)
        self.cdg_pose = gymapi.Transform(p=gymapi.Vec3(0.18,1.5,0.625),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        stapler_file = "urdf/office/07_stapler_0/model.urdf"
        self.stapler_asset = self.gym.load_asset(self.sim, self.asset_root, stapler_file, simple_options)
        self.stapler_pose = gymapi.Transform(p=gymapi.Vec3(0.35,1.55,0.63),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/4))

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

class Env2(EnvBase):
    def set_robot_asset(self):
        print("Setting")

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

        # floor_file = "urdf/floor/wood1.urdf"
        # floor_file = "urdf/floor/wood2.urdf"
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

        desk_file = "urdf/kitchen/Desk/model_dark_wood.urdf"
        self.desk_asset = self.gym.load_asset(self.sim, self.asset_root, desk_file, simple_options)
        self.desk_pose = gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi),p=gymapi.Vec3(2.7-0.125,2.43-0.21,0.0))
        self.desk_pose2 = gymapi.Transform(p=gymapi.Vec3(0.42,2.43-0.21,0.0))

        sorter_blue_file ="urdf/office/Poppin_File_Sorter_Blue/model.urdf"
        self.sorter_blue_asset = self.gym.load_asset(self.sim, self.asset_root,sorter_blue_file, simple_options)
        self.sorter_blue_pose = gymapi.Transform(p=gymapi.Vec3(0.15,2.7,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        sorter_pink_file ="urdf/office/Poppin_File_Sorter_Pink/model.urdf"
        self.sorter_pink_asset = self.gym.load_asset(self.sim, self.asset_root,sorter_pink_file, simple_options)
        self.sorter_pink_pose = gymapi.Transform(p=gymapi.Vec3(2.85,2.7,0.825))

        laptop_file ="urdf/office/Lenovo_Yoga_2_11/model.urdf"
        self.laptop_asset = self.gym.load_asset(self.sim, self.asset_root,laptop_file, simple_options)
        self.laptop_pose = gymapi.Transform(p=gymapi.Vec3(0.375,2.43-0.11,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        laptop2_file ="urdf/office/Travel_Mate_P_series_Notebook/model.urdf"
        self.laptop2_asset = self.gym.load_asset(self.sim, self.asset_root,laptop2_file, simple_options)
        self.laptop2_pose =gymapi.Transform(p=gymapi.Vec3(2.7-0.125,2.43-0.21,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        white_mouse_file ="urdf/office/Razer_Taipan_White_Ambidextrous_Gaming_Mouse/model.urdf"
        self.white_mouse_asset = self.gym.load_asset(self.sim, self.asset_root,white_mouse_file, simple_options)
        self.white_mouse_pose = gymapi.Transform(p=gymapi.Vec3(0.375,2.53,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        black_mouse_file ="urdf/office/Razer_Taipan_Black_Ambidextrous_Gaming_Mouse/model.urdf"
        self.black_mouse_asset = self.gym.load_asset(self.sim, self.asset_root,black_mouse_file, simple_options)
        self.black_mouse_pose = gymapi.Transform(p=gymapi.Vec3(2.7-0.125,2.43-0.42,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))
        
        chair_file = "urdf/office/OfficeChairGrey/model.urdf"
        self.chair_asset = self.gym.load_asset(self.sim, self.asset_root, chair_file, simple_options)
        self.chair_pose = gymapi.Transform(p=gymapi.Vec3(2.2,2.3,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        cabinet_file = "urdf/office/WhiteCabinet/model.urdf"
        self.cabinet_asset = self.gym.load_asset(self.sim, self.asset_root, cabinet_file, simple_options)
        self.cabinet_pose = gymapi.Transform(p=gymapi.Vec3(0.275,1.138,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))
        self.cabinet_pose2 = gymapi.Transform(p=gymapi.Vec3(2.725,1.138,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        book_shelf_file = "urdf/office/17_Book_shelf/model.urdf"
        self.book_shelf_asset = self.gym.load_asset(self.sim, self.asset_root, book_shelf_file, simple_options)
        self.book_shelf_pose = gymapi.Transform(p=gymapi.Vec3(0.21,1.66,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        mug_file = "urdf/office/Room_Essentials_Mug_White_Yellow/model.urdf"
        self.mug_asset = self.gym.load_asset(self.sim, self.asset_root, mug_file, simple_options)
        self.mug_pose = gymapi.Transform(p=gymapi.Vec3(0.34,1.95,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/4))

        pencil_sharpener_file = "urdf/office/03_Pencil_sharpener/model.urdf"
        self.pencil_sharpener_asset = self.gym.load_asset(self.sim, self.asset_root, pencil_sharpener_file, simple_options)
        self.pencil_sharpener_pose = gymapi.Transform(p=gymapi.Vec3(0.1,1.55,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        flower_pot_file = "urdf/office/20_Flower_pot/model.urdf"
        self.flower_pot_asset = self.gym.load_asset(self.sim, self.asset_root, flower_pot_file, simple_options)
        self.flower_pot_pose = gymapi.Transform(p=gymapi.Vec3(0.12,0.8,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))
        self.flower_pot_pose2 = gymapi.Transform(p=gymapi.Vec3(0.12,0.9,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))
        self.flower_pot_pose3 = gymapi.Transform(p=gymapi.Vec3(0.12,1.0,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))
        
        camera_file = "urdf/office/24_Camera/model.urdf"
        self.camera_asset = self.gym.load_asset(self.sim, self.asset_root, camera_file, simple_options)
        self.camera_pose = gymapi.Transform(p=gymapi.Vec3(0.12,1.4,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        ink_black_file = "urdf/office/Canon_Pixma_Ink_Cartridge_8/model.urdf"
        self.ink_black_asset = self.gym.load_asset(self.sim, self.asset_root, ink_black_file, simple_options)
        self.ink_black_pose = gymapi.Transform(p=gymapi.Vec3(0.10,1.15,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        ink_green_file = "urdf/office/Canon_Pixma_Ink_Cartridge_8_Green/model.urdf"
        self.ink_green_asset = self.gym.load_asset(self.sim, self.asset_root, ink_green_file, simple_options)
        self.ink_green_pose = gymapi.Transform(p=gymapi.Vec3(0.10,1.25,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        ink_red_file = "urdf/office/Canon_Pixma_Ink_Cartridge_8_Red/model.urdf"
        self.ink_red_asset = self.gym.load_asset(self.sim, self.asset_root, ink_red_file, simple_options)
        self.ink_red_pose = gymapi.Transform(p=gymapi.Vec3(0.15,1.15,1.08),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

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
        ink_black_handle = self.gym.create_actor(env, self.ink_black_asset, self.ink_black_pose, "blackink", num_env, 0)
        ink_green_handle = self.gym.create_actor(env, self.ink_green_asset, self.ink_green_pose, "greenink", num_env, 0)
        ink_red_handle = self.gym.create_actor(env, self.ink_red_asset, self.ink_red_pose, "redink", num_env, 0)

class Env3(EnvBase):
    def set_robot_asset(self):
        print("Setting")

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

        # floor_file = "urdf/floor/wood1.urdf"
        # floor_file = "urdf/floor/wood2.urdf"
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

        desk_file = "urdf/kitchen/Desk/model.urdf"
        self.desk_asset = self.gym.load_asset(self.sim, self.asset_root, desk_file, simple_options)
        self.desk_pose = gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi),p=gymapi.Vec3(2.7-0.125,2.43-1.015,0.0))
        self.desk_pose2 = gymapi.Transform(p=gymapi.Vec3(0.68,2.575,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        sorter_pink_file ="urdf/office/Poppin_File_Sorter_Pink/model.urdf"
        self.sorter_pink_asset = self.gym.load_asset(self.sim, self.asset_root,sorter_pink_file, simple_options)
        self.sorter_pink_pose = gymapi.Transform(p=gymapi.Vec3(2.85,1.9,0.825))

        laptop_file ="urdf/office/Travel_Mate_P_series_Notebook/model.urdf"
        self.laptop_asset = self.gym.load_asset(self.sim, self.asset_root,laptop_file, simple_options)
        self.laptop_pose =gymapi.Transform(p=gymapi.Vec3(2.7-0.125,2.43-1.01,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        black_mouse_file ="urdf/office/Razer_Taipan_Black_Ambidextrous_Gaming_Mouse/model.urdf"
        self.black_mouse_asset = self.gym.load_asset(self.sim, self.asset_root,black_mouse_file, simple_options)
        self.black_mouse_pose = gymapi.Transform(p=gymapi.Vec3(2.7-0.125,2.43-1.22,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        cabinet_file = "urdf/office/WhiteCabinet/model.urdf"
        self.cabinet_asset = self.gym.load_asset(self.sim, self.asset_root, cabinet_file, simple_options)
        self.cabinet_pose = gymapi.Transform(p=gymapi.Vec3(2.725,2.5,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        chair_file = "urdf/office/OfficeChairGrey/model.urdf"
        self.chair_asset = self.gym.load_asset(self.sim, self.asset_root, chair_file, simple_options)
        self.chair_pose = gymapi.Transform(p=gymapi.Vec3(2.2,1.4,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        trashbin_file = "urdf/office/TrashBin/model.urdf"
        self.trashbin_asset = self.gym.load_asset(self.sim, self.asset_root, trashbin_file, simple_options)
        self.trashbin_pose = gymapi.Transform(p=gymapi.Vec3(1.42,2.525,0.0))

        book_shelf_file = "urdf/office/17_Book_shelf/model.urdf"
        self.book_shelf_asset = self.gym.load_asset(self.sim, self.asset_root, book_shelf_file, simple_options)
        self.book_shelf_pose = gymapi.Transform(p=gymapi.Vec3(0.11,2.78,0.825))

        osilloscope_file = "urdf/office/00_Oscilloscope/model.urdf"
        self.osilloscope_asset = self.gym.load_asset(self.sim, self.asset_root, osilloscope_file, simple_options)
        self.osilloscope_pose = gymapi.Transform(p=gymapi.Vec3(0.4,2.8,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/6))

        gopro_file = "urdf/office/25_Gopro/model.urdf"
        self.gopro_asset = self.gym.load_asset(self.sim, self.asset_root, gopro_file, simple_options)
        self.gopro_pose = gymapi.Transform(p=gymapi.Vec3(1.0,2.75,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/6))

        camera_file = "urdf/office/24_Camera/model.urdf"
        self.camera_asset = self.gym.load_asset(self.sim, self.asset_root, camera_file, simple_options)
        self.camera_pose = gymapi.Transform(p=gymapi.Vec3(0.75,2.5,0.825))

        lens_file = "urdf/office/Nikon_1_AW1_w11275mm_Lens_Silver/model.urdf"
        self.lens_asset = self.gym.load_asset(self.sim, self.asset_root, lens_file, simple_options)
        self.lens_pose = gymapi.Transform(p=gymapi.Vec3(0.6,2.4,0.825))

        driver_file = "urdf/office/043_phillips_screwdriver/model.urdf"
        self.driver_asset = self.gym.load_asset(self.sim, self.asset_root, driver_file, simple_options)
        self.driver_pose = gymapi.Transform(p=gymapi.Vec3(0.9,2.45,0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/6))

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

        cabinet_handle = self.gym.create_actor(env, self.cabinet_asset, self.cabinet_pose, "cabinet", num_env, 0)
        chair_handle = self.gym.create_actor(env, self.chair_asset, self.chair_pose, "chair", num_env, 0)
        trashbin_handle = self.gym.create_actor(env, self.trashbin_asset, self.trashbin_pose, "trashbin", num_env, 0)
        book_shelf_handle = self.gym.create_actor(env, self.book_shelf_asset, self.book_shelf_pose, "bookshelf", num_env, 0)
        osilloscope_handle = self.gym.create_actor(env, self.osilloscope_asset, self.osilloscope_pose, "osilloscope", num_env, 0)
        self.gym.set_actor_scale(env, osilloscope_handle, 5)
        gopro_handle = self.gym.create_actor(env, self.gopro_asset, self.gopro_pose, "gopro", num_env, 0)
        self.gym.set_actor_scale(env, gopro_handle, 0.3175)
        camera_handle = self.gym.create_actor(env, self.camera_asset, self.camera_pose, "camera", num_env, 0)
        lens_handle = self.gym.create_actor(env, self.lens_asset, self.lens_pose, "lens", num_env, 0)
        driver_handle = self.gym.create_actor(env, self.driver_asset, self.driver_pose, "driver", num_env, 0)

class Env4(EnvBase):
    def set_robot_asset(self):
        print("Setting")

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

        # floor_file = "urdf/floor/wood1.urdf"
        # floor_file = "urdf/floor/wood2.urdf"
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

        table_options = gymapi.AssetOptions()
        table_options.fix_base_link = True
        table_options.vhacd_enabled = True
        table_options.vhacd_params.resolution = 300000

        desk_file = "urdf/office/SmallCubicle/model.urdf"
        self.desk_asset = self.gym.load_asset(self.sim, self.asset_root, desk_file, table_options)
        self.desk_pose = gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(2.525,2.275,0.0))
        self.desk_pose2 = gymapi.Transform(p=gymapi.Vec3(2.525,0.925,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        shelf_file = "urdf/office/shelf/model.urdf"
        self.shelf_asset = self.gym.load_asset(self.sim, self.asset_root, shelf_file, simple_options)
        self.shelf_pose = gymapi.Transform(p=gymapi.Vec3(0.27,2.5,1.08), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))
        self.shelf_pose2 = gymapi.Transform(p=gymapi.Vec3(0.27,1.6,1.08), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))
        
        cabinet_file = "urdf/office/WhiteCabinet/model.urdf"
        self.cabinet_asset = self.gym.load_asset(self.sim, self.asset_root, cabinet_file, simple_options)
        self.cabinet_pose = gymapi.Transform(p=gymapi.Vec3(0.275,2.5,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))
        self.cabinet_pose2 = gymapi.Transform(p=gymapi.Vec3(0.275,1.6,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        chair_file = "urdf/office/OfficeChairBlack/model.urdf"
        self.chair_asset = self.gym.load_asset(self.sim, self.asset_root, chair_file, simple_options)
        self.chair_pose = gymapi.Transform(p=gymapi.Vec3(2.0,2.3,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        laptop_file ="urdf/office/Travel_Mate_P_series_Notebook/model.urdf"
        self.laptop_asset = self.gym.load_asset(self.sim, self.asset_root,laptop_file, simple_options)
        self.laptop_pose =gymapi.Transform(p=gymapi.Vec3(2.6,2.3,0.745),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))
        
        black_mouse_file ="urdf/office/Razer_Taipan_Black_Ambidextrous_Gaming_Mouse/model.urdf"
        self.black_mouse_asset = self.gym.load_asset(self.sim, self.asset_root,black_mouse_file, simple_options)
        self.black_mouse_pose = gymapi.Transform(p=gymapi.Vec3(2.6,2.09,0.745),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))
        
        cup_file = "urdf/office/PlasticCup/model.urdf"
        self.cup_asset = self.gym.load_asset(self.sim, self.asset_root, cup_file, simple_options)
        self.cup_pose = gymapi.Transform(p=gymapi.Vec3(2.65,2.6,0.82),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))
        self.cup_pose2 = gymapi.Transform(p=gymapi.Vec3(2.8,0.6,0.82),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        power_strip_file = "urdf/office/18_Power_strip/model.urdf"
        self.power_strip_asset = self.gym.load_asset(self.sim, self.asset_root, power_strip_file, simple_options)
        self.power_strip_pose = gymapi.Transform(p=gymapi.Vec3(2.85,1.35,0.76),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        stapler_file = "urdf/office/07_stapler_0/model.urdf"
        self.stapler_asset = self.gym.load_asset(self.sim, self.asset_root, stapler_file, simple_options)
        self.stapler_pose = gymapi.Transform(p=gymapi.Vec3(2.65,1.4,0.76))

        pencil_sharpener_file = "urdf/office/03_Pencil_sharpener/model.urdf"
        self.pencil_sharpener_asset = self.gym.load_asset(self.sim, self.asset_root, pencil_sharpener_file, simple_options)
        self.pencil_sharpener_pose = gymapi.Transform(p=gymapi.Vec3(2.8,0.5,0.76),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        marker_file = "urdf/office/040_large_marker/model.urdf"
        self.marker_asset = self.gym.load_asset(self.sim, self.asset_root, marker_file, simple_options)
        self.marker_pose = gymapi.Transform(p=gymapi.Vec3(2.8,0.75,0.76))

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
        
        cabinet_handle = self.gym.create_actor(env, self.cabinet_asset,self.cabinet_pose, "cabinet", num_env, 0)
        shelf_handle = self.gym.create_actor(env, self.shelf_asset,self.shelf_pose, "shelf", num_env, 0)
        cabinet2_handle = self.gym.create_actor(env, self.cabinet_asset,self.cabinet_pose2, "cabinet", num_env, 0)
        shelf2_handle = self.gym.create_actor(env, self.shelf_asset,self.shelf_pose2, "shelf", num_env, 0)
        self.gym.set_actor_scale(env, shelf_handle, 1.15)
        self.gym.set_actor_scale(env, shelf2_handle, 1.15)
        chair_handle = self.gym.create_actor(env, self.chair_asset, self.chair_pose, "chair", num_env, 0)
        laptop_handle = self.gym.create_actor(env, self.laptop_asset, self.laptop_pose, "laptop", num_env, 0)
        black_mouse_handle = self.gym.create_actor(env, self.black_mouse_asset, self.black_mouse_pose, "black_mouse", num_env, 0)
        cup_handle = self.gym.create_actor(env, self.cup_asset, self.cup_pose, "cup", num_env, 0)
        cup_handle2 = self.gym.create_actor(env, self.cup_asset, self.cup_pose2, "cup2", num_env, 0)
        power_strip_handle = self.gym.create_actor(env, self.power_strip_asset, self.power_strip_pose, "power", num_env, 0)
        stapler_handle = self.gym.create_actor(env, self.stapler_asset, self.stapler_pose, "stapler", num_env, 0)
        pencil_sharpener_handle = self.gym.create_actor(env, self.pencil_sharpener_asset, self.pencil_sharpener_pose, "pencil_sharpner", num_env, 0)
        marker_handle = self.gym.create_actor(env, self.marker_asset, self.marker_pose, "marker", num_env, 0)

class Env5(EnvBase):
    def set_robot_asset(self):
        print("Setting")

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

        # floor_file = "urdf/floor/wood1.urdf"
        # floor_file = "urdf/floor/wood2.urdf"
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

        tool_options = gymapi.AssetOptions()
        tool_options.fix_base_link = True
        tool_options.vhacd_enabled = True
        tool_options.vhacd_params.resolution = 300000

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
        self.desk_caddy_asset = self.gym.load_asset(self.sim, self.asset_root, desk_caddy_file, simple_options)
        self.desk_caddy_pose = gymapi.Transform(p=gymapi.Vec3(2.85,2.71,0.72))
        self.desk_caddy_pose2 = gymapi.Transform(p=gymapi.Vec3(2.85,1.27,0.72))

        laptop_file ="urdf/office/Travel_Mate_P_series_Notebook/model.urdf"
        self.laptop_asset = self.gym.load_asset(self.sim, self.asset_root,laptop_file, simple_options)
        self.laptop_pose =gymapi.Transform(p=gymapi.Vec3(2.7-0.125,2.43-0.21,0.72),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        white_mouse_file ="urdf/office/Razer_Taipan_White_Ambidextrous_Gaming_Mouse/model.urdf"
        self.white_mouse_asset = self.gym.load_asset(self.sim, self.asset_root,white_mouse_file, simple_options)
        self.white_mouse_pose = gymapi.Transform(p=gymapi.Vec3(2.7-0.125,2.43-0.42,0.72),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        wrench_file = "urdf/office/042_adjustable_wrench/model.urdf"
        self.wrench_asset = self.gym.load_asset(self.sim, self.asset_root, wrench_file, tool_options)
        self.wrench_pose = gymapi.Transform(p=gymapi.Vec3(2.7,0.35,0.74),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        driver_file = "urdf/office/043_phillips_screwdriver/model.urdf"
        self.driver_asset = self.gym.load_asset(self.sim, self.asset_root, driver_file, tool_options)
        self.driver_pose = gymapi.Transform(p=gymapi.Vec3(2.4,0.25,0.74),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        flat_driver_file = "urdf/office/044_flat_screwdriver/model.urdf"
        self.flat_driver_asset = self.gym.load_asset(self.sim, self.asset_root, flat_driver_file, tool_options)
        self.flat_driver_pose = gymapi.Transform(p=gymapi.Vec3(2.4,0.2,0.74),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2))

        hammer_file = "urdf/office/Cole_Hardware_Hammer_Black/model.urdf"
        self.hammer_asset = self.gym.load_asset(self.sim, self.asset_root, hammer_file, tool_options)
        self.hammer_pose = gymapi.Transform(p=gymapi.Vec3(2.7,0.2,0.74),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), -np.pi/2))

        clamp_file = "urdf/office/050_medium_clamp/model.urdf"
        self.clamp_asset = self.gym.load_asset(self.sim, self.asset_root, clamp_file, tool_options)
        self.clamp_pose = gymapi.Transform(p=gymapi.Vec3(2.8,0.5,0.74))

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
        flat_driver_handle = self.gym.create_actor(env, self.flat_driver_asset, self.flat_driver_pose, "flat_driver", num_env, 0)
        hammer_handle = self.gym.create_actor(env, self.hammer_asset, self.hammer_pose, "hammer", num_env, 0)
        clamp_handle = self.gym.create_actor(env, self.clamp_asset, self.clamp_pose, "clamp", num_env, 0)

class OfficeEnvManager:
    def __init__(self, asset_root, gym, sim, device):
        self.envs = {
            "env1": Env1(asset_root, gym, sim, device),
            "env2": Env2(asset_root, gym, sim, device),
            "env3": Env3(asset_root, gym, sim, device),
            "env4": Env4(asset_root, gym, sim, device),
            "env5": Env5(asset_root, gym, sim, device),
            # Add more environments as needed...
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
        self.current_env.set_wall_type(wall_type)    