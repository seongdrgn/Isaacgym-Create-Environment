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
        print("UR X")

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
        complex_options.vhacd_params.resolution = 3000000
        complex_options.vhacd_params.max_convex_hulls = 512
        complex_options.vhacd_params.convex_hull_approximation = False

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

        sofa_file = "urdf/living_room/sofa/model.urdf"
        self.sofa_asset = self.gym.load_asset(self.sim, self.asset_root, sofa_file, simple_options)
        self.sofa_pose = gymapi.Transform(p=gymapi.Vec3(2.55,1.3,0.0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi+np.pi/2))
        
        shelf_file = "urdf/living_room/shelf/model.urdf"
        self.shelf_asset = self.gym.load_asset(self.sim, self.asset_root, shelf_file, simple_options)
        
        LampAndStand_file = "urdf/living_room/LampAndStand/model.urdf"
        self.LampAndStand_asset = self.gym.load_asset(self.sim, self.asset_root, LampAndStand_file, simple_options)
        self.LampAndStand_pose = gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 2*np.pi),p=gymapi.Vec3(0.37,2.25,0))
        
        wood_block_file = "urdf/living_room/wood_block/model.urdf"
        self.wood_block_asset = self.gym.load_asset(self.sim, self.asset_root, wood_block_file, simple_options)
        
        soccer_ball_file = "urdf/living_room/soccer_ball/model.urdf"
        self.soccer_ball_asset = self.gym.load_asset(self.sim, self.asset_root, soccer_ball_file, simple_options)
        
        desk_file = "urdf/kitchen/DiningTableWood/model.urdf"
        self.desk_asset = self.gym.load_asset(self.sim, self.asset_root, desk_file, simple_options)
        self.desk_pose = gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0,0.0,1.0), -2*np.pi/2),p=gymapi.Vec3(1.5,1.1,0.0))
    
        mug_file = "urdf/living_room/mug/model.urdf"
        self.mug_asset = self.gym.load_asset(self.sim, self.asset_root, mug_file, simple_options)
        
        bowl_file = "urdf/living_room/bowl/model.urdf"
        self.bowl_asset = self.gym.load_asset(self.sim, self.asset_root, bowl_file, simple_options)
        
        Pokmon_X_Nintendo_3DS_Game_file = "urdf/living_room/Pokmon_X_Nintendo_3DS_Game/model.urdf"
        self.Pokmon_X_Nintendo_3DS_Game_asset = self.gym.load_asset(self.sim, self.asset_root, Pokmon_X_Nintendo_3DS_Game_file, simple_options)
        
        Pokmon_Y_Nintendo_3DS_Game_file = "urdf/living_room/Pokmon_Y_Nintendo_3DS_Game/model.urdf"
        self.Pokmon_Y_Nintendo_3DS_Game_asset = self.gym.load_asset(self.sim, self.asset_root, Pokmon_Y_Nintendo_3DS_Game_file, simple_options)
        
        Pokmon_Conquest_Nintendo_DS_Game_file = "urdf/living_room/Pokmon_Conquest_Nintendo_DS_Game/model.urdf"
        self.Pokmon_Conquest_Nintendo_DS_Game_asset = self.gym.load_asset(self.sim, self.asset_root, Pokmon_Conquest_Nintendo_DS_Game_file, simple_options)
        
        Super_Mario_3D_World_Deluxe_Set_file = "urdf/living_room/Super_Mario_3D_World_Deluxe_Set/model.urdf"
        self.Super_Mario_3D_World_Deluxe_Set_asset = self.gym.load_asset(self.sim, self.asset_root, Super_Mario_3D_World_Deluxe_Set_file, simple_options)
        
        Headset_file = "urdf/living_room/09_Headset/model.urdf"
        self.Headset_file_asset = self.gym.load_asset(self.sim, self.asset_root, Headset_file, simple_options)
        
        Racoon_file = "urdf/living_room/Racoon/model.urdf"
        self.Racoon_asset = self.gym.load_asset(self.sim, self.asset_root, Racoon_file, simple_options)
        
        Dino_3_file = "urdf/living_room/Dino_3/model.urdf"
        self.Dino_3_asset = self.gym.load_asset(self.sim, self.asset_root, Dino_3_file, simple_options)
        
        Dino_4_file = "urdf/living_room/Dino_4/model.urdf"
        self.Dino_4_asset = self.gym.load_asset(self.sim, self.asset_root, Dino_4_file, simple_options)


    def set_object_asset(self):
        print("object asset is set")

    def create_robot(self, env, num_env):
        print("x")

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

        sofa_handle = self.gym.create_actor(env, self.sofa_asset, self.sofa_pose, "sofa", num_env, 0)
        LampAndStand_handle = self.gym.create_actor(env, self.LampAndStand_asset, self.LampAndStand_pose, "LampAndStand", num_env, 0)
        desk_handle = self.gym.create_actor(env, self.desk_asset, self.desk_pose, "desk", num_env, 0)
        self.gym.set_rigid_body_color(env,desk_handle,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.05,0.02,0.02))
        self.gym.set_actor_scale(env, desk_handle, 0.5)
        shelf_actor = self.gym.create_actor(env, self.shelf_asset, gymapi.Transform(p=gymapi.Vec3(1.41,2.4,1.5)),'shelf',num_env,0)
        Pokmon_X_Nintendo_3DS_Game_actor = self.gym.create_actor(env, self.Pokmon_X_Nintendo_3DS_Game_asset, gymapi.Transform(p=gymapi.Vec3(1.55,1.35,0.375),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/3)),'pokmon_x',num_env,0)
        Pokmon_Y_Nintendo_3DS_Game_actor = self.gym.create_actor(env, self.Pokmon_Y_Nintendo_3DS_Game_asset, gymapi.Transform(p=gymapi.Vec3(1.45,1.15,0.375),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/5)),'pokmon_y',num_env,0)
        Pokmon_Conquest_Nintendo_DS_Game_actor = self.gym.create_actor(env, self.Pokmon_Conquest_Nintendo_DS_Game_asset, gymapi.Transform(p=gymapi.Vec3(1.45,1.55,0.375),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi)),'Pokmon_Conquest',num_env,0)
        Super_Mario_3D_World_Deluxe_Set_actor = self.gym.create_actor(env, self.Super_Mario_3D_World_Deluxe_Set_asset, gymapi.Transform(p=gymapi.Vec3(1.45,0.88,0.38),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi)),'Super_Mario_3D',num_env,0)
        Headset_actor = self.gym.create_actor(env, self.Headset_file_asset, gymapi.Transform(p=gymapi.Vec3(1.65,0.65,0.38),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2)),'Headset_file',num_env,0)
        Racoon_actor = self.gym.create_actor(env, self.Racoon_asset, gymapi.Transform(p=gymapi.Vec3(2.55,1.65,0.385),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/3)),'Racoon',num_env,0)
        Dino_3_actor = self.gym.create_actor(env, self.Dino_3_asset, gymapi.Transform(p=gymapi.Vec3(2.45,1.47,0.4),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), np.pi/3)),'Dino_3',num_env,0)
        Dino_4_actor = self.gym.create_actor(env, self.Dino_4_asset, gymapi.Transform(p=gymapi.Vec3(2.35,1.75,0.42),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), np.pi/2)),'Dino_4',num_env,0)

        self.gym.set_actor_scale(env, shelf_actor,1.5)
        self.gym.set_rigid_body_color(env,shelf_actor,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.05,0.0,0.02))
        wood_block_actor = self.gym.create_actor(env, self.wood_block_asset, gymapi.Transform(p=gymapi.Vec3(1.75,2.25,2.2),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2)), 'wood_block', num_env, 0)
        wood_block_actor = self.gym.create_actor(env, self.wood_block_asset, gymapi.Transform(p=gymapi.Vec3(1.65,2.25,2.2),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2)), 'wood_block', num_env, 0)
        wood_block_actor = self.gym.create_actor(env, self.wood_block_asset, gymapi.Transform(p=gymapi.Vec3(1.55,2.25,2.2),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2)), 'wood_block', num_env, 0)
        mug_actor = self.gym.create_actor(env, self.mug_asset, gymapi.Transform(p=gymapi.Vec3(1.55,2.2,1.60),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2)), 'mug', num_env, 0)
        self.gym.set_rigid_body_color(env,mug_actor,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.1,0.2,0.2))
        mug_actor2 = self.gym.create_actor(env, self.mug_asset, gymapi.Transform(p=gymapi.Vec3(1.75,2.2,1.60),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2)), 'mug', num_env, 0)
        self.gym.set_rigid_body_color(env,mug_actor2,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.2,0.2,0.2))
        bowl_actor = self.gym.create_actor(env, self.bowl_asset, gymapi.Transform(p=gymapi.Vec3(1.15,2.3,2.15),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2)), 'bowl', num_env, 0)
        self.gym.set_actor_scale(env, bowl_actor,1.5)
        self.gym.set_rigid_body_color(env,bowl_actor,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.9,0.92,0.92))
        soccer_ball_actor = self.gym.create_actor(env, self.soccer_ball_asset, gymapi.Transform(p=gymapi.Vec3(1.05,2.2,1.65),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2)), 'soccer_ball', num_env, 0)
        self.gym.set_actor_scale(env, soccer_ball_actor,1.5)
        soccer_ball_actor = self.gym.create_actor(env, self.soccer_ball_asset, gymapi.Transform(p=gymapi.Vec3(1.25,2.2,1.65),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2)), 'soccer_ball', num_env, 0)
        self.gym.set_actor_scale(env, soccer_ball_actor,1.5)

class Env2(EnvBase):
    def set_robot_asset(self):
        print("UR X")

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
        self.floor_pose1 = gymapi.Transform(p=gymapi.Vec3(0.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose2 = gymapi.Transform(p=gymapi.Vec3(1.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose3 = gymapi.Transform(p=gymapi.Vec3(2.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose4 = gymapi.Transform(p=gymapi.Vec3(0.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose5 = gymapi.Transform(p=gymapi.Vec3(1.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose6 = gymapi.Transform(p=gymapi.Vec3(2.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose7 = gymapi.Transform(p=gymapi.Vec3(0.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose8 = gymapi.Transform(p=gymapi.Vec3(1.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose9 = gymapi.Transform(p=gymapi.Vec3(2.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose10 = gymapi.Transform(p=gymapi.Vec3(0.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose11 = gymapi.Transform(p=gymapi.Vec3(1.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose12 = gymapi.Transform(p=gymapi.Vec3(2.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose13 = gymapi.Transform(p=gymapi.Vec3(3.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose14 = gymapi.Transform(p=gymapi.Vec3(3.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose15 = gymapi.Transform(p=gymapi.Vec3(3.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose16 = gymapi.Transform(p=gymapi.Vec3(3.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))

        self.wall_asset_1 = self.gym.create_box(self.sim,0.1,4.0,3.0,simple_options)
        self.wall1_pose = gymapi.Transform(p=gymapi.Vec3(-0.5,1.0,1.5))
        self.wall3_pose = gymapi.Transform(p=gymapi.Vec3(3.5,1.0,1.5))
        self.wall_asset_2 = self.gym.create_box(self.sim,4.0,0.1,3.0,simple_options)
        self.wall2_pose = gymapi.Transform(p=gymapi.Vec3(1.5,3.0,1.5))
    
        sofa_file = "urdf/living_room/sofa/model.urdf"
        self.sofa_asset = self.gym.load_asset(self.sim, self.asset_root, sofa_file, simple_options)
        
        mini_sofa_file = "urdf/living_room/minisofa/model.urdf"
        self.mini_sofa_asset = self.gym.load_asset(self.sim, self.asset_root, mini_sofa_file, simple_options)
        
        tv_stand_file = "urdf/living_room/TVStand/model.urdf"
        self.tv_stand_asset = self.gym.load_asset(self.sim, self.asset_root, tv_stand_file, simple_options)
        
        shelf_file = "urdf/living_room/shelf/model.urdf"
        self.shelf_asset = self.gym.load_asset(self.sim, self.asset_root, shelf_file, simple_options)
        
        lamp_file = "urdf/living_room/LampAndStand/model.urdf"
        self.lamp_asset = self.gym.load_asset(self.sim, self.asset_root, lamp_file, simple_options)

        tv_file = "urdf/living_room/tv/model.urdf"
        self.tv_asset = self.gym.load_asset(self.sim, self.asset_root, tv_file, simple_options)
        
        CoQ10_BjTLbuRVt1t_file = "urdf/living_room/CoQ10_BjTLbuRVt1t/model.urdf"
        self.CoQ10_BjTLbuRVt1t_asset = self.gym.load_asset(self.sim, self.asset_root, CoQ10_BjTLbuRVt1t_file, simple_options)
        
        Marc_Anthony_True_Professional_Oil_of_Morocco_Argan_Oil_Treatment_file = "urdf/living_room/Marc_Anthony_True_Professional_Oil_of_Morocco_Argan_Oil_Treatment/model.urdf"
        self.Marc_Anthony_True_Professional_Oil_of_Morocco_Argan_Oil_Treatment_asset = self.gym.load_asset(self.sim, self.asset_root,Marc_Anthony_True_Professional_Oil_of_Morocco_Argan_Oil_Treatment_file, simple_options)
        
        Krill_Oil_file = "urdf/living_room/Krill_Oil/model.urdf"
        self.Krill_Oil_asset = self.gym.load_asset(self.sim, self.asset_root,Krill_Oil_file, simple_options)
        
        Timberland_Mens_Earthkeepers_Newmarket_6Inch_Cupsole_Boot_file = "urdf/living_room/Timberland_Mens_Earthkeepers_Newmarket_6Inch_Cupsole_Boot/model.urdf"
        self.Timberland_Mens_Earthkeepers_Newmarket_6Inch_Cupsole_Boot_asset = self.gym.load_asset(self.sim, self.asset_root,Timberland_Mens_Earthkeepers_Newmarket_6Inch_Cupsole_Boot_file, simple_options)
        
        Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack_file = "urdf/living_room/Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack/model.urdf"
        self.Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack_asset = self.gym.load_asset(self.sim, self.asset_root,Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack_file, simple_options)
        
        Phillips_Caplets_Size_24_file = "urdf/living_room/Phillips_Caplets_Size_24/model.urdf"
        self.Phillips_Caplets_Size_24_asset = self.gym.load_asset(self.sim, self.asset_root, Phillips_Caplets_Size_24_file, simple_options)
        
        Phillips_Colon_Health_Probiotic_Capsule_file = "urdf/living_room/Phillips_Colon_Health_Probiotic_Capsule/model.urdf"
        self.Phillips_Colon_Health_Probiotic_Capsule_asset = self.gym.load_asset(self.sim, self.asset_root, Phillips_Colon_Health_Probiotic_Capsule_file, simple_options)
        
        Phillips_Milk_of_Magnesia_Saline_Laxative_Liquid_Original_file = "urdf/living_room/Phillips_Milk_of_Magnesia_Saline_Laxative_Liquid_Original/model.urdf"
        self.Phillips_Milk_of_Magnesia_Saline_Laxative_Liquid_Original_asset = self.gym.load_asset(self.sim, self.asset_root, Phillips_Milk_of_Magnesia_Saline_Laxative_Liquid_Original_file, simple_options)
        
        YumYum_D3_Liquid_file = "urdf/living_room/YumYum_D3_Liquid/model.urdf"
        self.YumYum_D3_Liquid_asset = self.gym.load_asset(self.sim, self.asset_root, YumYum_D3_Liquid_file, simple_options)
        
        BLOCKS_file = "urdf/living_room/50_BLOCKS/model.urdf"
        self.BLOCKS_asset = self.gym.load_asset(self.sim, self.asset_root, BLOCKS_file, simple_options)
        
        Mattel_SKIP_BO_Card_Game_file = "urdf/living_room/Mattel_SKIP_BO_Card_Game/model.urdf"
        self.Mattel_SKIP_BO_Card_Game_asset = self.gym.load_asset(self.sim, self.asset_root,Mattel_SKIP_BO_Card_Game_file, simple_options)
        
        Pepsi_Cola_Wild_Cherry_Diet_file = "urdf/living_room/Pepsi_Cola_Wild_Cherry_Diet/model.urdf"
        self.Pepsi_Cola_Wild_Cherry_Diet_asset = self.gym.load_asset(self.sim, self.asset_root,Pepsi_Cola_Wild_Cherry_Diet_file, simple_options)
        
        Pepsi_Cola_Caffeine_Free_file = "urdf/living_room/Pepsi_Cola_Caffeine_Free/model.urdf"
        self.Pepsi_Cola_Caffeine_Free_asset = self.gym.load_asset(self.sim, self.asset_root,Pepsi_Cola_Caffeine_Free_file, simple_options)
        
        Quercetin_500_file = "urdf/living_room/Quercetin_500/model.urdf"
        self.Quercetin_500_asset = self.gym.load_asset(self.sim, self.asset_root,Quercetin_500_file, simple_options)
        
        Pet_Dophilus_powder_file = "urdf/living_room/Pet_Dophilus_powder/model.urdf"
        self.Pet_Dophilus_powder_asset = self.gym.load_asset(self.sim, self.asset_root,Pet_Dophilus_powder_file, simple_options)
        
        Playmates_Industrial_CoSplinter_Teenage_Mutant_Ninja_Turtle_Action_Figure_file = "urdf/living_room/Playmates_Industrial_CoSplinter_Teenage_Mutant_Ninja_Turtle_Action_Figure/model.urdf"
        self.Playmates_Industrial_CoSplinter_Teenage_Mutant_Ninja_Turtle_Action_Figure_asset = self.gym.load_asset(self.sim, self.asset_root,Playmates_Industrial_CoSplinter_Teenage_Mutant_Ninja_Turtle_Action_Figure_file, simple_options)

    def set_object_asset(self):
        print("object asset is set")
    
    def create_robot(self, env, num_env):
        print("x")

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

        sofa_actor = self.gym.create_actor(env, self.sofa_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2),p=gymapi.Vec3(0.1,1.8,0.0)),'sofa',num_env,0)
        lamp_actor = self.gym.create_actor(env, self.lamp_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(2.675,2.55,0.0)),'lamp',num_env,0)
        Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack_actor = self.gym.create_actor(env, self.Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/3),p=gymapi.Vec3(2.8,2.33,0.54)),'Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack',num_env,0)
        stand_actor = self.gym.create_actor(env, self.tv_stand_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.33,1.54,0.0)),'stand',num_env,0)
        Phillips_Caplets_Size_24_actor = self.gym.create_actor(env, self.Phillips_Caplets_Size_24_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.33,1.5,0.45)),'Phillips_Caplets_Size_24',num_env,0)
        Phillips_Colon_Health_Probiotic_Capsule_actor = self.gym.create_actor(env, self.Phillips_Colon_Health_Probiotic_Capsule_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.33,1.3,0.45)),'Phillips_Colon_Health_Probiotic_Capsule',num_env,0)
        Phillips_Milk_of_Magnesia_Saline_Laxative_Liquid_Original_actor = self.gym.create_actor(env, self.Phillips_Milk_of_Magnesia_Saline_Laxative_Liquid_Original_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.453,1.4,0.45)),'Phillips_Milk_of_Magnesia_Saline_Laxative_Liquid_Original',num_env,0)
        YumYum_D3_Liquid_actor = self.gym.create_actor(env, self.YumYum_D3_Liquid_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.43,1.3,0.45)),'YumYum_D3_Liquid',num_env,0)
        BLOCKS_actor = self.gym.create_actor(env, self.BLOCKS_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.24,1.55,0.25)),'BLOCKS',num_env,0)
        Mattel_SKIP_BO_Card_Game_actor = self.gym.create_actor(env, self.Mattel_SKIP_BO_Card_Game_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 1.0), -np.pi),p=gymapi.Vec3(3.24,1.65,0.06)),'Mattel_SKIP_BO_Card_Game',num_env,0)
        Pepsi_Cola_Wild_Cherry_Diet_actor = self.gym.create_actor(env, self.Pepsi_Cola_Wild_Cherry_Diet_asset, gymapi.Transform(p=gymapi.Vec3(0.24,0.8,0)),'Pepsi_Cola_Wild_Cherry_Diet',num_env,0)
        Pepsi_Cola_Caffeine_Free_actor = self.gym.create_actor(env, self.Pepsi_Cola_Caffeine_Free_asset, gymapi.Transform(p=gymapi.Vec3(0.24,0.8,0.12)),'Pepsi_Cola_Caffeine_Free',num_env,0)

        shelf_actor = self.gym.create_actor(env, self.shelf_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.35,1.95,0.45)),'shelf',num_env,0)
        CoQ10_BjTLbuRVt1t_actor = self.gym.create_actor(env, self.CoQ10_BjTLbuRVt1t_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.35,1.75,0.5)),'CoQ10_BjTLbuRVt1t',num_env,0)
        Quercetin_500_actor = self.gym.create_actor(env, self.Quercetin_500_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.35,1.85,0.5)),'Quercetin_500',num_env,0)
        Marc_Anthony_True_Professional_Oil_of_Morocco_Argan_Oil_Treatment_actor = self.gym.create_actor(env, self.Marc_Anthony_True_Professional_Oil_of_Morocco_Argan_Oil_Treatment_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.3,1.75,0.85)),'Marc_Anthony_True_Professional_Oil_of_Morocco_Argan_Oil_Treatment_file',num_env,0)
        Playmates_Industrial_CoSplinter_Teenage_Mutant_Ninja_Turtle_Action_Figure_actor = self.gym.create_actor(env, self.Playmates_Industrial_CoSplinter_Teenage_Mutant_Ninja_Turtle_Action_Figure_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.3,1.85,0.85)),'Playmates_Industrial_CoSplinter_Teenage_Mutant_Ninja_Turtle_Action_Figure',num_env,0)
        Krill_Oil_actor = self.gym.create_actor(env, self.Krill_Oil_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.3,2.08,0.85)),'Krill_Oil',num_env,0)
        Pet_Dophilus_powder_actor = self.gym.create_actor(env, self.Pet_Dophilus_powder_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.3,2.22,0.85)),'Pet_Dophilus_powder',num_env,0)
        Timberland_Mens_Earthkeepers_Newmarket_6Inch_Cupsole_Boot_actor = self.gym.create_actor(env, self.Timberland_Mens_Earthkeepers_Newmarket_6Inch_Cupsole_Boot_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.35,2.08,0.47)),'Timberland_Mens_Earthkeepers_Newmarket_6Inch_Cupsole_Boot',num_env,0)
        Timberland_Mens_Earthkeepers_Newmarket_6Inch_Cupsole_Boot_actor = self.gym.create_actor(env, self.Timberland_Mens_Earthkeepers_Newmarket_6Inch_Cupsole_Boot_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.35,2.22,0.47)),'Timberland_Mens_Earthkeepers_Newmarket_6Inch_Cupsole_Boot',num_env,0)
        self.gym.set_rigid_body_color(env,sofa_actor,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.3,0.2,0.2))
        self.gym.set_rigid_body_color(env,wall1_handle,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.9,0.9,0.9))
        self.gym.set_rigid_body_color(env,wall2_handle,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.9,0.9,0.9))
        self.gym.set_rigid_body_color(env,wall3_handle,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.9,0.9,0.9))
        tv_actor = self.gym.create_actor(env, self.tv_asset, gymapi.Transform(p=gymapi.Vec3(1.5,2.44,1.5),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2)), 'tv', num_env, 0)

class Env3(EnvBase):
    def set_robot_asset(self):
        print("UR X")

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
        self.floor_pose1 = gymapi.Transform(p=gymapi.Vec3(0.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose2 = gymapi.Transform(p=gymapi.Vec3(1.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose3 = gymapi.Transform(p=gymapi.Vec3(2.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose4 = gymapi.Transform(p=gymapi.Vec3(0.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose5 = gymapi.Transform(p=gymapi.Vec3(1.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose6 = gymapi.Transform(p=gymapi.Vec3(2.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose7 = gymapi.Transform(p=gymapi.Vec3(0.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose8 = gymapi.Transform(p=gymapi.Vec3(1.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose9 = gymapi.Transform(p=gymapi.Vec3(2.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose10 = gymapi.Transform(p=gymapi.Vec3(0.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose11 = gymapi.Transform(p=gymapi.Vec3(1.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose12 = gymapi.Transform(p=gymapi.Vec3(2.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose13 = gymapi.Transform(p=gymapi.Vec3(3.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose14 = gymapi.Transform(p=gymapi.Vec3(3.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose15 = gymapi.Transform(p=gymapi.Vec3(3.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose16 = gymapi.Transform(p=gymapi.Vec3(3.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))

        self.wall_asset_1 = self.gym.create_box(self.sim,0.1,4.0,3.0,simple_options)
        self.wall1_pose = gymapi.Transform(p=gymapi.Vec3(-0.5,1.0,1.5))
        self.wall3_pose = gymapi.Transform(p=gymapi.Vec3(3.5,1.0,1.5))
        self.wall_asset_2 = self.gym.create_box(self.sim,4.0,0.1,3.0,simple_options)
        self.wall2_pose = gymapi.Transform(p=gymapi.Vec3(1.5,3.0,1.5))

        self.rug_asset = self.gym.create_box(self.sim,1.5,1.8,0.01,simple_options)
        self.rug_pose = gymapi.Transform(p=gymapi.Vec3(1.5,1.5,0.01))
        
        mini_sofa_file = "urdf/living_room/minisofa/model.urdf"
        self.mini_sofa_asset = self.gym.load_asset(self.sim, self.asset_root, mini_sofa_file, simple_options)
        
        Desk_file = "urdf/living_room/Desk/model.urdf"
        self.Desk_asset = self.gym.load_asset(self.sim, self.asset_root, Desk_file, simple_options)
        
        shelf_file = "urdf/living_room/shelf/model.urdf"
        self.shelf_asset = self.gym.load_asset(self.sim, self.asset_root, shelf_file, simple_options)
        
        lamp_file = "urdf/living_room/LampAndStand/model.urdf"
        self.lamp_asset = self.gym.load_asset(self.sim, self.asset_root, lamp_file, simple_options)
        
        Waste_Basket_file = "urdf/living_room/Hefty_Waste_Basket_Decorative_Bronze_85_liter/model.urdf"
        self.Waste_Basket_asset = self.gym.load_asset(self.sim, self.asset_root, Waste_Basket_file, simple_options)
        
        bowl_file = "urdf/living_room/bowl/model.urdf"
        self.bowl_asset = self.gym.load_asset(self.sim, self.asset_root, bowl_file, simple_options)
        
        banana_file = "urdf/living_room/011_banana/011_banana.urdf"
        self.banana_asset = self.gym.load_asset(self.sim, self.asset_root, banana_file, simple_options)
        
        tv_file = "urdf/living_room/tv/model.urdf"
        self.tv_asset = self.gym.load_asset(self.sim, self.asset_root, tv_file, simple_options)
        
        Jenga_Classic_Game_file = "urdf/living_room/2_of_Jenga_Classic_Game/model.urdf"
        self.Jenga_Classic_Game_asset = self.gym.load_asset(self.sim, self.asset_root, Jenga_Classic_Game_file, simple_options)
        
        Tetris_Link_Game_file = "urdf/living_room/Tetris_Link_Game/model.urdf"
        self.Tetris_Link_Game_asset = self.gym.load_asset(self.sim, self.asset_root, Tetris_Link_Game_file, simple_options)
        
        Threshold_Porcelain_Teapot_White_file = "urdf/living_room/Threshold_Porcelain_Teapot_White/model.urdf"
        self.Threshold_Porcelain_Teapot_White_asset = self.gym.load_asset(self.sim, self.asset_root, Threshold_Porcelain_Teapot_White_file, simple_options)
        
        Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White_file = "urdf/living_room/Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White/model.urdf"
        self.Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White_asset = self.gym.load_asset(self.sim, self.asset_root, Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White_file, simple_options)
        
        MY_MOOD_MEMO_file = "urdf/living_room/MY_MOOD_MEMO/model.urdf"
        self.MY_MOOD_MEMO_asset = self.gym.load_asset(self.sim, self.asset_root, MY_MOOD_MEMO_file, simple_options)

    def set_object_asset(self):
        print("object asset is set")
    
    def create_robot(self, env, num_env):
        print("x")

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

        rug_handle = self.gym.create_actor(env, self.rug_asset, self.rug_pose, "rug", num_env, 0)
        self.gym.set_rigid_body_color(env,rug_handle,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.15,0.1,0.1))
        sofa_actor = self.gym.create_actor(env, self.mini_sofa_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 1),p=gymapi.Vec3(0.45,2.15,0.0)),'sofa',num_env,0)
        sofa_actor2 = self.gym.create_actor(env, self.mini_sofa_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -1),p=gymapi.Vec3(2.5,2.15,0.0)),'sofa2',num_env,0)
        Desk_actor = self.gym.create_actor(env, self.Desk_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(1.5,1.5,-0.5)),'desk',num_env,0)
        self.gym.set_actor_scale(env, Desk_actor,1.5)
        Waste_Basket_actor = self.gym.create_actor(env, self.Waste_Basket_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 1),p=gymapi.Vec3(0.5,1.5,0.0)),'waste_basket',num_env,0)
        self.gym.set_rigid_body_color(env,sofa_actor,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.1,0.1,0.1))
        self.gym.set_rigid_body_color(env,Desk_actor,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.106,0.073,0.064))
        bowl_actor = self.gym.create_actor(env, self.bowl_asset, gymapi.Transform(p=gymapi.Vec3(1.5,1.5,0.385),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2)), 'bowl', num_env, 0)
        banana_actor = self.gym.create_actor(env, self.banana_asset, gymapi.Transform(p=gymapi.Vec3(1.5,1.53,0.37),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 1.0, 0.0), -np.pi/2)), 'banana1', num_env, 0)
        banana_actor = self.gym.create_actor(env, self.banana_asset, gymapi.Transform(p=gymapi.Vec3(1.46,1.53,0.37),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 1.0, 0.0), -np.pi/2)), 'banana2', num_env, 0)
        banana_actor = self.gym.create_actor(env, self.banana_asset, gymapi.Transform(p=gymapi.Vec3(1.53,1.53,0.37),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 1.0, 0.0), -np.pi/2)), 'banana3', num_env, 0)
        self.gym.set_actor_scale(env, bowl_actor,2.0)
        self.gym.set_rigid_body_color(env,bowl_actor,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.1,0.22,0.1))
        tv_actor = self.gym.create_actor(env, self.tv_asset, gymapi.Transform(p=gymapi.Vec3(1.5,2.74,1.5),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2)), 'tv', num_env, 0)
        Jenga_Classic_Game_actor = self.gym.create_actor(env, self.Jenga_Classic_Game_asset, gymapi.Transform(p=gymapi.Vec3(1.23,1.7,0.34),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2)), 'Jenga_Classic_Game', num_env, 0)
        Tetris_Link_Game_actor = self.gym.create_actor(env, self.Tetris_Link_Game_asset, gymapi.Transform(p=gymapi.Vec3(1.85,1.25,0.37),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), -np.pi/2)), 'Tetris_Link_Game', num_env, 0)
        Threshold_Porcelain_Teapot_White_actor = self.gym.create_actor(env, self.Threshold_Porcelain_Teapot_White_asset, gymapi.Transform(p=gymapi.Vec3(1.15,1.35,0.32),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2)), 'Threshold_Porcelain_Teapot_White', num_env, 0)
        Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White_actor = self.gym.create_actor(env, self.Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White_asset, gymapi.Transform(p=gymapi.Vec3(1.05,1.75,0.32)), 'Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White', num_env, 0)
        Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White_actor2 = self.gym.create_actor(env, self.Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White_asset, gymapi.Transform(p=gymapi.Vec3(1.72,1.75,0.32)), 'Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White2', num_env, 0)
        MY_MOOD_MEMO_actor = self.gym.create_actor(env, self.MY_MOOD_MEMO_asset, gymapi.Transform(p=gymapi.Vec3(0.1,2.7,1.6),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), np.pi/2)), 'MY_MOOD_MEMO', num_env, 0)
        self.gym.set_actor_scale(env, MY_MOOD_MEMO_actor,1.3)
        
class Env4(EnvBase):
    def set_robot_asset(self):
        print("UR X")

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
        self.floor_pose1 = gymapi.Transform(p=gymapi.Vec3(0.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose2 = gymapi.Transform(p=gymapi.Vec3(1.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose3 = gymapi.Transform(p=gymapi.Vec3(2.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose4 = gymapi.Transform(p=gymapi.Vec3(0.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose5 = gymapi.Transform(p=gymapi.Vec3(1.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose6 = gymapi.Transform(p=gymapi.Vec3(2.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose7 = gymapi.Transform(p=gymapi.Vec3(0.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose8 = gymapi.Transform(p=gymapi.Vec3(1.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose9 = gymapi.Transform(p=gymapi.Vec3(2.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose10 = gymapi.Transform(p=gymapi.Vec3(0.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose11 = gymapi.Transform(p=gymapi.Vec3(1.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose12 = gymapi.Transform(p=gymapi.Vec3(2.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose13 = gymapi.Transform(p=gymapi.Vec3(3.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose14 = gymapi.Transform(p=gymapi.Vec3(3.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose15 = gymapi.Transform(p=gymapi.Vec3(3.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose16 = gymapi.Transform(p=gymapi.Vec3(3.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))

        self.wall_asset_1 = self.gym.create_box(self.sim,0.1,4.0,3.0,simple_options)
        self.wall1_pose = gymapi.Transform(p=gymapi.Vec3(-0.5,1.0,1.5))
        self.wall3_pose = gymapi.Transform(p=gymapi.Vec3(3.5,1.0,1.5))
        self.wall_asset_2 = self.gym.create_box(self.sim,4.0,0.1,3.0,simple_options)
        self.wall2_pose = gymapi.Transform(p=gymapi.Vec3(1.5,3.0,1.5))

        shelf_file = "urdf/living_room/shelf/model.urdf"
        self.shelf_asset = self.gym.load_asset(self.sim, self.asset_root, shelf_file, simple_options)
        
        self.rug_asset = self.gym.create_box(self.sim,1.9,1.9,0.01,simple_options)
        self.rug_pose = gymapi.Transform(p=gymapi.Vec3(1.5,1.5,0.01))
        
        WoodenChair_file = "urdf/living_room/VisitorChair/model.urdf"
        self.WoodenChair_asset = self.gym.load_asset(self.sim, self.asset_root,WoodenChair_file, simple_options)
        
        tv_stand_file = "urdf/living_room/TVStand/model.urdf"
        self.tv_stand_asset = self.gym.load_asset(self.sim, self.asset_root, tv_stand_file, simple_options)
        
        AdjTable_file = "urdf/living_room/AdjTable/model.urdf"
        self.AdjTable_asset = self.gym.load_asset(self.sim, self.asset_root, AdjTable_file, simple_options)
        
        Desk_file = "urdf/living_room/Desk/model.urdf"
        self.Desk_asset = self.gym.load_asset(self.sim, self.asset_root, Desk_file, simple_options)
        
        shelf_file = "urdf/living_room/shelf/model.urdf"
        self.shelf_asset = self.gym.load_asset(self.sim, self.asset_root, shelf_file, simple_options)
        
        lamp_file = "urdf/living_room/LampAndStand/model.urdf"
        self.lamp_asset = self.gym.load_asset(self.sim, self.asset_root, lamp_file, simple_options)
        
        Waste_Basket_file = "urdf/living_room/Hefty_Waste_Basket_Decorative_Bronze_85_liter/model.urdf"
        self.Waste_Basket_asset = self.gym.load_asset(self.sim, self.asset_root, Waste_Basket_file, simple_options)
        
        Drawer_file = "urdf/living_room/Drawer/model.urdf"
        self.Drawer_asset = self.gym.load_asset(self.sim, self.asset_root, Drawer_file, simple_options)
        
        lamp_file = "urdf/living_room/LampAndStand/model.urdf"
        self.lamp_asset = self.gym.load_asset(self.sim, self.asset_root, lamp_file, simple_options)
        
        tv_file = "urdf/living_room/tv/model.urdf"
        self.tv_asset = self.gym.load_asset(self.sim, self.asset_root, tv_file, simple_options)
        
        SPEED_BOAT_file = "urdf/living_room/SPEED_BOAT/model.urdf"
        self.SPEED_BOAT_asset = self.gym.load_asset(self.sim, self.asset_root, SPEED_BOAT_file, simple_options)
        
        Lenovo_Yoga_2_11_file = "urdf/living_room/Lenovo_Yoga_2_11/model.urdf"
        self.Lenovo_Yoga_2_11_asset = self.gym.load_asset(self.sim, self.asset_root,Lenovo_Yoga_2_11_file, simple_options)
        
        INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count_file = "urdf/living_room/INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count/model.urdf"
        self.INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count_asset = self.gym.load_asset(self.sim, self.asset_root,INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count_file, simple_options)
        
        Android_Figure_Chrome_file = "urdf/living_room/Android_Figure_Chrome/model.urdf"
        self.Android_Figure_Chrome_asset = self.gym.load_asset(self.sim, self.asset_root,Android_Figure_Chrome_file, simple_options)
        
        Android_Figure_Orange_file = "urdf/living_room/Android_Figure_Orange/model.urdf"
        self.Android_Figure_Orange_asset = self.gym.load_asset(self.sim, self.asset_root,Android_Figure_Orange_file, simple_options)
        
        Android_Figure_Panda_file = "urdf/living_room/Android_Figure_Panda/model.urdf"
        self.Android_Figure_Panda_asset = self.gym.load_asset(self.sim, self.asset_root,Android_Figure_Panda_file, simple_options)
        
        Android_Lego_file = "urdf/living_room/Android_Lego/model.urdf"
        self.Android_Lego_asset = self.gym.load_asset(self.sim, self.asset_root,Android_Lego_file, simple_options)
        
        LEGO_Star_Wars_Advent_Calendar_file = "urdf/living_room/LEGO_Star_Wars_Advent_Calendar/model.urdf"
        self.LEGO_Star_Wars_Advent_Calendar_asset = self.gym.load_asset(self.sim, self.asset_root,LEGO_Star_Wars_Advent_Calendar_file, simple_options)
        
        MONKEY_BOWLING_file = "urdf/living_room/MONKEY_BOWLING/model.urdf"
        self.MONKEY_BOWLING_asset = self.gym.load_asset(self.sim, self.asset_root,MONKEY_BOWLING_file, simple_options)
        
        DOLL_FAMILY_file = "urdf/living_room/DOLL_FAMILY/model.urdf"
        self.DOLL_FAMILY_asset = self.gym.load_asset(self.sim, self.asset_root,DOLL_FAMILY_file, simple_options)
        
        My_First_Rolling_Lion_file = "urdf/living_room/My_First_Rolling_Lion/model.urdf"
        self.My_First_Rolling_Lion_asset = self.gym.load_asset(self.sim, self.asset_root,My_First_Rolling_Lion_file, simple_options)
        
        My_Little_Pony_Princess_Celestia_file = "urdf/living_room/My_Little_Pony_Princess_Celestia/model.urdf"
        self.My_Little_Pony_Princess_Celestia_asset = self.gym.load_asset(self.sim, self.asset_root,My_Little_Pony_Princess_Celestia_file, simple_options)
        
        Thomas_Friends_Woodan_Railway_Henry_file = "urdf/living_room/Thomas_Friends_Woodan_Railway_Henry/model.urdf"
        self.Thomas_Friends_Woodan_Railway_Henry_asset = self.gym.load_asset(self.sim, self.asset_root,Thomas_Friends_Woodan_Railway_Henry_file, simple_options)
        
        Perricone_MD_OVM_file = "urdf/living_room/Perricone_MD_OVM/model.urdf"
        self.Perricone_MD_OVM_asset = self.gym.load_asset(self.sim, self.asset_root,Perricone_MD_OVM_file, simple_options)
        
        Perricone_MD_The_Cold_Plasma_Face_Eyes_Duo_file = "urdf/living_room/Perricone_MD_The_Cold_Plasma_Face_Eyes_Duo/model.urdf"
        self.Perricone_MD_The_Cold_Plasma_Face_Eyes_Duo_asset = self.gym.load_asset(self.sim, self.asset_root,Perricone_MD_The_Cold_Plasma_Face_Eyes_Duo_file, simple_options)
        
        Perricone_MD_Vitamin_C_Ester_Serum_file = "urdf/living_room/Perricone_MD_Vitamin_C_Ester_Serum/model.urdf"
        self.Perricone_MD_Vitamin_C_Ester_Serum_asset = self.gym.load_asset(self.sim, self.asset_root,Perricone_MD_Vitamin_C_Ester_Serum_file, simple_options)
        
        Wrigley_Orbit_Mint_Variety_18_Count_file = "urdf/living_room/Wrigley_Orbit_Mint_Variety_18_Count/model.urdf"
        self.Wrigley_Orbit_Mint_Variety_18_Count_asset = self.gym.load_asset(self.sim, self.asset_root,Wrigley_Orbit_Mint_Variety_18_Count_file, simple_options)

    def set_object_asset(self):
        print("object asset is set")
    
    def create_robot(self, env, num_env):
        print("x")

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

        shelf_actor = self.gym.create_actor(env, self.shelf_asset, gymapi.Transform(p=gymapi.Vec3(0.7,2.7,1.1)),'shelf',num_env,0)
        shelf_actor2 = self.gym.create_actor(env, self.shelf_asset, gymapi.Transform(p=gymapi.Vec3(1.5,2.7,1.1)),'shelf2',num_env,0)
        shelf_actor3 = self.gym.create_actor(env, self.shelf_asset, gymapi.Transform(p=gymapi.Vec3(2.3,2.7,1.1)),'shelf3',num_env,0)
        self.gym.set_rigid_body_color(env,shelf_actor,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.106,0.073,0.064))
        self.gym.set_rigid_body_color(env,shelf_actor2,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.106,0.073,0.064))
        self.gym.set_rigid_body_color(env,shelf_actor3,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.106,0.073,0.064))
        Drawer_actor = self.gym.create_actor(env, self.Drawer_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.2,1.8,0.0)),'drawer',num_env,0)
        Desk_actor = self.gym.create_actor(env, self.Desk_asset, gymapi.Transform(p=gymapi.Vec3(1.5,1.5,0)),'desk',num_env,0)
        self.gym.set_actor_scale(env,Desk_actor,1.2)
        self.gym.set_rigid_body_color(env,Desk_actor,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.05,0,0))
        rug_handle = self.gym.create_actor(env, self.rug_asset, self.rug_pose, "rug", num_env, 0)
        self.gym.set_rigid_body_color(env,rug_handle,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.25,0.25,0.0))
        tv_stand_actor = self.gym.create_actor(env, self.tv_stand_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2),p=gymapi.Vec3(-0.21,1.5,0.0)),'dining',num_env,0)
        self.gym.set_rigid_body_color(env,tv_stand_actor,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.1,0.1,0.1))
        WoodenChair_actor = self.gym.create_actor(env, self.WoodenChair_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(2.2,1.8,0)),'WoodenChair',num_env,0)
        WoodenChair_actor2 = self.gym.create_actor(env, self.WoodenChair_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(2.2,1.2,0)),'WoodenChair2',num_env,0)
        self.gym.set_actor_scale(env,WoodenChair_actor,0.7)
        self.gym.set_actor_scale(env,WoodenChair_actor2,0.7)
        #WoodenChair_actor3 = self.gym.create_actor(env, self.WoodenChair_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0),np.pi),p=gymapi.Vec3(1.35,1,0)),'WoodenChair3',num_env,0)
        #WoodenChair_actor4 = self.gym.create_actor(env, self.WoodenChair_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0),np.pi),p=gymapi.Vec3(1.75,1,0)),'WoodenChair4',num_env,0)
        lamp_actor = self.gym.create_actor(env, self.lamp_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(3.2,2.5,0.0)),'lamp',num_env,0)
        self.gym.set_actor_scale(env,lamp_actor,0.8)
        tv_actor = self.gym.create_actor(env, self.tv_asset, gymapi.Transform(p=gymapi.Vec3(-0.43,1.5,1.0)), 'tv', num_env, 0)
        SPEED_BOAT_actor = self.gym.create_actor(env, self.SPEED_BOAT_asset, gymapi.Transform(p=gymapi.Vec3(0.9,2.7,1.5),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2)), 'SPEED_BOAT', num_env, 0)
        Wishbone_Pencil_Case_actor = self.gym.create_actor(env, self.Lenovo_Yoga_2_11_asset, gymapi.Transform(p=gymapi.Vec3(1.55,1.25,0.65),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2)), 'Lenovo_Yoga_2_11', num_env, 0)
        Android_Figure_Chrome_actor = self.gym.create_actor(env, self.Android_Figure_Chrome_asset, gymapi.Transform(p=gymapi.Vec3(0.9,2.7,1.13)), 'Android_Figure_Chrome', num_env, 0)
        Android_Figure_Orange_actor = self.gym.create_actor(env, self.Android_Figure_Orange_asset, gymapi.Transform(p=gymapi.Vec3(0.5,2.7,1.13)), 'Android_Figure_Orange', num_env, 0)
        Android_Figure_Panda_actor = self.gym.create_actor(env, self.Android_Figure_Panda_asset, gymapi.Transform(p=gymapi.Vec3(0.6,2.7,1.13)), 'Android_Figure_Panda', num_env, 0)
        Android_Lego_actor = self.gym.create_actor(env, self.Android_Lego_asset, gymapi.Transform(p=gymapi.Vec3(0.8,2.7,1.13)), 'Android_Lego', num_env, 0)
        LEGO_Star_Wars_Advent_Calendar_actor = self.gym.create_actor(env, self.LEGO_Star_Wars_Advent_Calendar_asset, gymapi.Transform(p=gymapi.Vec3(-0.2,1.5,0.48),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), -np.pi/2)), 'LEGO_Star_Wars_Advent_Calendar', num_env, 0)
        MONKEY_BOWLING_actor = self.gym.create_actor(env, self.MONKEY_BOWLING_asset, gymapi.Transform(p=gymapi.Vec3(0.53,2.65,1.5)), 'MONKEY_BOWLING', num_env, 0)
        DOLL_FAMILY_actor = self.gym.create_actor(env, self.DOLL_FAMILY_asset, gymapi.Transform(p=gymapi.Vec3(1.35,2.65,1.5)), 'DOLL_FAMILY', num_env, 0)
        My_First_Rolling_Lion_actor = self.gym.create_actor(env, self.My_First_Rolling_Lion_asset, gymapi.Transform(p=gymapi.Vec3(1.35,2.65,1.13)), 'My_First_Rolling_Lion', num_env, 0)
        My_Little_Pony_Princess_Celestia_actor = self.gym.create_actor(env, self.My_Little_Pony_Princess_Celestia_asset, gymapi.Transform(p=gymapi.Vec3(1.72,2.66,1.13)), 'My_Little_Pony_Princess_Celestia', num_env, 0)
        Thomas_Friends_Woodan_Railway_Henry_actor = self.gym.create_actor(env, self.Thomas_Friends_Woodan_Railway_Henry_asset, gymapi.Transform(p=gymapi.Vec3(1.7,2.66,1.5)), 'Thomas_Friends_Woodan_Railway_Henry', num_env, 0)
        Perricone_MD_OVM_actor = self.gym.create_actor(env, self.Perricone_MD_OVM_asset, gymapi.Transform(p=gymapi.Vec3(2.5,2.66,1.13)), 'Perricone_MD_OVM', num_env, 0)
        Perricone_MD_Vitamin_C_Ester_Serum_actor = self.gym.create_actor(env, self.Perricone_MD_Vitamin_C_Ester_Serum_asset, gymapi.Transform(p=gymapi.Vec3(2.5,2.66,1.5)), 'Perricone_MD_Vitamin_C_Ester_Serum', num_env, 0)
        Perricone_MD_The_Cold_Plasma_Face_Eyes_Duo_actor = self.gym.create_actor(env, self.Perricone_MD_The_Cold_Plasma_Face_Eyes_Duo_asset, gymapi.Transform(p=gymapi.Vec3(2.2,2.66,1.145)), 'Perricone_MD_The_Cold_Plasma_Face_Eyes_Duo', num_env, 0)
        Wrigley_Orbit_Mint_Variety_18_Count_actor = self.gym.create_actor(env, self.Wrigley_Orbit_Mint_Variety_18_Count_asset, gymapi.Transform(p=gymapi.Vec3(2.1,2.66,1.5)), 'Wrigley_Orbit_Mint_Variety_18_Count', num_env, 0)
        
        INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count_actor = self.gym.create_actor(env, self.INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count_asset, gymapi.Transform(p=gymapi.Vec3(1.55,1.75,0.65)), 'INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count', num_env, 0)

class Env5(EnvBase):
    def set_robot_asset(self):
        print("UR X")

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
        self.floor_pose1 = gymapi.Transform(p=gymapi.Vec3(0.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose2 = gymapi.Transform(p=gymapi.Vec3(1.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose3 = gymapi.Transform(p=gymapi.Vec3(2.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose4 = gymapi.Transform(p=gymapi.Vec3(0.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose5 = gymapi.Transform(p=gymapi.Vec3(1.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose6 = gymapi.Transform(p=gymapi.Vec3(2.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose7 = gymapi.Transform(p=gymapi.Vec3(0.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose8 = gymapi.Transform(p=gymapi.Vec3(1.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose9 = gymapi.Transform(p=gymapi.Vec3(2.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose10 = gymapi.Transform(p=gymapi.Vec3(0.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose11 = gymapi.Transform(p=gymapi.Vec3(1.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose12 = gymapi.Transform(p=gymapi.Vec3(2.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose13 = gymapi.Transform(p=gymapi.Vec3(3.0,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose14 = gymapi.Transform(p=gymapi.Vec3(3.0,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose15 = gymapi.Transform(p=gymapi.Vec3(3.0,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose16 = gymapi.Transform(p=gymapi.Vec3(3.0,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))

        self.wall_asset_1 = self.gym.create_box(self.sim,0.1,4.0,3.0,simple_options)
        self.wall1_pose = gymapi.Transform(p=gymapi.Vec3(-0.5,1.0,1.5))
        self.wall3_pose = gymapi.Transform(p=gymapi.Vec3(3.5,1.0,1.5))
        self.wall_asset_2 = self.gym.create_box(self.sim,4.0,0.1,3.0,simple_options)
        self.wall2_pose = gymapi.Transform(p=gymapi.Vec3(1.5,3.0,1.5))

        self.shelf_asset = self.gym.create_box(self.sim,1.2,0.3,0.05,simple_options)
        self.shelf_pose = gymapi.Transform(p=gymapi.Vec3(0.5,2.65,1.6))

        mini_sofa_file = "urdf/living_room/Bed/model.urdf"
        self.mini_sofa_asset = self.gym.load_asset(self.sim, self.asset_root, mini_sofa_file, simple_options)
        
        Desk_file = "urdf/living_room/Desk/model.urdf"
        self.Desk_asset = self.gym.load_asset(self.sim, self.asset_root, Desk_file, simple_options)
        
        lamp_file = "urdf/living_room/LampAndStand/model.urdf"
        self.lamp_asset = self.gym.load_asset(self.sim, self.asset_root, lamp_file, simple_options)
        
        Waste_Basket_file = "urdf/living_room/Hefty_Waste_Basket_Decorative_Bronze_85_liter/model.urdf"
        self.Waste_Basket_asset = self.gym.load_asset(self.sim, self.asset_root, Waste_Basket_file, simple_options)
        
        bowl_file = "urdf/living_room/bowl/model.urdf"
        self.bowl_asset = self.gym.load_asset(self.sim, self.asset_root, bowl_file, simple_options)
        
        WoodenChair_file = "urdf/living_room/WoodenChair/model.urdf"
        self.WoodenChair_asset = self.gym.load_asset(self.sim, self.asset_root,WoodenChair_file, simple_options)
        
        CoffeeMaker_file = "urdf/living_room/CoffeeMaker/model.urdf"
        self.CoffeeMaker_asset = self.gym.load_asset(self.sim, self.asset_root, CoffeeMaker_file, simple_options)
        
        Wine_bottle_file = "urdf/living_room/23_Wine_bottle/model.urdf"
        self.Wine_bottle_asset = self.gym.load_asset(self.sim, self.asset_root, Wine_bottle_file, simple_options)
        
        Camera_file = "urdf/living_room/24_Camera/model.urdf"
        self.Camera_asset = self.gym.load_asset(self.sim, self.asset_root, Camera_file, simple_options)
        
        Pencil_case_file = "urdf/living_room/02_Pencil_case/model.urdf"
        self.Pencil_case_asset = self.gym.load_asset(self.sim, self.asset_root, Pencil_case_file, simple_options)
        
        Pencil_sharpener_file = "urdf/living_room/03_Pencil_sharpener/model.urdf"
        self.Pencil_sharpener_asset = self.gym.load_asset(self.sim, self.asset_root, Pencil_sharpener_file, simple_options)
        
        Book_shelf_file = "urdf/living_room/17_Book_shelf/model.urdf"
        self.Book_shelf_asset = self.gym.load_asset(self.sim, self.asset_root, Book_shelf_file, simple_options)
        
        Dumbbell_file = "urdf/living_room/22_Dumbbell/model.urdf"
        self.Dumbbell_asset = self.gym.load_asset(self.sim, self.asset_root,Dumbbell_file, simple_options)
        
        Asus_80211ac_DualBand_Gigabit_Wireless_Router_RTAC68R_file = "urdf/living_room/Asus_80211ac_DualBand_Gigabit_Wireless_Router_RTAC68R/model.urdf"
        self.Asus_80211ac_DualBand_Gigabit_Wireless_Router_RTAC68R_asset = self.gym.load_asset(self.sim, self.asset_root,Asus_80211ac_DualBand_Gigabit_Wireless_Router_RTAC68R_file, simple_options)
        
        Asus_Z97AR_LGA_1150_Intel_ATX_Motherboard_file = "urdf/living_room/Asus_Z97AR_LGA_1150_Intel_ATX_Motherboard/model.urdf"
        self.Asus_Z97AR_LGA_1150_Intel_ATX_Motherboard_asset = self.gym.load_asset(self.sim, self.asset_root,Asus_Z97AR_LGA_1150_Intel_ATX_Motherboard_file, simple_options)
    
    def set_object_asset(self):
        print("object asset is set")
    
    def create_robot(self, env, num_env):
        print("x")

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
    
        shelf_actor = self.gym.create_actor(env, self.shelf_asset, self.shelf_pose,'shelf',num_env,0)
        self.gym.set_rigid_body_color(env,shelf_actor,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(111/255,79/255,40/255))
        CoffeeMaker_handle = self.gym.create_actor(env, self.CoffeeMaker_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 2*np.pi),p=gymapi.Vec3(0.2,2.62,1.62)),'CoffeeMaker',num_env,0)
        bed_actor = self.gym.create_actor(env, self.mini_sofa_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(2.45,2.05,0.0)),'bed',num_env,0)
        self.gym.set_actor_scale(env, bed_actor,0.8)
        Desk_actor = self.gym.create_actor(env, self.Desk_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2),p=gymapi.Vec3(0.5,1.7,0)),'desk',num_env,0)
        WoodenChair_actor = self.gym.create_actor(env, self.WoodenChair_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 2*np.pi),p=gymapi.Vec3(0.5,1.9,0)),'WoodenChair',num_env,0)
        self.gym.set_actor_scale(env, Desk_actor,1.2)
        Wine_bottle_actor = self.gym.create_actor(env, self.Wine_bottle_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 2*np.pi),p=gymapi.Vec3(0.78,1.5,0.65)),'Wine_bottle',num_env,0)
        Pencil_case_actor = self.gym.create_actor(env, self.Pencil_case_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 2*np.pi),p=gymapi.Vec3(0.2,1.8,0.65)),'Pencil_case',num_env,0)
        Asus_80211ac_DualBand_Gigabit_Wireless_Router_RTAC68R_actor = self.gym.create_actor(env, self.Asus_80211ac_DualBand_Gigabit_Wireless_Router_RTAC68R_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0,0.0,0.0),-np.pi/2),p=gymapi.Vec3(-0.13,0.7,0.05)),'Asus_80211ac_DualBand_Gigabit_Wireless_Router_RTAC68R',num_env,0)
        Asus_Z97AR_LGA_1150_Intel_ATX_Motherboard_actor = self.gym.create_actor(env, self.Asus_Z97AR_LGA_1150_Intel_ATX_Motherboard_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0,0.0,0.0),-np.pi/2),p=gymapi.Vec3(-0.11,0.75,0.13)),'Asus_Z97AR_LGA_1150_Intel_ATX_Motherboard',num_env,0)
        Dumbbell_actor = self.gym.create_actor(env, self.Dumbbell_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/3),p=gymapi.Vec3(0.17,1.52,0.68)),'Dumbbell',num_env,0)
        Dumbbell_actor2 = self.gym.create_actor(env, self.Dumbbell_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/4),p=gymapi.Vec3(0.25,1.62,0.68)),'Dumbbell',num_env,0)
        self.gym.set_actor_scale(env, Dumbbell_actor,0.3)
        self.gym.set_actor_scale(env, Dumbbell_actor2,0.3)
        self.gym.set_actor_scale(env, Pencil_case_actor,4)
        Pencil_sharpener_actor = self.gym.create_actor(env, self.Pencil_sharpener_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 2*np.pi),p=gymapi.Vec3(0.1,1.8,0.65)),'Pencil_sharpener',num_env,0)
        Camera_actor = self.gym.create_actor(env, self.Camera_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 2*np.pi),p=gymapi.Vec3(0.45,2.62,1.62)),'Camera',num_env,0)
        Book_shelf_actor = self.gym.create_actor(env, self.Book_shelf_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 2*np.pi),p=gymapi.Vec3(0.68,2.55,1.62)),'Book_shelf',num_env,0)
        Book_shelf_actor2 = self.gym.create_actor(env, self.Book_shelf_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 2*np.pi),p=gymapi.Vec3(0.8,2.55,1.62)),'Book_shelf',num_env,0)
        Book_shelf_actor3= self.gym.create_actor(env, self.Book_shelf_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 2*np.pi),p=gymapi.Vec3(0.92,2.55,1.62)),'Book_shelf',num_env,0)
        self.gym.set_rigid_body_color(env,Desk_actor,0,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.106,0.073,0.064))
        Waste_Basket_actor = self.gym.create_actor(env, self.Waste_Basket_asset, gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 1),p=gymapi.Vec3(3.15,0.95,0.0)),'waste_basket',num_env,0)
    
class LivingRoomEnvManager:
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
        self.current_env.set_wall_type(wall_type)