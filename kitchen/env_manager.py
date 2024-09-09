"""
date: 2024.08.27
author: Seongyong Kim
description: kitchen environment class
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

def get_object_positions(num_objects, x_lim, y_lim):
    min_distance = 0.1

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
        print("set robot asset")

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

        gravity_options = gymapi.AssetOptions()

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

        sink_file = "urdf/kitchen/KitchenSink/model.urdf"
        self.sink_asset = self.gym.load_asset(self.sim, self.asset_root, sink_file, complex_options)
        self.sink_pose = gymapi.Transform(p=gymapi.Vec3(0.45,0.5,0.0), r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        fridge_file = "urdf/kitchen/refrigerator2/model.urdf"
        self.fridge_asset = self.gym.load_asset(self.sim, self.asset_root, fridge_file, simple_options)
        self.fridge_pose = gymapi.Transform(r=gymapi.Quat(0.7071068, 0, 0, 0.7071068),p=gymapi.Vec3(0.5,2.5,-0.05))

        desk_file = "urdf/kitchen/DiningTableWood/model_marble.urdf"
        self.desk_asset = self.gym.load_asset(self.sim, self.asset_root, desk_file, simple_options)
        self.desk_pose = gymapi.Transform(r=gymapi.Quat(0.0,0.0,0.0,1.0),p=gymapi.Vec3(2.5,1.5,0.0))

        cooker_file = "urdf/kitchen/cooker/model.urdf"
        self.cooker_asset = self.gym.load_asset(self.sim, self.asset_root, cooker_file, simple_options)
        self.cooker_pose = gymapi.Transform(r=gymapi.Quat(0.0,0.0,0.0,1.0),p=gymapi.Vec3(2.62,2.4,0.75))

        toaster_file = "urdf/kitchen/toaster/model.urdf"
        self.toaster_asset = self.gym.load_asset(self.sim, self.asset_root, toaster_file, simple_options)
        self.toaster_pose = gymapi.Transform(r=gymapi.Quat(0, 0, -0.7071068, 0.7071068),p=gymapi.Vec3(2.62,2.0,0.75))

        shelf_file = "urdf/kitchen/SquareShelf/model.urdf"
        self.shelf_asset = self.gym.load_asset(self.sim, self.asset_root, shelf_file, simple_options)
        self.shelf_pose = gymapi.Transform(r=gymapi.Quat(0, 0, -0.7071068, 0.7071068),p=gymapi.Vec3(2.8,2.2,0.73))

        coffee_maker_file = "urdf/kitchen/coffeemaker/model.urdf"
        self.coffee_maker_asset = self.gym.load_asset(self.sim, self.asset_root, coffee_maker_file, simple_options)
        self.coffee_maker_pose = gymapi.Transform(p=gymapi.Vec3(2.7,1.6,0.73),r=gymapi.Quat(0.0,0.0,0.0,1.0))
    
        chair_file = "urdf/kitchen/WoodenChair/model.urdf"
        self.chair_asset = self.gym.load_asset(self.sim, self.asset_root, chair_file, simple_options)
        self.chair_pose = gymapi.Transform(p=gymapi.Vec3(1.8,2.0,0.0),r=gymapi.Quat(0.0,0.0,0.0,1.0))

        drawer_file = "urdf/kitchen/drawer/model.urdf"
        self.drawer_asset = self.gym.load_asset(self.sim, self.asset_root, drawer_file, simple_options)
        self.drawer_pose = gymapi.Transform(p=gymapi.Vec3(2.5,0.6,0.73),r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        mugcup_file = "urdf/kitchen/mugcup/model.urdf"
        self.mugcup_asset = self.gym.load_asset(self.sim, self.asset_root, mugcup_file, simple_options)
        self.mugcup_pose = gymapi.Transform(p=gymapi.Vec3(2.5,0.75,0.75),r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        coffeemug_file = "urdf/kitchen/coffeemug/model.urdf"
        self.coffeemug_asset = self.gym.load_asset(self.sim, self.asset_root, coffeemug_file, simple_options)
        self.coffeemug_pose = gymapi.Transform(p=gymapi.Vec3(2.5,0.6,0.75),r=gymapi.Quat(0, 0, 0.0871557, 0.9961947))

        coffeemug2_file = "urdf/kitchen/coffeemug/model.urdf"
        self.coffeemug2_asset = self.gym.load_asset(self.sim, self.asset_root, coffeemug2_file, simple_options)
        self.coffeemug2_pose = gymapi.Transform(p=gymapi.Vec3(2.5,0.45,0.75),r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        gelatin_box_file = "urdf/ycb/009_gelatin_box/model.urdf"
        self.gelatin_box_asset = self.gym.load_asset(self.sim, self.asset_root, gelatin_box_file, gravity_options)
        self.gelatin_box_pose = gymapi.Transform(p=gymapi.Vec3(2.8, 1.3, 0.79),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        cracker_box_file = "urdf/ycb/003_cracker_box/model.urdf"
        self.cracker_box_asset = self.gym.load_asset(self.sim, self.asset_root, cracker_box_file, simple_options)
        self.cracker_pose = gymapi.Transform(p=gymapi.Vec3(2.8, 1.2, 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        sugar_box_file = "urdf/ycb/004_sugar_box/model.urdf"
        self.sugar_box_asset = self.gym.load_asset(self.sim, self.asset_root, sugar_box_file, simple_options)
        self.sugar_box_pose = gymapi.Transform(p=gymapi.Vec3(2.8, 1.1, 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        pudding_box_file = "urdf/ycb/008_pudding_box/model.urdf"
        self.pudding_box_asset = self.gym.load_asset(self.sim, self.asset_root, pudding_box_file, gravity_options)
        self.pudding_box_pose = gymapi.Transform(p=gymapi.Vec3(2.8, 1.0, 0.79),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        yellow_bowl_file = "urdf/kitchen/yellow_bowl/model.urdf"
        self.yellow_bowl_asset = self.gym.load_asset(self.sim, self.asset_root, yellow_bowl_file, simple_options)
        self.yellow_bowl_pose = gymapi.Transform(p=gymapi.Vec3(2.2,1.9,0.73))

        bowl_designed_file = "urdf/kitchen/bowl_designed/model.urdf"
        self.bowl_designed_asset = self.gym.load_asset(self.sim, self.asset_root, bowl_designed_file, simple_options)
        self.bowl_designed_pose = gymapi.Transform(p=gymapi.Vec3(2.2,1.74,0.73))

        sauce1_file = "urdf/kitchen/sauce1/model.urdf"
        self.sauce1_asset = self.gym.load_asset(self.sim, self.asset_root, sauce1_file, simple_options)
        self.sauce1_pose = gymapi.Transform(p=gymapi.Vec3(2.3,1.6,0.73))

        sauce2_file = "urdf/kitchen/sauce2/model.urdf"
        self.sauce2_asset = self.gym.load_asset(self.sim, self.asset_root, sauce2_file, simple_options)
        self.sauce2_pose = gymapi.Transform(p=gymapi.Vec3(2.2,1.6,0.73))

        # set random position for other objects
        y_lim = [0.95,1.4]
        x_lim = [2.35,2.6]
        num_objects = 11
        random_x, random_y = get_object_positions(num_objects, x_lim, y_lim)

        tomato_soup_can_file = "urdf/kitchen/005_tomato_soup_can/model.urdf"
        self.tomato_soup_can_asset = self.gym.load_asset(self.sim, self.asset_root, tomato_soup_can_file, gravity_options)
        self.tomato_soup_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[0],random_y[0], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        potted_meat_can_file = "urdf/kitchen/010_potted_meat_can/model.urdf"
        self.potted_meat_can_asset = self.gym.load_asset(self.sim, self.asset_root, potted_meat_can_file, gravity_options)
        self.potted_meat_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[1],random_y[1], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        chef_can_file = "urdf/ycb/002_master_chef_can/model.urdf"
        self.chef_can_asset = self.gym.load_asset(self.sim, self.asset_root, chef_can_file, gravity_options)
        self.chef_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[2],random_y[2], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        tuna_can_file = "urdf/ycb/007_tuna_fish_can/model.urdf"
        self.tuna_can_asset = self.gym.load_asset(self.sim, self.asset_root, tuna_can_file, gravity_options)
        self.tuna_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[3],random_y[3], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        plum_file = "urdf/ycb/018_plum/model.urdf"
        self.plum_asset = self.gym.load_asset(self.sim, self.asset_root, plum_file, gravity_options)
        self.plum_pose = gymapi.Transform(p=gymapi.Vec3(random_x[4],random_y[4], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        strawberry_file = "urdf/ycb/012_strawberry/model.urdf"
        self.strawberry_asset = self.gym.load_asset(self.sim, self.asset_root, strawberry_file, gravity_options)
        self.strawberry_pose = gymapi.Transform(p=gymapi.Vec3(random_x[5],random_y[5], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        pear_file = "urdf/ycb/016_pear/model.urdf"
        self.pear_asset = self.gym.load_asset(self.sim, self.asset_root, pear_file, gravity_options)
        self.pear_pose = gymapi.Transform(p=gymapi.Vec3(random_x[6],random_y[6], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        mustard_file = "urdf/ycb/006_mustard_bottle/model.urdf"
        self.mustard_asset = self.gym.load_asset(self.sim, self.asset_root, mustard_file, gravity_options)
        self.mustard_pose = gymapi.Transform(p=gymapi.Vec3(random_x[7],random_y[7], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        self.tomato_soup_can2_asset = self.gym.load_asset(self.sim, self.asset_root, tomato_soup_can_file, gravity_options)
        self.tomato_soup_can2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[8],random_y[8], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        self.tuna_can2_asset = self.gym.load_asset(self.sim, self.asset_root, tuna_can_file, gravity_options)
        self.tuna_can2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[9],random_y[9], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        self.potted_meat_can2_asset = self.gym.load_asset(self.sim, self.asset_root, potted_meat_can_file, gravity_options)
        self.potted_meat_can2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[10],random_y[10], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

    def set_object_asset(self):
        object_options = gymapi.AssetOptions()

        bowl_file = "urdf/ycb/024_bowl/model.urdf"
        self.bowl_asset = self.gym.load_asset(self.sim, self.asset_root, bowl_file, object_options)
        self.bowl_pose = gymapi.Transform(p=gymapi.Vec3(2.1, 1.4, 0.825))

        lemon_file = "urdf/ycb/014_lemon/model.urdf"
        self.lemon_asset = self.gym.load_asset(self.sim, self.asset_root, lemon_file, object_options)
        self.lemon_pose = gymapi.Transform(p=gymapi.Vec3(2.1, 1.25, 0.825))

        peach_file = "urdf/ycb/015_peach/model.urdf"
        self.peach_asset = self.gym.load_asset(self.sim, self.asset_root, peach_file, object_options)
        self.peach_pose = gymapi.Transform(p=gymapi.Vec3(2.15, 1.2, 0.825))

        orange_file = "urdf/ycb/017_orange/model.urdf"
        self.orange_asset = self.gym.load_asset(self.sim, self.asset_root, orange_file, object_options)
        self.orange_pose = gymapi.Transform(p=gymapi.Vec3(2.19, 1.1, 0.825))

        mug_file = "urdf/ycb/025_mug/025_mug.urdf"
        self.mug_asset = self.gym.load_asset(self.sim, self.asset_root, mug_file, object_options)
        self.mug_pose = gymapi.Transform(p=gymapi.Vec3(2.3, 0.7, 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        print("object asset is set")

    def create_robot(self, env, num_env):
        print("robot is created")

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
        sink_handle = self.gym.create_actor(env, self.sink_asset, self.sink_pose, "sink", num_env, 0)
        fridge_handle = self.gym.create_actor(env, self.fridge_asset, self.fridge_pose, "fridge", num_env, 0)
        desk_handle = self.gym.create_actor(env, self.desk_asset, self.desk_pose, "desk", num_env, 0)
        cooker_handle = self.gym.create_actor(env, self.cooker_asset, self.cooker_pose, "cooker", num_env, 0)
        toaster_handle = self.gym.create_actor(env, self.toaster_asset, self.toaster_pose, "toaster", num_env, 0)
        shelf_handle = self.gym.create_actor(env, self.shelf_asset, self.shelf_pose, "shelf", num_env, 0)
        coffee_maker_handle = self.gym.create_actor(env, self.coffee_maker_asset, self.coffee_maker_pose, "coffee_maker", num_env, 0)
        chair_handle = self.gym.create_actor(env, self.chair_asset, self.chair_pose, "chair", num_env, 0)
        drawer_handle = self.gym.create_actor(env, self.drawer_asset, self.drawer_pose, "drawer", num_env, 0)
        self.gym.set_actor_scale(env, drawer_handle, 1.5)
        mugcup_handle = self.gym.create_actor(env, self.mugcup_asset, self.mugcup_pose, "mugcup", num_env, 0)
        coffeemug_handle = self.gym.create_actor(env, self.coffeemug_asset, self.coffeemug_pose, "coffeemug", num_env, 0)
        coffeemug2_handle = self.gym.create_actor(env, self.coffeemug2_asset, self.coffeemug2_pose, "coffeemug2", num_env, 0)

        bowl_handle = self.gym.create_actor(env, self.bowl_asset, self.bowl_pose, "bowl", num_env, 0)
        lemon_handle = self.gym.create_actor(env, self.lemon_asset, self.lemon_pose, "lemon", num_env, 0)
        peach_handle = self.gym.create_actor(env, self.peach_asset, self.peach_pose, "peach", num_env, 0)
        orange_handle = self.gym.create_actor(env, self.orange_asset, self.orange_pose, "orange", num_env, 0)
        mug_handle = self.gym.create_actor(env, self.mug_asset, self.mug_pose, "mug", num_env, 0)
        gelatin_box_handle = self.gym.create_actor(env, self.gelatin_box_asset, self.gelatin_box_pose, "gelatin_box", num_env, 0)
        cracker_handle = self.gym.create_actor(env, self.cracker_box_asset, self.cracker_pose, "cracker_box", num_env, 0)
        sugar_box_handle = self.gym.create_actor(env, self.sugar_box_asset, self.sugar_box_pose, "sugar_box", num_env, 0)
        pudding_box_handle = self.gym.create_actor(env, self.pudding_box_asset, self.pudding_box_pose, "pudding_box", num_env, 0)
        yellow_bowl_handle = self.gym.create_actor(env, self.yellow_bowl_asset, self.yellow_bowl_pose, "yellow_bowl", num_env, 0)
        bowl_designed_handle = self.gym.create_actor(env, self.bowl_designed_asset, self.bowl_designed_pose, "bowl_designed", num_env, 0)
        sauce1_handle = self.gym.create_actor(env, self.sauce1_asset, self.sauce1_pose, "sauce1", num_env, 0)
        sauce2_handle = self.gym.create_actor(env, self.sauce2_asset, self.sauce2_pose, "sauce2", num_env, 0)
        chef_can_handle = self.gym.create_actor(env, self.chef_can_asset, self.chef_can_pose, "chef_can", num_env, 0)
        tuna_can_handle = self.gym.create_actor(env, self.tuna_can_asset, self.tuna_can_pose, "tuna_can", num_env, 0)
        plum_handle = self.gym.create_actor(env, self.plum_asset, self.plum_pose, "plum", num_env, 0)
        strawberry_handle = self.gym.create_actor(env, self.strawberry_asset, self.strawberry_pose, "strawberry", num_env, 0)
        pear_handle = self.gym.create_actor(env, self.pear_asset, self.pear_pose, "pear", num_env, 0)
        mustard_handle = self.gym.create_actor(env, self.mustard_asset, self.mustard_pose, "mustard", num_env, 0)
        tomato_soup_can_handle = self.gym.create_actor(env, self.tomato_soup_can_asset, self.tomato_soup_can_pose, "tomato_soup_can", num_env, 0)
        potted_meat_can_handle = self.gym.create_actor(env, self.potted_meat_can_asset, self.potted_meat_can_pose, "potted_meat_can", num_env, 0)
        tomato_soup_can_handle2 = self.gym.create_actor(env, self.tomato_soup_can2_asset, self.tomato_soup_can2_pose, "tomato_soup_can2", num_env, 0)
        tuna_can_handle2 = self.gym.create_actor(env, self.tuna_can2_asset, self.tuna_can2_pose, "tuna_can2", num_env, 0)
        potted_meat_can_handle2 = self.gym.create_actor(env, self.potted_meat_can2_asset, self.potted_meat_can2_pose, "potted_meat_can2", num_env, 0)

class Env2(EnvBase):
    def set_robot_asset(self):
        # config UR5e asset
        ur_asset_file = "urdf/robot/ur_description/kimsy_ur5e_2fg7_small_gap.urdf"
        ur_asset_options = gymapi.AssetOptions()
        ur_asset_options.armature = 0.01
        ur_asset_options.fix_base_link = True
        ur_asset_options.disable_gravity = False
        ur_asset_options.flip_visual_attachments = False
        ur_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE
        ur_asset_options.vhacd_enabled = True
        ur_asset_options.vhacd_params = gymapi.VhacdParams()
        ur_asset_options.vhacd_params.resolution = 300000
        ur_asset_options.vhacd_params.max_convex_hulls = 512
        ur_asset_options.vhacd_params.convex_hull_approximation = False

        self.ur_asset = self.gym.load_asset(self.sim, self.asset_root, ur_asset_file, ur_asset_options)

        # config UR5e dofs
        self.ur_dof_props = self.gym.get_asset_dof_properties(self.ur_asset)
        ur_lower_limits = self.ur_dof_props["lower"]
        ur_upper_limits = self.ur_dof_props["upper"]
        ur_ranges = ur_upper_limits - ur_lower_limits

        self.num_ur_dofs = self.gym.get_asset_dof_count(self.ur_asset)

        stiffness = [400, 400, 400, 400, 400, 400]
        dampings = [40, 40, 40, 40, 40, 40]

        for i in range(6):
            self.ur_dof_props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
            self.ur_dof_props["stiffness"][i] = stiffness[i]
            self.ur_dof_props["damping"][i] = dampings[i]

        self.ur_dof_props["driveMode"][6:].fill(gymapi.DOF_MODE_POS)
        self.ur_dof_props["stiffness"][6:].fill(1)
        self.ur_dof_props["damping"][6:].fill(1)

        print("[INFO] Number of DOFs: ", self.num_ur_dofs)

        self.default_dof_pos = np.zeros(self.num_ur_dofs, dtype=np.float32)
        self.default_dof_pos[:6] = np.array([0.0, -np.pi/4*3, np.pi/2, -np.pi/2, -np.pi/2, 0])
        self.default_dof_pos[6:] = np.array([0.0, 0.0])

        self.default_dof_state = np.zeros(self.num_ur_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"] = self.default_dof_pos

        self.default_dof_pos_tensor = to_torch(self.default_dof_pos, device=self.device)

        ur_link_dict = self.gym.get_asset_rigid_body_dict(self.ur_asset)
        self.ur_gripper_index = ur_link_dict["ee_link"]

        self.ur_pose = gymapi.Transform()
        self.ur_pose.p = gymapi.Vec3(1.75, 1.75, 0.5)

        # add F/T sensor
        body_idx = self.gym.find_asset_rigid_body_index(self.ur_asset, "ee_link")
        sensor_pose = gymapi.Transform(gymapi.Vec3(0,0,0))
        sensor_props = gymapi.ForceSensorProperties()
        sensor_props.enable_forward_dynamics_forces = True
        sensor_props.enable_constraint_solver_forces = True
        sensor_props.use_world_frame = True
        self.gym.create_asset_force_sensor(self.ur_asset, body_idx, sensor_pose, sensor_props)

        robot_base_options = gymapi.AssetOptions()
        robot_base_options.fix_base_link = True
        self.robot_base_asset = self.gym.create_box(self.sim, 0.5, 0.5, 0.5, robot_base_options)
        self.robot_base_pose = gymapi.Transform(p=gymapi.Vec3(1.75,1.75,0.25))

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

        self.wall_asset_1 = self.gym.create_box(self.sim,0.1,4.0,3.0,simple_options)
        self.wall1_pose = gymapi.Transform(p=gymapi.Vec3(0.0,1.0,1.5))
        self.wall3_pose = gymapi.Transform(p=gymapi.Vec3(3.0,1.0,1.5))
        self.wall_asset_2 = self.gym.create_box(self.sim,3.0,0.1,3.0,simple_options)
        self.wall2_pose = gymapi.Transform(p=gymapi.Vec3(1.5,3.0,1.5))

        sink_file = "urdf/kitchen/KitchenCountertop/model.urdf"
        self.sink_asset = self.gym.load_asset(self.sim, self.asset_root, sink_file, simple_options)
        self.sink_pose = gymapi.Transform(p=gymapi.Vec3(0.37,1.8,0.0))

        fridge_file = "urdf/kitchen/Fridge/model.urdf"
        self.fridge_asset = self.gym.load_asset(self.sim, self.asset_root, fridge_file, simple_options)
        self.fridge_pose = gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), -np.pi/2),p=gymapi.Vec3(2.675,2.655,0.0))

        desk_file = "urdf/kitchen/Desk/model_bright_marble.urdf"
        self.desk_asset = self.gym.load_asset(self.sim, self.asset_root, desk_file, simple_options)
        self.desk_pose = gymapi.Transform(r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi),p=gymapi.Vec3(2.7-0.125,1.3,0.0))

        frypan_file = "urdf/kitchen/frypan/model.urdf"
        self.frypan_asset = self.gym.load_asset(self.sim, self.asset_root, frypan_file, simple_options)
        self.frypan_pose = gymapi.Transform(r=gymapi.Quat(0.0,0.0,0.0,1.0),p=gymapi.Vec3(0.35,1.0,0.76))

        frypan_file2 = "urdf/kitchen/frypan/model.urdf"
        self.frypan_asset2 = self.gym.load_asset(self.sim, self.asset_root, frypan_file2, simple_options)
        self.frypan_pose2 = gymapi.Transform(r=gymapi.Quat(0, 0, 0.0871557, 0.9961947),p=gymapi.Vec3(0.55,1.2,0.76))

        cutlery_file = "urdf/kitchen/cutlery_set/model.urdf"
        self.cutlery_asset = self.gym.load_asset(self.sim, self.asset_root, cutlery_file, simple_options)
        self.cutlery_pose = gymapi.Transform(r=gymapi.Quat(0.0,0.0,0.0,1.0),p=gymapi.Vec3(0.15,2.0,0.74))

        drainer_file = "urdf/kitchen/drainer/model.urdf"
        self.drainer_asset = self.gym.load_asset(self.sim, self.asset_root, drainer_file, simple_options)
        self.drainer_pose = gymapi.Transform(r=gymapi.Quat(0.0,0.0,0.0,1.0),p=gymapi.Vec3(0.47,2.0,0.74))

        cooker_file = "urdf/kitchen/cooker/model.urdf"
        self.cooker_asset = self.gym.load_asset(self.sim, self.asset_root, cooker_file, simple_options)
        self.cooker_pose = gymapi.Transform(r=gymapi.Quat(0.0,0.0,0.0,1.0),p=gymapi.Vec3(2.75,1.4,0.73+0.5+0.04))

        toaster_file = "urdf/kitchen/toaster/model.urdf"
        self.toaster_asset = self.gym.load_asset(self.sim, self.asset_root, toaster_file, simple_options)
        self.toaster_pose = gymapi.Transform(r=gymapi.Quat(0, 0, -0.7071068, 0.7071068),p=gymapi.Vec3(2.75,1.0,0.73+0.5+0.04))

        shelf_file = "urdf/kitchen/SquareShelf/model.urdf"
        self.shelf_asset = self.gym.load_asset(self.sim, self.asset_root, shelf_file, simple_options)
        self.shelf_pose = gymapi.Transform(r=gymapi.Quat(0, 0, -0.7071068, 0.7071068),p=gymapi.Vec3(2.8,1.2,0.73+0.5))

        # set random position for other objects
        y_lim = [0.81,1.85]
        x_lim = [2.3,2.89]
        num_objects = 18
        random_x, random_y = get_object_positions(num_objects, x_lim, y_lim)

        tomato_soup_can_file = "urdf/kitchen/005_tomato_soup_can/model.urdf"
        self.tomato_soup_can_asset = self.gym.load_asset(self.sim, self.asset_root, tomato_soup_can_file, gravity_options)
        self.tomato_soup_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[0],random_y[0], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        potted_meat_can_file = "urdf/kitchen/010_potted_meat_can/model.urdf"
        self.potted_meat_can_asset = self.gym.load_asset(self.sim, self.asset_root, potted_meat_can_file, gravity_options)
        self.potted_meat_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[1],random_y[1], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        chef_can_file = "urdf/ycb/002_master_chef_can/model.urdf"
        self.chef_can_asset = self.gym.load_asset(self.sim, self.asset_root, chef_can_file, gravity_options)
        self.chef_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[2],random_y[2], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        tuna_can_file = "urdf/ycb/007_tuna_fish_can/model.urdf"
        self.tuna_can_asset = self.gym.load_asset(self.sim, self.asset_root, tuna_can_file, gravity_options)
        self.tuna_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[3],random_y[3], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        plum_file = "urdf/ycb/018_plum/model.urdf"
        self.plum_asset = self.gym.load_asset(self.sim, self.asset_root, plum_file, gravity_options)
        self.plum_pose = gymapi.Transform(p=gymapi.Vec3(random_x[4],random_y[4], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        strawberry_file = "urdf/ycb/012_strawberry/model.urdf"
        self.strawberry_asset = self.gym.load_asset(self.sim, self.asset_root, strawberry_file, gravity_options)
        self.strawberry_pose = gymapi.Transform(p=gymapi.Vec3(random_x[5],random_y[5], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        pear_file = "urdf/ycb/016_pear/model.urdf"
        self.pear_asset = self.gym.load_asset(self.sim, self.asset_root, pear_file, gravity_options)
        self.pear_pose = gymapi.Transform(p=gymapi.Vec3(random_x[6],random_y[6], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        mustard_file = "urdf/ycb/006_mustard_bottle/model.urdf"
        self.mustard_asset = self.gym.load_asset(self.sim, self.asset_root, mustard_file, gravity_options)
        self.mustard_pose = gymapi.Transform(p=gymapi.Vec3(random_x[7],random_y[7], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        self.tomato_soup_can2_asset = self.gym.load_asset(self.sim, self.asset_root, tomato_soup_can_file, gravity_options)
        self.tomato_soup_can2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[8],random_y[8], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        self.tuna_can2_asset = self.gym.load_asset(self.sim, self.asset_root, tuna_can_file, gravity_options)
        self.tuna_can2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[9],random_y[9], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        self.potted_meat_can2_asset = self.gym.load_asset(self.sim, self.asset_root, potted_meat_can_file, gravity_options)
        self.potted_meat_can2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[10],random_y[10], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        gelatin_box_file = "urdf/ycb/009_gelatin_box/model.urdf"
        self.gelatin_box_asset = self.gym.load_asset(self.sim, self.asset_root, gelatin_box_file, gravity_options)
        self.gelatin_box_pose = gymapi.Transform(p=gymapi.Vec3(random_x[11],random_y[11], 0.79),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        cracker_box_file = "urdf/ycb/003_cracker_box/model.urdf"
        self.cracker_box_asset = self.gym.load_asset(self.sim, self.asset_root, cracker_box_file, gravity_options)
        self.cracker_pose = gymapi.Transform(p=gymapi.Vec3(random_x[12],random_y[12], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        sugar_box_file = "urdf/ycb/004_sugar_box/model.urdf"
        self.sugar_box_asset = self.gym.load_asset(self.sim, self.asset_root, sugar_box_file, gravity_options)
        self.sugar_box_pose = gymapi.Transform(p=gymapi.Vec3(random_x[12],random_y[12], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        pudding_box_file = "urdf/ycb/008_pudding_box/model.urdf"
        self.pudding_box_asset = self.gym.load_asset(self.sim, self.asset_root, pudding_box_file, gravity_options)
        self.pudding_box_pose = gymapi.Transform(p=gymapi.Vec3(random_x[13],random_y[13], 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        yellow_bowl_file = "urdf/kitchen/yellow_bowl/model.urdf"
        self.yellow_bowl_asset = self.gym.load_asset(self.sim, self.asset_root, yellow_bowl_file, gravity_options)
        self.yellow_bowl_pose = gymapi.Transform(p=gymapi.Vec3(random_x[14],random_y[14],0.825))

        bowl_designed_file = "urdf/kitchen/bowl_designed/model.urdf"
        self.bowl_designed_asset = self.gym.load_asset(self.sim, self.asset_root, bowl_designed_file, gravity_options)
        self.bowl_designed_pose = gymapi.Transform(p=gymapi.Vec3(random_x[15],random_y[15],0.825))

        sauce1_file = "urdf/kitchen/sauce1/model.urdf"
        self.sauce1_asset = self.gym.load_asset(self.sim, self.asset_root, sauce1_file, gravity_options)
        self.sauce1_pose = gymapi.Transform(p=gymapi.Vec3(random_x[16],random_y[16],0.825))

        sauce2_file = "urdf/kitchen/sauce2/model.urdf"
        self.sauce2_asset = self.gym.load_asset(self.sim, self.asset_root, sauce2_file, gravity_options)
        self.sauce2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[17],random_y[17],0.825))

    def set_object_asset(self):
        print("object asset is set")
    
    def create_robot(self, env, num_env):
        print("robot is created")

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
        sink_handle = self.gym.create_actor(env, self.sink_asset, self.sink_pose, "sink", num_env, 0)
        fridge_handle = self.gym.create_actor(env, self.fridge_asset, self.fridge_pose, "fridge", num_env, 0)
        desk_handle = self.gym.create_actor(env, self.desk_asset, self.desk_pose, "desk", num_env, 0)
        self.gym.set_actor_scale(env, desk_handle, 1.5)
        frypan_handle = self.gym.create_actor(env, self.frypan_asset, self.frypan_pose, "frypan", num_env, 0)
        frypan_handle2 = self.gym.create_actor(env, self.frypan_asset2, self.frypan_pose2, "frypan2", num_env, 0)
        cutlery_handle = self.gym.create_actor(env, self.cutlery_asset, self.cutlery_pose, "cutlery", num_env, 0)
        drainer_handle = self.gym.create_actor(env, self.drainer_asset, self.drainer_pose, "drainer", num_env, 0)
        cooker_handle = self.gym.create_actor(env, self.cooker_asset, self.cooker_pose, "cooker", num_env, 0)
        toaster_handle = self.gym.create_actor(env, self.toaster_asset, self.toaster_pose, "toaster", num_env, 0)
        shelf_handle = self.gym.create_actor(env, self.shelf_asset, self.shelf_pose, "shelf", num_env, 0)
        tomato_soup_can_handle = self.gym.create_actor(env, self.tomato_soup_can_asset, self.tomato_soup_can_pose, "tomato_soup_can", num_env, 0)
        self.gym.set_actor_scale(env, tomato_soup_can_handle, 1.5)
        potted_meat_can_handle = self.gym.create_actor(env, self.potted_meat_can_asset, self.potted_meat_can_pose, "potted_meat_can", num_env, 0)
        chef_can_handle = self.gym.create_actor(env, self.chef_can_asset, self.chef_can_pose, "chef_can", num_env, 0)
        tuna_can_handle = self.gym.create_actor(env, self.tuna_can_asset, self.tuna_can_pose, "tuna_can", num_env, 0)
        plum_handle = self.gym.create_actor(env, self.plum_asset, self.plum_pose, "plum", num_env, 0)
        strawberry_handle = self.gym.create_actor(env, self.strawberry_asset, self.strawberry_pose, "strawberry", num_env, 0)
        pear_handle = self.gym.create_actor(env, self.pear_asset, self.pear_pose, "pear", num_env, 0)
        mustard_handle = self.gym.create_actor(env, self.mustard_asset, self.mustard_pose, "mustard", num_env, 0)
        tomato_soup_can2_handle = self.gym.create_actor(env, self.tomato_soup_can2_asset, self.tomato_soup_can2_pose, "tomato_soup_can2", num_env, 0)
        self.gym.set_actor_scale(env, tomato_soup_can2_handle, 1.5)
        tuna_can2_handle = self.gym.create_actor(env, self.tuna_can2_asset, self.tuna_can2_pose, "tuna_can2", num_env, 0)
        potted_meat_can2_handle = self.gym.create_actor(env, self.potted_meat_can2_asset, self.potted_meat_can2_pose, "potted_meat_can2", num_env, 0)
        gelatin_box_handle = self.gym.create_actor(env, self.gelatin_box_asset, self.gelatin_box_pose, "gelatin_box", num_env, 0)
        cracker_handle = self.gym.create_actor(env, self.cracker_box_asset, self.cracker_pose, "cracker_box", num_env, 0)
        sugar_box_handle = self.gym.create_actor(env, self.sugar_box_asset, self.sugar_box_pose, "sugar_box", num_env, 0)
        pudding_box_handle = self.gym.create_actor(env, self.pudding_box_asset, self.pudding_box_pose, "pudding_box", num_env, 0)
        yellow_bowl_handle = self.gym.create_actor(env, self.yellow_bowl_asset, self.yellow_bowl_pose, "yellow_bowl", num_env, 0)
        bowl_designed_handle = self.gym.create_actor(env, self.bowl_designed_asset, self.bowl_designed_pose, "bowl_designed", num_env, 0)
        sauce1_handle = self.gym.create_actor(env, self.sauce1_asset, self.sauce1_pose, "sauce1", num_env, 0)
        sauce2_handle = self.gym.create_actor(env, self.sauce2_asset, self.sauce2_pose, "sauce2", num_env, 0)

class Env3(EnvBase):
    def set_robot_asset(self):
        # config UR5e asset
        print("robot asset is set")

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

        sink_file = "urdf/kitchen/KitchenCountertop/model_white.urdf"
        self.sink_asset = self.gym.load_asset(self.sim, self.asset_root, sink_file, simple_options)
        self.sink_pose = gymapi.Transform(p=gymapi.Vec3(2.7-0.125,1.93-0.15,0.0),r=gymapi.Quat(0, 0.7071068, 0.7071068, 0))

        fridge_file = "urdf/kitchen/Refrigerator/model_roughness.urdf"
        self.fridge_asset = self.gym.load_asset(self.sim, self.asset_root, fridge_file, simple_options)
        self.fridge_pose = gymapi.Transform(r=gymapi.Quat(0, 0.7071068, 0.7071068, 0),p=gymapi.Vec3(2.6,0.25,0.0))

        desk_file = "urdf/kitchen/kitchen_table/model.urdf"
        self.desk_asset = self.gym.load_asset(self.sim, self.asset_root, desk_file, simple_options)
        self.desk_pose = gymapi.Transform(r=gymapi.Quat(0.7071068, 0, 0, 0.7071068),p=gymapi.Vec3(0.37,1.8,0.0))

        drainer_file = "urdf/kitchen/dish_drainer/model.urdf"
        self.drainer_asset = self.gym.load_asset(self.sim, self.asset_root, drainer_file, simple_options)
        self.drainer_pose = gymapi.Transform(r=gymapi.Quat(0, 0, 0.7071068, 0.7071068),p=gymapi.Vec3(2.5,1.4,0.74))

        chair_file = "urdf/kitchen/WoodenChair/model.urdf"
        self.chair_asset = self.gym.load_asset(self.sim, self.asset_root, chair_file, simple_options)
        self.chair_pose = gymapi.Transform(p=gymapi.Vec3(0.9,2.0,0.0),r=gymapi.Quat(0.0,0.0,1.0,0.0))

        drawer_file = "urdf/kitchen/drawer/model.urdf"
        self.drawer_asset = self.gym.load_asset(self.sim, self.asset_root, drawer_file, simple_options)
        self.drawer_pose = gymapi.Transform(p=gymapi.Vec3(0.25,0.6+0.6+1.0,0.8),r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        mugcup_file = "urdf/kitchen/mugcup/model.urdf"
        self.mugcup_asset = self.gym.load_asset(self.sim, self.asset_root, mugcup_file, simple_options)
        self.mugcup_pose = gymapi.Transform(p=gymapi.Vec3(0.25,0.75+0.6+1.0,0.82),r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        coffeemug_file = "urdf/kitchen/coffeemug/model.urdf"
        self.coffeemug_asset = self.gym.load_asset(self.sim, self.asset_root, coffeemug_file, simple_options)
        self.coffeemug_pose = gymapi.Transform(p=gymapi.Vec3(0.25,0.6+0.6+1.0,0.82),r=gymapi.Quat(0, 0, 0.0871557, 0.9961947))

        coffeemug2_file = "urdf/kitchen/coffeemug/model.urdf"
        self.coffeemug2_asset = self.gym.load_asset(self.sim, self.asset_root, coffeemug2_file, simple_options)
        self.coffeemug2_pose = gymapi.Transform(p=gymapi.Vec3(0.25,0.45+0.6+1.0,0.82),r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        frypan_file = "urdf/kitchen/frypan/model.urdf"
        self.frypan_asset = self.gym.load_asset(self.sim, self.asset_root, frypan_file, simple_options)
        self.frypan_pose = gymapi.Transform(r=gymapi.Quat(0, 0, 0.9961947, 0.0871557),p=gymapi.Vec3(2.38,2.41,0.76))

        shelf_file = "urdf/kitchen/SquareShelf/model.urdf"
        self.shelf_asset = self.gym.load_asset(self.sim, self.asset_root, shelf_file, simple_options)
        self.shelf_pose = gymapi.Transform(r=gymapi.Quat(0, 0, -0.7071068, 0.7071068),p=gymapi.Vec3(2.7,2.41,1.3))

        cabinet_file = "urdf/kitchen/WhiteCabinet/model.urdf"
        self.cabinet1_asset = self.gym.load_asset(self.sim, self.asset_root, cabinet_file, simple_options)
        self.cabinet1_pose = gymapi.Transform(r=gymapi.Quat(0, 0, -0.7071068, 0.7071068),p=gymapi.Vec3(2.9,2.7,1.2))

        self.cabinet2_asset = self.gym.load_asset(self.sim, self.asset_root, cabinet_file, simple_options)
        self.cabinet2_pose = gymapi.Transform(r=gymapi.Quat(0, 0, -0.7071068, 0.7071068),p=gymapi.Vec3(2.9,2.7-0.28,1.2))

        self.cabinet3_asset = self.gym.load_asset(self.sim, self.asset_root, cabinet_file, simple_options)
        self.cabinet3_pose = gymapi.Transform(r=gymapi.Quat(0, 0, -0.7071068, 0.7071068),p=gymapi.Vec3(2.9,2.7-0.28-0.28,1.2))

        cooker_file = "urdf/kitchen/cooker/model.urdf"
        self.cooker_asset = self.gym.load_asset(self.sim, self.asset_root, cooker_file, simple_options)
        self.cooker_pose = gymapi.Transform(r=gymapi.Quat(0, 0, -0.5, 0.8660254),p=gymapi.Vec3(2.75,1.8,0.74))

        # set random position for other objects
        y_lim = [0.95,1.9]
        x_lim = [0.1,0.6]
        num_objects = 18
        random_x, random_y = get_object_positions(num_objects, x_lim, y_lim)

        tomato_soup_can_file = "urdf/kitchen/005_tomato_soup_can/model.urdf"
        self.tomato_soup_can_asset = self.gym.load_asset(self.sim, self.asset_root, tomato_soup_can_file, gravity_options)
        self.tomato_soup_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[0],random_y[0], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        potted_meat_can_file = "urdf/kitchen/010_potted_meat_can/model.urdf"
        self.potted_meat_can_asset = self.gym.load_asset(self.sim, self.asset_root, potted_meat_can_file, gravity_options)
        self.potted_meat_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[1],random_y[1], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        chef_can_file = "urdf/ycb/002_master_chef_can/model.urdf"
        self.chef_can_asset = self.gym.load_asset(self.sim, self.asset_root, chef_can_file, gravity_options)
        self.chef_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[2],random_y[2], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        tuna_can_file = "urdf/ycb/007_tuna_fish_can/model.urdf"
        self.tuna_can_asset = self.gym.load_asset(self.sim, self.asset_root, tuna_can_file, gravity_options)
        self.tuna_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[3],random_y[3], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        plum_file = "urdf/ycb/018_plum/model.urdf"
        self.plum_asset = self.gym.load_asset(self.sim, self.asset_root, plum_file, gravity_options)
        self.plum_pose = gymapi.Transform(p=gymapi.Vec3(random_x[4],random_y[4], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        strawberry_file = "urdf/ycb/012_strawberry/model.urdf"
        self.strawberry_asset = self.gym.load_asset(self.sim, self.asset_root, strawberry_file, gravity_options)
        self.strawberry_pose = gymapi.Transform(p=gymapi.Vec3(random_x[5],random_y[5], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        pear_file = "urdf/ycb/016_pear/model.urdf"
        self.pear_asset = self.gym.load_asset(self.sim, self.asset_root, pear_file, gravity_options)
        self.pear_pose = gymapi.Transform(p=gymapi.Vec3(random_x[6],random_y[6], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        mustard_file = "urdf/ycb/006_mustard_bottle/model.urdf"
        self.mustard_asset = self.gym.load_asset(self.sim, self.asset_root, mustard_file, gravity_options)
        self.mustard_pose = gymapi.Transform(p=gymapi.Vec3(random_x[7],random_y[7], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        self.tomato_soup_can2_asset = self.gym.load_asset(self.sim, self.asset_root, tomato_soup_can_file, gravity_options)
        self.tomato_soup_can2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[8],random_y[8], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        self.tuna_can2_asset = self.gym.load_asset(self.sim, self.asset_root, tuna_can_file, gravity_options)
        self.tuna_can2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[9],random_y[9], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        self.potted_meat_can2_asset = self.gym.load_asset(self.sim, self.asset_root, potted_meat_can_file, gravity_options)
        self.potted_meat_can2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[10],random_y[10], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        gelatin_box_file = "urdf/ycb/009_gelatin_box/model.urdf"
        self.gelatin_box_asset = self.gym.load_asset(self.sim, self.asset_root, gelatin_box_file, gravity_options)
        self.gelatin_box_pose = gymapi.Transform(p=gymapi.Vec3(random_x[11],random_y[11], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        cracker_box_file = "urdf/ycb/003_cracker_box/model.urdf"
        self.cracker_box_asset = self.gym.load_asset(self.sim, self.asset_root, cracker_box_file, gravity_options)
        self.cracker_pose = gymapi.Transform(p=gymapi.Vec3(random_x[12],random_y[12], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        sugar_box_file = "urdf/ycb/004_sugar_box/model.urdf"
        self.sugar_box_asset = self.gym.load_asset(self.sim, self.asset_root, sugar_box_file, gravity_options)
        self.sugar_box_pose = gymapi.Transform(p=gymapi.Vec3(random_x[12],random_y[12], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        pudding_box_file = "urdf/ycb/008_pudding_box/model.urdf"
        self.pudding_box_asset = self.gym.load_asset(self.sim, self.asset_root, pudding_box_file, gravity_options)
        self.pudding_box_pose = gymapi.Transform(p=gymapi.Vec3(random_x[13],random_y[13], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        yellow_bowl_file = "urdf/kitchen/yellow_bowl/model.urdf"
        self.yellow_bowl_asset = self.gym.load_asset(self.sim, self.asset_root, yellow_bowl_file, gravity_options)
        self.yellow_bowl_pose = gymapi.Transform(p=gymapi.Vec3(random_x[14],random_y[14],0.835))

        bowl_designed_file = "urdf/kitchen/bowl_designed/model.urdf"
        self.bowl_designed_asset = self.gym.load_asset(self.sim, self.asset_root, bowl_designed_file, gravity_options)
        self.bowl_designed_pose = gymapi.Transform(p=gymapi.Vec3(random_x[15],random_y[15],0.835))

        sauce1_file = "urdf/kitchen/sauce1/model.urdf"
        self.sauce1_asset = self.gym.load_asset(self.sim, self.asset_root, sauce1_file, gravity_options)
        self.sauce1_pose = gymapi.Transform(p=gymapi.Vec3(random_x[16],random_y[16],0.835))

        sauce2_file = "urdf/kitchen/sauce2/model.urdf"
        self.sauce2_asset = self.gym.load_asset(self.sim, self.asset_root, sauce2_file, gravity_options)
        self.sauce2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[17],random_y[17],0.835))

    def set_object_asset(self):
        print("object asset is set")
    
    def create_robot(self, env, num_env):
        print("robot is created")

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
        sink_handle = self.gym.create_actor(env, self.sink_asset, self.sink_pose, "sink", num_env, 0)
        fridge_handle = self.gym.create_actor(env, self.fridge_asset, self.fridge_pose, "fridge", num_env, 0)
        desk_handle = self.gym.create_actor(env, self.desk_asset, self.desk_pose, "desk", num_env, 0)
        drainer_handle = self.gym.create_actor(env, self.drainer_asset, self.drainer_pose, "drainer", num_env, 0)
        chair_handle = self.gym.create_actor(env, self.chair_asset, self.chair_pose, "chair", num_env, 0)
        drawer_handle = self.gym.create_actor(env, self.drawer_asset, self.drawer_pose, "drawer", num_env, 0)
        self.gym.set_actor_scale(env, drawer_handle, 1.5)
        mugcup_handle = self.gym.create_actor(env, self.mugcup_asset, self.mugcup_pose, "mugcup", num_env, 0)
        coffeemug_handle = self.gym.create_actor(env, self.coffeemug_asset, self.coffeemug_pose, "coffeemug", num_env, 0)
        coffeemug2_handle = self.gym.create_actor(env, self.coffeemug2_asset, self.coffeemug2_pose, "coffeemug2", num_env, 0)
        frypan_handle = self.gym.create_actor(env, self.frypan_asset, self.frypan_pose, "frypan", num_env, 0)
        # shelf_handle = self.gym.create_actor(env, self.shelf_asset, self.shelf_pose, "shelf", num_env, 0)
        cabinet1_handle = self.gym.create_actor(env, self.cabinet1_asset, self.cabinet1_pose, "cabinet", num_env, 0)
        self.gym.set_actor_scale(env, cabinet1_handle, 0.3)
        cabinet2_handle = self.gym.create_actor(env, self.cabinet2_asset, self.cabinet2_pose, "cabinet", num_env, 0)
        self.gym.set_actor_scale(env, cabinet2_handle, 0.3)
        cabinet3_handle = self.gym.create_actor(env, self.cabinet3_asset, self.cabinet3_pose, "cabinet", num_env, 0)
        self.gym.set_actor_scale(env, cabinet3_handle, 0.3)
        cooker_handle = self.gym.create_actor(env, self.cooker_asset, self.cooker_pose, "cooker", num_env, 0)
        self.gym.set_actor_scale(env, cooker_handle, 0.8)

        tomato_soup_can_handle = self.gym.create_actor(env, self.tomato_soup_can_asset, self.tomato_soup_can_pose, "tomato_soup_can", num_env, 0)
        self.gym.set_actor_scale(env, tomato_soup_can_handle, 1.5)
        potted_meat_can_handle = self.gym.create_actor(env, self.potted_meat_can_asset, self.potted_meat_can_pose, "potted_meat_can", num_env, 0)
        chef_can_handle = self.gym.create_actor(env, self.chef_can_asset, self.chef_can_pose, "chef_can", num_env, 0)
        tuna_can_handle = self.gym.create_actor(env, self.tuna_can_asset, self.tuna_can_pose, "tuna_can", num_env, 0)
        plum_handle = self.gym.create_actor(env, self.plum_asset, self.plum_pose, "plum", num_env, 0)
        strawberry_handle = self.gym.create_actor(env, self.strawberry_asset, self.strawberry_pose, "strawberry", num_env, 0)
        pear_handle = self.gym.create_actor(env, self.pear_asset, self.pear_pose, "pear", num_env, 0)
        mustard_handle = self.gym.create_actor(env, self.mustard_asset, self.mustard_pose, "mustard", num_env, 0)
        tomato_soup_can2_handle = self.gym.create_actor(env, self.tomato_soup_can2_asset, self.tomato_soup_can2_pose, "tomato_soup_can2", num_env, 0)
        self.gym.set_actor_scale(env, tomato_soup_can2_handle, 1.5)
        tuna_can2_handle = self.gym.create_actor(env, self.tuna_can2_asset, self.tuna_can2_pose, "tuna_can2", num_env, 0)
        potted_meat_can2_handle = self.gym.create_actor(env, self.potted_meat_can2_asset, self.potted_meat_can2_pose, "potted_meat_can2", num_env, 0)
        gelatin_box_handle = self.gym.create_actor(env, self.gelatin_box_asset, self.gelatin_box_pose, "gelatin_box", num_env, 0)
        cracker_handle = self.gym.create_actor(env, self.cracker_box_asset, self.cracker_pose, "cracker_box", num_env, 0)
        sugar_box_handle = self.gym.create_actor(env, self.sugar_box_asset, self.sugar_box_pose, "sugar_box", num_env, 0)
        pudding_box_handle = self.gym.create_actor(env, self.pudding_box_asset, self.pudding_box_pose, "pudding_box", num_env, 0)
        yellow_bowl_handle = self.gym.create_actor(env, self.yellow_bowl_asset, self.yellow_bowl_pose, "yellow_bowl", num_env, 0)
        bowl_designed_handle = self.gym.create_actor(env, self.bowl_designed_asset, self.bowl_designed_pose, "bowl_designed", num_env, 0)
        sauce1_handle = self.gym.create_actor(env, self.sauce1_asset, self.sauce1_pose, "sauce1", num_env, 0)
        sauce2_handle = self.gym.create_actor(env, self.sauce2_asset, self.sauce2_pose, "sauce2", num_env, 0)

class Env4(EnvBase):
    def set_robot_asset(self):
        # config UR5e asset
        print("robot asset is set")

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
        self.floor_pose13 = gymapi.Transform(p=gymapi.Vec3(3.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose14 = gymapi.Transform(p=gymapi.Vec3(3.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose15 = gymapi.Transform(p=gymapi.Vec3(3.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose16 = gymapi.Transform(p=gymapi.Vec3(3.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))

        self.wall_asset_1 = self.gym.create_box(self.sim,0.1,4.0,3.0,simple_options)
        self.wall1_pose = gymapi.Transform(p=gymapi.Vec3(0.0,1.0,1.5))
        self.wall3_pose = gymapi.Transform(p=gymapi.Vec3(4.0,1.0,1.5))
        self.wall_asset_2 = self.gym.create_box(self.sim,4.0,0.1,3.0,simple_options)
        self.wall2_pose = gymapi.Transform(p=gymapi.Vec3(2.0,3.0,1.5))

        oven_file = "urdf/kitchen/Oven/model.urdf"
        self.oven_asset = self.gym.load_asset(self.sim, self.asset_root, oven_file, simple_options)
        self.oven_pose = gymapi.Transform(p=gymapi.Vec3(3.13,2.50,0.0),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), 0.0))

        fridge_file = "urdf/kitchen/Refrigerator/model_albedo.urdf"
        self.fridge_asset = self.gym.load_asset(self.sim, self.asset_root, fridge_file, simple_options)
        self.fridge_pose = gymapi.Transform(p=gymapi.Vec3(3.45,0.1,0.0), r=gymapi.Quat(0, 0.7071068, 0.7071068, 0))

        kitchen_cabinet_file = "urdf/kitchen/KitchenCabinet/model.urdf"
        kitchen_cabinet_file_options = gymapi.AssetOptions()
        kitchen_cabinet_file_options.fix_base_link = True
        kitchen_cabinet_file_options.vhacd_enabled = True
        kitchen_cabinet_file_options.vhacd_params = gymapi.VhacdParams()
        kitchen_cabinet_file_options.vhacd_params.resolution = 300000
        kitchen_cabinet_file_options.vhacd_params.max_convex_hulls = 512
        kitchen_cabinet_file_options.vhacd_params.convex_hull_approximation = False
        self.kitchen_cabinet_file_asset = self.gym.load_asset(self.sim, self.asset_root, kitchen_cabinet_file, kitchen_cabinet_file_options)
        self.kitchen_cabinet_file_pose = gymapi.Transform(p=gymapi.Vec3(1.37,1.35,0.0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), np.pi/2))

        utensil_file = "urdf/kitchen/KitchenUtensils/model.urdf"
        self.utensil_asset = self.gym.load_asset(self.sim, self.asset_root, utensil_file, simple_options)
        self.utensil_pose = gymapi.Transform(p=gymapi.Vec3(0.09, 1.6, 1.1),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), np.pi/2))

        rangehood_file = "urdf/kitchen/rangehood/model.urdf"
        self.rangehood_asset = self.gym.load_asset(self.sim, self.asset_root, rangehood_file, simple_options)
        self.rangehood_pose = gymapi.Transform(p=gymapi.Vec3(3.13,2.7,1.8),r=gymapi.Quat(0.5, 0.5, 0.5, 0.5))

        desk_file = "urdf/kitchen/kitchen_table/model_white_marble.urdf"
        self.desk_asset = self.gym.load_asset(self.sim, self.asset_root, desk_file, simple_options)
        self.desk_pose = gymapi.Transform(p=gymapi.Vec3(2.3,0.9,0.0), r=gymapi.Quat(0.5, 0.5, 0.5, 0.5))

        drawer_file = "urdf/kitchen/drawer/model.urdf"
        self.drawer_asset = self.gym.load_asset(self.sim, self.asset_root, drawer_file, simple_options)
        self.drawer_pose = gymapi.Transform(p=gymapi.Vec3(1.55,0.6+0.35,0.8),r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        vitamin_file = "urdf/kitchen/Vitamin/model.urdf"
        self.vitamin_asset = self.gym.load_asset(self.sim, self.asset_root, vitamin_file, simple_options)
        self.vitamin_pose = gymapi.Transform(p=gymapi.Vec3(1.55,0.75+0.35,0.82),r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        germanium_file = "urdf/kitchen/Germanium/model.urdf"
        self.germanium_asset = self.gym.load_asset(self.sim, self.asset_root, germanium_file, simple_options)
        self.germanium_pose = gymapi.Transform(p=gymapi.Vec3(1.6,0.6+0.31,0.82),r=gymapi.Quat(0, 0, 0.0871557, 0.9961947))

        xylitol_file = "urdf/kitchen/xylitol/model.urdf"
        self.xylitol_asset = self.gym.load_asset(self.sim, self.asset_root, xylitol_file, simple_options)
        self.xylitol_pose = gymapi.Transform(p=gymapi.Vec3(1.58,0.45+0.35,0.82),r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        vitamin2_file = "urdf/kitchen/Vitamin2/model.urdf"
        self.vitamin2_asset = self.gym.load_asset(self.sim, self.asset_root, vitamin2_file, simple_options)
        self.vitamin2_pose = gymapi.Transform(p=gymapi.Vec3(1.53,0.65+0.35,0.82),r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        vitamin3_file = "urdf/kitchen/Vitamin3/model.urdf"
        self.vitamin3_asset = self.gym.load_asset(self.sim, self.asset_root, vitamin3_file, simple_options)
        self.vitamin3_pose = gymapi.Transform(p=gymapi.Vec3(1.6,0.65+0.35,0.82),r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        vitamin4_file = "urdf/kitchen/Vitamin4/model.urdf"
        self.vitamin4_asset = self.gym.load_asset(self.sim, self.asset_root, vitamin4_file, simple_options)
        self.vitamin4_pose = gymapi.Transform(p=gymapi.Vec3(1.53,0.6+0.31,0.82),r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        vitamin5_file = "urdf/kitchen/Vitamin5/model.urdf"
        self.vitamin5_asset = self.gym.load_asset(self.sim, self.asset_root, vitamin5_file, simple_options)
        self.vitamin5_pose = gymapi.Transform(p=gymapi.Vec3(1.49,0.45+0.37,0.82),r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        tissue_file = "urdf/kitchen/tissue/model.urdf"
        self.tissue_asset = self.gym.load_asset(self.sim, self.asset_root, tissue_file, simple_options)
        self.tissue_pose = gymapi.Transform(p=gymapi.Vec3(1.7,0.65,0.8),r=gymapi.Quat(0, 0, 0, 1))

        rubber_gloves_file = "urdf/kitchen/rubber_gloves/model.urdf"
        self.rubber_gloves_asset = self.gym.load_asset(self.sim, self.asset_root, rubber_gloves_file, simple_options)
        self.rubber_gloves_pose = gymapi.Transform(p=gymapi.Vec3(1.8,2.6,0.89),r=gymapi.Quat(0, 0, 0, 1))

        drainer_file = "urdf/kitchen/drainer/model.urdf"
        self.drainer_asset = self.gym.load_asset(self.sim, self.asset_root, drainer_file, simple_options)
        self.drainer_pose = gymapi.Transform(r=gymapi.Quat(0, 0, 0.7071068, 0.7071068),p=gymapi.Vec3(0.45,2.7,0.89))

        cereal_box_file = "urdf/kitchen/cereal_box/model.urdf"
        self.cereal_box_asset = self.gym.load_asset(self.sim, self.asset_root, cereal_box_file, simple_options)
        self.cereal_box_pose = gymapi.Transform(p=gymapi.Vec3(0.3,0.5,0.89),r=gymapi.Quat(0,0,0,1))

        nesquik_box_file = "urdf/kitchen/nesquik_box/model.urdf"
        self.nesquik_box_asset = self.gym.load_asset(self.sim, self.asset_root, nesquik_box_file, simple_options)
        self.nesquik_box_pose = gymapi.Transform(p=gymapi.Vec3(0.3,0.41,0.89),r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        cabinet_file = "urdf/kitchen/WhiteCabinet/model_white_marble.urdf"
        self.cabinet_asset = self.gym.load_asset(self.sim, self.asset_root, cabinet_file, simple_options)
        self.cabinet1_pose = gymapi.Transform(r=gymapi.Quat(0, 0, 0.7071068, 0.7071068),p=gymapi.Vec3(0.17,0.15,1.5))
        self.cabinet2_pose = gymapi.Transform(r=gymapi.Quat(0, 0, 0.7071068, 0.7071068),p=gymapi.Vec3(0.17,0.15+0.37,1.5))
        self.cabinet3_pose = gymapi.Transform(r=gymapi.Quat(0, 0, 0.7071068, 0.7071068),p=gymapi.Vec3(0.17,0.15+0.37+0.37,1.5))

        # set random position for other objects
        y_lim = [0.65,1.2]
        x_lim = [1.74,3.0]
        num_objects = 18
        random_x, random_y = get_object_positions(num_objects, x_lim, y_lim)

        tomato_soup_can_file = "urdf/kitchen/005_tomato_soup_can/model.urdf"
        self.tomato_soup_can_asset = self.gym.load_asset(self.sim, self.asset_root, tomato_soup_can_file, gravity_options)
        self.tomato_soup_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[0],random_y[0], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        potted_meat_can_file = "urdf/kitchen/010_potted_meat_can/model.urdf"
        self.potted_meat_can_asset = self.gym.load_asset(self.sim, self.asset_root, potted_meat_can_file, gravity_options)
        self.potted_meat_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[1],random_y[1], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        chef_can_file = "urdf/ycb/002_master_chef_can/model.urdf"
        self.chef_can_asset = self.gym.load_asset(self.sim, self.asset_root, chef_can_file, gravity_options)
        self.chef_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[2],random_y[2], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        tuna_can_file = "urdf/ycb/007_tuna_fish_can/model.urdf"
        self.tuna_can_asset = self.gym.load_asset(self.sim, self.asset_root, tuna_can_file, gravity_options)
        self.tuna_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[3],random_y[3], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        plum_file = "urdf/ycb/018_plum/model.urdf"
        self.plum_asset = self.gym.load_asset(self.sim, self.asset_root, plum_file, gravity_options)
        self.plum_pose = gymapi.Transform(p=gymapi.Vec3(random_x[4],random_y[4], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        strawberry_file = "urdf/ycb/012_strawberry/model.urdf"
        self.strawberry_asset = self.gym.load_asset(self.sim, self.asset_root, strawberry_file, gravity_options)
        self.strawberry_pose = gymapi.Transform(p=gymapi.Vec3(random_x[5],random_y[5], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        pear_file = "urdf/ycb/016_pear/model.urdf"
        self.pear_asset = self.gym.load_asset(self.sim, self.asset_root, pear_file, gravity_options)
        self.pear_pose = gymapi.Transform(p=gymapi.Vec3(random_x[6],random_y[6], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        mustard_file = "urdf/ycb/006_mustard_bottle/model.urdf"
        self.mustard_asset = self.gym.load_asset(self.sim, self.asset_root, mustard_file, gravity_options)
        self.mustard_pose = gymapi.Transform(p=gymapi.Vec3(random_x[7],random_y[7], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        self.tomato_soup_can2_asset = self.gym.load_asset(self.sim, self.asset_root, tomato_soup_can_file, gravity_options)
        self.tomato_soup_can2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[8],random_y[8], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        self.tuna_can2_asset = self.gym.load_asset(self.sim, self.asset_root, tuna_can_file, gravity_options)
        self.tuna_can2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[9],random_y[9], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        self.potted_meat_can2_asset = self.gym.load_asset(self.sim, self.asset_root, potted_meat_can_file, gravity_options)
        self.potted_meat_can2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[10],random_y[10], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        gelatin_box_file = "urdf/ycb/009_gelatin_box/model.urdf"
        self.gelatin_box_asset = self.gym.load_asset(self.sim, self.asset_root, gelatin_box_file, gravity_options)
        self.gelatin_box_pose = gymapi.Transform(p=gymapi.Vec3(random_x[11],random_y[11], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        cracker_box_file = "urdf/ycb/003_cracker_box/model.urdf"
        self.cracker_box_asset = self.gym.load_asset(self.sim, self.asset_root, cracker_box_file, gravity_options)
        self.cracker_pose = gymapi.Transform(p=gymapi.Vec3(random_x[12],random_y[12], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        sugar_box_file = "urdf/ycb/004_sugar_box/model.urdf"
        self.sugar_box_asset = self.gym.load_asset(self.sim, self.asset_root, sugar_box_file, gravity_options)
        self.sugar_box_pose = gymapi.Transform(p=gymapi.Vec3(random_x[12],random_y[12], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        pudding_box_file = "urdf/ycb/008_pudding_box/model.urdf"
        self.pudding_box_asset = self.gym.load_asset(self.sim, self.asset_root, pudding_box_file, gravity_options)
        self.pudding_box_pose = gymapi.Transform(p=gymapi.Vec3(random_x[13],random_y[13], 0.835),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        yellow_bowl_file = "urdf/kitchen/yellow_bowl/model.urdf"
        self.yellow_bowl_asset = self.gym.load_asset(self.sim, self.asset_root, yellow_bowl_file, gravity_options)
        self.yellow_bowl_pose = gymapi.Transform(p=gymapi.Vec3(random_x[14],random_y[14],0.835))

        bowl_designed_file = "urdf/kitchen/bowl_designed/model.urdf"
        self.bowl_designed_asset = self.gym.load_asset(self.sim, self.asset_root, bowl_designed_file, gravity_options)
        self.bowl_designed_pose = gymapi.Transform(p=gymapi.Vec3(random_x[15],random_y[15],0.835))

        sauce1_file = "urdf/kitchen/sauce1/model.urdf"
        self.sauce1_asset = self.gym.load_asset(self.sim, self.asset_root, sauce1_file, gravity_options)
        self.sauce1_pose = gymapi.Transform(p=gymapi.Vec3(random_x[16],random_y[16],0.835))

        sauce2_file = "urdf/kitchen/sauce2/model.urdf"
        self.sauce2_asset = self.gym.load_asset(self.sim, self.asset_root, sauce2_file, gravity_options)
        self.sauce2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[17],random_y[17],0.835))

    def set_object_asset(self):
        print("object asset is set")

    def create_robot(self, env, num_env):
        print("robot is created")

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
        fridge_handle = self.gym.create_actor(env, self.fridge_asset, self.fridge_pose, "fridge", num_env, 0)
        kitchen_cabinet_file_handle = self.gym.create_actor(env, self.kitchen_cabinet_file_asset, self.kitchen_cabinet_file_pose, "kitchen_cabinet", num_env, 0)
        utensil_handle = self.gym.create_actor(env, self.utensil_asset, self.utensil_pose, "utensil", num_env, 0)
        oven_handle = self.gym.create_actor(env, self.oven_asset, self.oven_pose, "oven", num_env, 0)
        rangehood_handle = self.gym.create_actor(env, self.rangehood_asset, self.rangehood_pose, "rangehood", num_env, 0)
        desk_handle = self.gym.create_actor(env, self.desk_asset, self.desk_pose, "desk", num_env, 0)
        drawer_handle = self.gym.create_actor(env, self.drawer_asset, self.drawer_pose, "drawer", num_env, 0)
        self.gym.set_actor_scale(env, drawer_handle, 1.5)
        vitamin_handle = self.gym.create_actor(env, self.vitamin_asset, self.vitamin_pose, "vitamin", num_env, 0)
        germanium_handle = self.gym.create_actor(env, self.germanium_asset, self.germanium_pose, "germanium", num_env, 0)
        xylitol_handle = self.gym.create_actor(env, self.xylitol_asset, self.xylitol_pose, "xylitol", num_env, 0)
        vitamin2_handle = self.gym.create_actor(env, self.vitamin2_asset, self.vitamin2_pose, "vitamin2", num_env, 0)
        vitamin3_handle = self.gym.create_actor(env, self.vitamin3_asset, self.vitamin3_pose, "vitamin3", num_env, 0)
        vitamin4_handle = self.gym.create_actor(env, self.vitamin4_asset, self.vitamin4_pose, "vitamin4", num_env, 0)
        vitamin5_handle = self.gym.create_actor(env, self.vitamin5_asset, self.vitamin5_pose, "vitamin5", num_env, 0)
        tissue_handle = self.gym.create_actor(env, self.tissue_asset, self.tissue_pose, "tissue", num_env, 0)
        self.gym.set_actor_scale(env, tissue_handle, 0.8)
        rubber_gloves_handle = self.gym.create_actor(env, self.rubber_gloves_asset, self.rubber_gloves_pose, "rubber_gloves", num_env, 0)
        drainer_handle = self.gym.create_actor(env, self.drainer_asset, self.drainer_pose, "drainer", num_env, 0)
        self.gym.set_actor_scale(env, drainer_handle, 1.3)
        cereal_box_handle = self.gym.create_actor(env, self.cereal_box_asset, self.cereal_box_pose, "cereal_box", num_env, 0)
        nesquik_box_handle = self.gym.create_actor(env, self.nesquik_box_asset, self.nesquik_box_pose, "nesquik_box", num_env, 0)
        cabinet1_handle = self.gym.create_actor(env, self.cabinet_asset, self.cabinet1_pose, "cabinet", num_env, 0)
        cabinet2_handle = self.gym.create_actor(env, self.cabinet_asset, self.cabinet2_pose, "cabinet", num_env, 0)
        cabinet3_handle = self.gym.create_actor(env, self.cabinet_asset, self.cabinet3_pose, "cabinet", num_env, 0)
        self.gym.set_actor_scale(env, cabinet1_handle, 0.4)
        self.gym.set_actor_scale(env, cabinet2_handle, 0.4)
        self.gym.set_actor_scale(env, cabinet3_handle, 0.4)
        tomato_soup_can_handle = self.gym.create_actor(env, self.tomato_soup_can_asset, self.tomato_soup_can_pose, "tomato_soup_can", num_env, 0)
        self.gym.set_actor_scale(env, tomato_soup_can_handle, 1.5)
        potted_meat_can_handle = self.gym.create_actor(env, self.potted_meat_can_asset, self.potted_meat_can_pose, "potted_meat_can", num_env, 0)
        chef_can_handle = self.gym.create_actor(env, self.chef_can_asset, self.chef_can_pose, "chef_can", num_env, 0)
        tuna_can_handle = self.gym.create_actor(env, self.tuna_can_asset, self.tuna_can_pose, "tuna_can", num_env, 0)
        plum_handle = self.gym.create_actor(env, self.plum_asset, self.plum_pose, "plum", num_env, 0)
        strawberry_handle = self.gym.create_actor(env, self.strawberry_asset, self.strawberry_pose, "strawberry", num_env, 0)
        pear_handle = self.gym.create_actor(env, self.pear_asset, self.pear_pose, "pear", num_env, 0)
        mustard_handle = self.gym.create_actor(env, self.mustard_asset, self.mustard_pose, "mustard", num_env, 0)
        tomato_soup_can2_handle = self.gym.create_actor(env, self.tomato_soup_can2_asset, self.tomato_soup_can2_pose, "tomato_soup_can2", num_env, 0)
        self.gym.set_actor_scale(env, tomato_soup_can2_handle, 1.5)
        tuna_can2_handle = self.gym.create_actor(env, self.tuna_can2_asset, self.tuna_can2_pose, "tuna_can2", num_env, 0)
        potted_meat_can2_handle = self.gym.create_actor(env, self.potted_meat_can2_asset, self.potted_meat_can2_pose, "potted_meat_can2", num_env, 0)
        gelatin_box_handle = self.gym.create_actor(env, self.gelatin_box_asset, self.gelatin_box_pose, "gelatin_box", num_env, 0)
        cracker_handle = self.gym.create_actor(env, self.cracker_box_asset, self.cracker_pose, "cracker_box", num_env, 0)
        sugar_box_handle = self.gym.create_actor(env, self.sugar_box_asset, self.sugar_box_pose, "sugar_box", num_env, 0)
        pudding_box_handle = self.gym.create_actor(env, self.pudding_box_asset, self.pudding_box_pose, "pudding_box", num_env, 0)
        yellow_bowl_handle = self.gym.create_actor(env, self.yellow_bowl_asset, self.yellow_bowl_pose, "yellow_bowl", num_env, 0)
        bowl_designed_handle = self.gym.create_actor(env, self.bowl_designed_asset, self.bowl_designed_pose, "bowl_designed", num_env, 0)
        sauce1_handle = self.gym.create_actor(env, self.sauce1_asset, self.sauce1_pose, "sauce1", num_env, 0)
        sauce2_handle = self.gym.create_actor(env, self.sauce2_asset, self.sauce2_pose, "sauce2", num_env, 0)

class Env5(EnvBase):
    def set_robot_asset(self):
        # config UR5e asset
        print("robot asset is set")

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
        self.floor_pose13 = gymapi.Transform(p=gymapi.Vec3(3.5,-0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose14 = gymapi.Transform(p=gymapi.Vec3(3.5,0.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose15 = gymapi.Transform(p=gymapi.Vec3(3.5,1.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))
        self.floor_pose16 = gymapi.Transform(p=gymapi.Vec3(3.5,2.5,0.01),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))

        self.wall_asset_1 = self.gym.create_box(self.sim,0.1,4.0,3.0,simple_options)
        self.wall1_pose = gymapi.Transform(p=gymapi.Vec3(0.0,1.0,1.5))
        self.wall3_pose = gymapi.Transform(p=gymapi.Vec3(4.0,1.0,1.5))
        self.wall_asset_2 = self.gym.create_box(self.sim,4.0,0.1,3.0,simple_options)
        self.wall2_pose = gymapi.Transform(p=gymapi.Vec3(2.0,3.0,1.5))

        fridge_file = "urdf/kitchen/Refrigerator/model_albedo.urdf"
        self.fridge_asset = self.gym.load_asset(self.sim, self.asset_root, fridge_file, simple_options)
        self.fridge_pose = gymapi.Transform(p=gymapi.Vec3(3.1,2.50,0.0), r=gymapi.Quat(0.5, -0.5, -0.5, 0.5))

        kitchen_cabinet_file = "urdf/kitchen/KitchenCabinet/model_dark.urdf"
        kitchen_cabinet_options = gymapi.AssetOptions()
        kitchen_cabinet_options.fix_base_link = True
        kitchen_cabinet_options.vhacd_enabled = True
        kitchen_cabinet_options.vhacd_params = gymapi.VhacdParams()
        kitchen_cabinet_options.vhacd_params.resolution = 300000
        kitchen_cabinet_options.vhacd_params.max_convex_hulls = 512
        kitchen_cabinet_options.vhacd_params.convex_hull_approximation = False
        self.kitchen_cabinet_asset = self.gym.load_asset(self.sim, self.asset_root, kitchen_cabinet_file, kitchen_cabinet_options)
        self.kitchen_cabinet_pose = gymapi.Transform(p=gymapi.Vec3(1.37,1.35,0.0), r=gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), np.pi/2))

        utensil_file = "urdf/kitchen/KitchenUtensils/model.urdf"
        self.utensil_asset = self.gym.load_asset(self.sim, self.asset_root, utensil_file, simple_options)
        self.utensil_pose = gymapi.Transform(p=gymapi.Vec3(2.1, 2.91, 1.1),r=gymapi.Quat(0.5, -0.5, -0.5, 0.5))

        rangehood_file = "urdf/kitchen/rangehood/model.urdf"
        self.rangehood_asset = self.gym.load_asset(self.sim, self.asset_root, rangehood_file, simple_options)
        self.rangehood_pose = gymapi.Transform(p=gymapi.Vec3(0.2,-0.3+0.37+0.37,1.8),r=gymapi.Quat(0, 0.7071068, 0.7071068, 0))

        cooking_bench_file = "urdf/kitchen/cooking_bench/model.urdf"
        self.cooking_bench_asset = self.gym.load_asset(self.sim, self.asset_root, cooking_bench_file, simple_options)
        self.cooking_bench_pose = gymapi.Transform(p=gymapi.Vec3(0.4,-0.3+0.37+0.37,0.86),r=gymapi.Quat(0.7071068, 0, 0, 0.7071068))

        desk_file = "urdf/kitchen/kitchen_table/model_black_marble.urdf"
        self.desk_asset = self.gym.load_asset(self.sim, self.asset_root, desk_file, simple_options)
        self.desk_pose = gymapi.Transform(p=gymapi.Vec3(2.3,0.9,0.0), r=gymapi.Quat(0.5, 0.5, 0.5, 0.5))

        cabinet_file = "urdf/kitchen/WhiteCabinet/model_black_marble.urdf"
        self.cabinet_asset = self.gym.load_asset(self.sim, self.asset_root, cabinet_file, simple_options)
        self.cabinet1_pose = gymapi.Transform(r=gymapi.Quat(0, 0, 0, 1),p=gymapi.Vec3(2.1, 2.91,1.5))
        self.cabinet2_pose = gymapi.Transform(r=gymapi.Quat(0, 0, 0, 1),p=gymapi.Vec3(2.1-0.37, 2.91,1.5))
        self.cabinet3_pose = gymapi.Transform(r=gymapi.Quat(0, 0, 0, 1),p=gymapi.Vec3(2.1-0.37-0.37, 2.91,1.5))
        self.cabinet4_pose = gymapi.Transform(r=gymapi.Quat(0, 0, 0.7071068, 0.7071068),p=gymapi.Vec3(0.17,1.5,1.5))
        self.cabinet5_pose = gymapi.Transform(r=gymapi.Quat(0, 0, 0.7071068, 0.7071068),p=gymapi.Vec3(0.17,1.5+0.37,1.5))
        self.cabinet6_pose = gymapi.Transform(r=gymapi.Quat(0, 0, 0.7071068, 0.7071068),p=gymapi.Vec3(0.17,1.5+0.37+0.37,1.5))

        # set random position for other objects
        y_lim = [0.65,1.2]
        x_lim = [1.45,3.0]
        num_objects = 20
        random_x, random_y = get_object_positions(num_objects, x_lim, y_lim)

        tomato_soup_can_file = "urdf/kitchen/005_tomato_soup_can/model.urdf"
        self.tomato_soup_can_asset = self.gym.load_asset(self.sim, self.asset_root, tomato_soup_can_file, gravity_options)
        self.tomato_soup_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[0],random_y[0], 0.81),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        potted_meat_can_file = "urdf/kitchen/010_potted_meat_can/model.urdf"
        self.potted_meat_can_asset = self.gym.load_asset(self.sim, self.asset_root, potted_meat_can_file, gravity_options)
        self.potted_meat_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[1],random_y[1], 0.81),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        chef_can_file = "urdf/ycb/002_master_chef_can/model.urdf"
        self.chef_can_asset = self.gym.load_asset(self.sim, self.asset_root, chef_can_file, gravity_options)
        self.chef_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[2],random_y[2], 0.81),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        tuna_can_file = "urdf/ycb/007_tuna_fish_can/model.urdf"
        self.tuna_can_asset = self.gym.load_asset(self.sim, self.asset_root, tuna_can_file, gravity_options)
        self.tuna_can_pose = gymapi.Transform(p=gymapi.Vec3(random_x[3],random_y[3], 0.81),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        plum_file = "urdf/ycb/018_plum/model.urdf"
        self.plum_asset = self.gym.load_asset(self.sim, self.asset_root, plum_file, gravity_options)
        self.plum_pose = gymapi.Transform(p=gymapi.Vec3(random_x[4],random_y[4], 0.81),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        strawberry_file = "urdf/ycb/012_strawberry/model.urdf"
        self.strawberry_asset = self.gym.load_asset(self.sim, self.asset_root, strawberry_file, gravity_options)
        self.strawberry_pose = gymapi.Transform(p=gymapi.Vec3(random_x[5],random_y[5], 0.81),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        pear_file = "urdf/ycb/016_pear/model.urdf"
        self.pear_asset = self.gym.load_asset(self.sim, self.asset_root, pear_file, gravity_options)
        self.pear_pose = gymapi.Transform(p=gymapi.Vec3(random_x[6],random_y[6], 0.81),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        mustard_file = "urdf/ycb/006_mustard_bottle/model.urdf"
        self.mustard_asset = self.gym.load_asset(self.sim, self.asset_root, mustard_file, gravity_options)
        self.mustard_pose = gymapi.Transform(p=gymapi.Vec3(random_x[7],random_y[7], 0.81),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        self.tomato_soup_can2_asset = self.gym.load_asset(self.sim, self.asset_root, tomato_soup_can_file, gravity_options)
        self.tomato_soup_can2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[8],random_y[8], 0.81),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        self.tuna_can2_asset = self.gym.load_asset(self.sim, self.asset_root, tuna_can_file, gravity_options)
        self.tuna_can2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[9],random_y[9], 0.81),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        self.potted_meat_can2_asset = self.gym.load_asset(self.sim, self.asset_root, potted_meat_can_file, gravity_options)
        self.potted_meat_can2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[10],random_y[10], 0.81),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        gelatin_box_file = "urdf/ycb/009_gelatin_box/model.urdf"
        self.gelatin_box_asset = self.gym.load_asset(self.sim, self.asset_root, gelatin_box_file, gravity_options)
        self.gelatin_box_pose = gymapi.Transform(p=gymapi.Vec3(random_x[11],random_y[11], 0.81),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        cracker_box_file = "urdf/ycb/003_cracker_box/model.urdf"
        self.cracker_box_asset = self.gym.load_asset(self.sim, self.asset_root, cracker_box_file, gravity_options)
        self.cracker_pose = gymapi.Transform(p=gymapi.Vec3(random_x[12],random_y[12], 0.81),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        sugar_box_file = "urdf/ycb/004_sugar_box/model.urdf"
        self.sugar_box_asset = self.gym.load_asset(self.sim, self.asset_root, sugar_box_file, gravity_options)
        self.sugar_box_pose = gymapi.Transform(p=gymapi.Vec3(random_x[12],random_y[12], 0.81),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        pudding_box_file = "urdf/ycb/008_pudding_box/model.urdf"
        self.pudding_box_asset = self.gym.load_asset(self.sim, self.asset_root, pudding_box_file, gravity_options)
        self.pudding_box_pose = gymapi.Transform(p=gymapi.Vec3(random_x[13],random_y[13], 0.81),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2))

        yellow_bowl_file = "urdf/kitchen/yellow_bowl/model.urdf"
        self.yellow_bowl_asset = self.gym.load_asset(self.sim, self.asset_root, yellow_bowl_file, gravity_options)
        self.yellow_bowl_pose = gymapi.Transform(p=gymapi.Vec3(random_x[14],random_y[14],0.81))

        bowl_designed_file = "urdf/kitchen/bowl_designed/model.urdf"
        self.bowl_designed_asset = self.gym.load_asset(self.sim, self.asset_root, bowl_designed_file, gravity_options)
        self.bowl_designed_pose = gymapi.Transform(p=gymapi.Vec3(random_x[15],random_y[15],0.81))

        sauce1_file = "urdf/kitchen/sauce1/model.urdf"
        self.sauce1_asset = self.gym.load_asset(self.sim, self.asset_root, sauce1_file, gravity_options)
        self.sauce1_pose = gymapi.Transform(p=gymapi.Vec3(random_x[16],random_y[16],0.81))

        sauce2_file = "urdf/kitchen/sauce2/model.urdf"
        self.sauce2_asset = self.gym.load_asset(self.sim, self.asset_root, sauce2_file, gravity_options)
        self.sauce2_pose = gymapi.Transform(p=gymapi.Vec3(random_x[17],random_y[17],0.81))

        cereal_box_file = "urdf/kitchen/cereal_box/model.urdf"
        self.cereal_box_asset = self.gym.load_asset(self.sim, self.asset_root, cereal_box_file, simple_options)
        self.cereal_box_pose = gymapi.Transform(p=gymapi.Vec3(random_x[18],random_y[18],0.81),r=gymapi.Quat(0,0,0,1))

        nesquik_box_file = "urdf/kitchen/nesquik_box/model.urdf"
        self.nesquik_box_asset = self.gym.load_asset(self.sim, self.asset_root, nesquik_box_file, simple_options)
        self.nesquik_box_pose = gymapi.Transform(p=gymapi.Vec3(random_x[19],random_y[19],0.81),r=gymapi.Quat(0, 0, 0.7071068, 0.7071068))

    def set_object_asset(self):
        print("object asset is set")

    def create_robot(self, env, num_env):
        print("robot is created")

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
        fridge_handle = self.gym.create_actor(env, self.fridge_asset, self.fridge_pose, "fridge", num_env, 0)
        kitchen_cabinet_handle = self.gym.create_actor(env, self.kitchen_cabinet_asset, self.kitchen_cabinet_pose, "kitchen_cabinet", num_env, 0)
        utensil_handle = self.gym.create_actor(env, self.utensil_asset, self.utensil_pose, "utensil", num_env, 0)
        rangehood_handle = self.gym.create_actor(env, self.rangehood_asset, self.rangehood_pose, "rangehood", num_env, 0)
        cooking_bench_handle = self.gym.create_actor(env, self.cooking_bench_asset, self.cooking_bench_pose, "cooking_bench", num_env, 0)
        desk_handle = self.gym.create_actor(env, self.desk_asset, self.desk_pose, "desk", num_env, 0)

        cabinet1_handle = self.gym.create_actor(env, self.cabinet_asset, self.cabinet1_pose, "cabinet", num_env, 0)
        cabinet2_handle = self.gym.create_actor(env, self.cabinet_asset, self.cabinet2_pose, "cabinet", num_env, 0)
        cabinet3_handle = self.gym.create_actor(env, self.cabinet_asset, self.cabinet3_pose, "cabinet", num_env, 0)
        cabinet4_handle = self.gym.create_actor(env, self.cabinet_asset, self.cabinet4_pose, "cabinet", num_env, 0)
        cabinet5_handle = self.gym.create_actor(env, self.cabinet_asset, self.cabinet5_pose, "cabinet", num_env, 0)
        cabinet6_handle = self.gym.create_actor(env, self.cabinet_asset, self.cabinet6_pose, "cabinet", num_env, 0)
        self.gym.set_actor_scale(env, cabinet1_handle, 0.4)
        self.gym.set_actor_scale(env, cabinet2_handle, 0.4)
        self.gym.set_actor_scale(env, cabinet3_handle, 0.4)
        self.gym.set_actor_scale(env, cabinet4_handle, 0.4)
        self.gym.set_actor_scale(env, cabinet5_handle, 0.4)
        self.gym.set_actor_scale(env, cabinet6_handle, 0.4)

        tomato_soup_can_handle = self.gym.create_actor(env, self.tomato_soup_can_asset, self.tomato_soup_can_pose, "tomato_soup_can", num_env, 0)
        self.gym.set_actor_scale(env, tomato_soup_can_handle, 1.5)
        potted_meat_can_handle = self.gym.create_actor(env, self.potted_meat_can_asset, self.potted_meat_can_pose, "potted_meat_can", num_env, 0)
        chef_can_handle = self.gym.create_actor(env, self.chef_can_asset, self.chef_can_pose, "chef_can", num_env, 0)
        tuna_can_handle = self.gym.create_actor(env, self.tuna_can_asset, self.tuna_can_pose, "tuna_can", num_env, 0)
        plum_handle = self.gym.create_actor(env, self.plum_asset, self.plum_pose, "plum", num_env, 0)
        strawberry_handle = self.gym.create_actor(env, self.strawberry_asset, self.strawberry_pose, "strawberry", num_env, 0)
        pear_handle = self.gym.create_actor(env, self.pear_asset, self.pear_pose, "pear", num_env, 0)
        mustard_handle = self.gym.create_actor(env, self.mustard_asset, self.mustard_pose, "mustard", num_env, 0)
        tomato_soup_can2_handle = self.gym.create_actor(env, self.tomato_soup_can2_asset, self.tomato_soup_can2_pose, "tomato_soup_can2", num_env, 0)
        self.gym.set_actor_scale(env, tomato_soup_can2_handle, 1.5)
        tuna_can2_handle = self.gym.create_actor(env, self.tuna_can2_asset, self.tuna_can2_pose, "tuna_can2", num_env, 0)
        potted_meat_can2_handle = self.gym.create_actor(env, self.potted_meat_can2_asset, self.potted_meat_can2_pose, "potted_meat_can2", num_env, 0)
        gelatin_box_handle = self.gym.create_actor(env, self.gelatin_box_asset, self.gelatin_box_pose, "gelatin_box", num_env, 0)
        cracker_handle = self.gym.create_actor(env, self.cracker_box_asset, self.cracker_pose, "cracker_box", num_env, 0)
        sugar_box_handle = self.gym.create_actor(env, self.sugar_box_asset, self.sugar_box_pose, "sugar_box", num_env, 0)
        pudding_box_handle = self.gym.create_actor(env, self.pudding_box_asset, self.pudding_box_pose, "pudding_box", num_env, 0)
        yellow_bowl_handle = self.gym.create_actor(env, self.yellow_bowl_asset, self.yellow_bowl_pose, "yellow_bowl", num_env, 0)
        bowl_designed_handle = self.gym.create_actor(env, self.bowl_designed_asset, self.bowl_designed_pose, "bowl_designed", num_env, 0)
        sauce1_handle = self.gym.create_actor(env, self.sauce1_asset, self.sauce1_pose, "sauce1", num_env, 0)
        sauce2_handle = self.gym.create_actor(env, self.sauce2_asset, self.sauce2_pose, "sauce2", num_env, 0)

class KitchenEnvManager:
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