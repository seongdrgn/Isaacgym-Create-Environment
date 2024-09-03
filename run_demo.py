"""
date: 2024.08.23
author: Seongyong Kim
description: test file for sim_env
"""

import os
os.system('clear')

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import pandas as pd
import random
from sim_env.kitchen.env_manager import KitchenEnvManager

def tensor_to_rotation_matrix(rot):
    # print(rot.shape)
    w, x, y, z=rot[0]

    R = torch.tensor([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])

    return R

def local_force_from_global_force(R, global_force):
    R=R.to('cuda:0')
    local_force=torch.matmul(R.T, global_force)

    return local_force

class SimEnv:
    def __init__(self):
        # set random seed
        np.random.seed(0)
        torch.set_printoptions(precision=4, sci_mode=False)

        # acquire gym interface
        self.gym = gymapi.acquire_gym()

        # config gym argument
        custom_parameters = [
            {"name": "--controller", "type": str, "default": "ik"},
            {"name": "--num_envs", "type": int, "default": 1},
        ]
        self.args = gymutil.parse_arguments(
            description="sim_env_test",
            custom_parameters=custom_parameters,)

        # set torch device
        self.device = self.args.sim_device if self.args.use_gpu_pipeline else 'cpu'

        # modify asset root path & data save path
        self.asset_root = "/home/kimsy/isaacgym/IsaacGymEnvs/NIA_for_sample_dataset/code"
        self.save_dir = "/home/kimsy/isaacgym/IsaacGymEnvs/NIA_for_sample_dataset/code/sensor_data/test/0822/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print("Directory is created")

        # set sensor link
        self.net_force_sensor_names=["ur10_leftfinger_ft_sensor", "ur10_rightfinger_ft_sensor"]
    
        # set interact object pose
        self.interact_object_pose = gymapi.Transform(p=gymapi.Vec3(2.4, 1.9, 0.825),r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi))

        # num_envs
        self.num_envs = self.args.num_envs

    def create_sim(self):
        # config sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 120.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        if self.args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = self.args.num_threads
            sim_params.physx.use_gpu = self.args.use_gpu
        else:
            raise Exception("This example can only be used with PhysX")

        # create sim
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.viewer_pos = gymapi.Vec3(1.0, 1.0, 1.5)
        self.viewer_target = gymapi.Vec3(3.0, 2.8, 0.8)
        if self.viewer is None:
            raise Exception("Failed to create viewer")

    def set_camera(self):
        self.cam_positions = []
        self.cam_targets = []

        self.cam_props = gymapi.CameraProperties()
        self.cam_props.horizontal_fov = 70.0
        self.cam_props.width = 1280
        self.cam_props.height = 720

        self.cam_positions.append(gymapi.Vec3(1.0, 1.0, 1.5))
        self.cam_targets.append(gymapi.Vec3(3.0, 2.8, 0.8))

        self.cam_positions.append(gymapi.Vec3(2.95, 1.7, 1.3))
        self.cam_targets.append(gymapi.Vec3(1.5, 1.7, 0.4))

    def create_env(self):
        # set design asset
        self.kitchen_env_manager.set_current_robot_asset()
        self.ur_dof_props = self.kitchen_env_manager.current_env.ur_dof_props
        self.num_ur_dofs = self.kitchen_env_manager.current_env.num_ur_dofs
        self.default_dof_pos = self.kitchen_env_manager.current_env.default_dof_pos
        self.default_dof_state = self.kitchen_env_manager.current_env.default_dof_state
        self.ur_gripper_index = self.kitchen_env_manager.current_env.ur_gripper_index

        # self.ur_dof_props = ur_dof_props
        # self.num_ur_dofs = num_ur_dofs
        # self.default_dof_pos = default_dof_pos
        # self.default_dof_state = default_dof_state
        # self.ur_gripper_index = ur_gripper_index

        self.kitchen_env_manager.set_current_interior_asset()
        self.kitchen_env_manager.set_current_object_asset()

        # config env grid
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print("Creating %d environments" % self.num_envs)

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # set camera
        self.set_camera()

        # create buffers
        self.envs = []
        box_idxs = []
        self.gripper_idxs = []
        self.ee_idxs=[]
        self.gripper_left_idxs=[]
        self.gripper_right_idxs=[]

        object_idxs = []
        self.init_pos_list = []
        self.init_rot_list = []

        self.cam_handles = []

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            # add ur5e
            # self.kitchen_env_manager.create_current_robot(env, i)
            # ur_handle = self.kitchen_env_manager.current_env.ur_handle

            # self.gym.set_actor_dof_properties(env, ur_handle, self.ur_dof_props)
            # self.gym.set_actor_dof_states(env, ur_handle, self.default_dof_state, gymapi.STATE_ALL)
            # self.gym.set_actor_dof_position_targets(env, ur_handle, self.default_dof_pos)
            # self.gym.enable_actor_dof_force_sensors(env, ur_handle)

            # self.sensor_handles = [self.gym.find_actor_rigid_body_handle(env, ur_handle, sensor_name) for sensor_name in self.net_force_sensor_names]

            # # get initial gipper pose
            # gipper_handle = self.gym.find_actor_rigid_body_handle(env, ur_handle, "gripper_center")
            # gripper_pose = self.gym.get_rigid_transform(env, gipper_handle)
            # self.init_pos_list.append([gripper_pose.p.x, gripper_pose.p.y, gripper_pose.p.z])
            # self.init_rot_list.append([gripper_pose.r.x, gripper_pose.r.y, gripper_pose.r.z, gripper_pose.r.w])

            # # get global index of gripper in rigid body state tensor
            # gripper_idx = self.gym.find_actor_rigid_body_index(env, ur_handle, "gripper_center", gymapi.DOMAIN_SIM)
            # self.gripper_idxs.append(gripper_idx)

            # ee_idx = self.gym.find_actor_rigid_body_index(env, ur_handle, "ee_link", gymapi.DOMAIN_SIM)
            # self.ee_idxs.append(ee_idx)

            # gripper_left_idx = self.gym.find_actor_rigid_body_index(env, ur_handle, "ur10_leftfinger_ft_sensor", gymapi.DOMAIN_SIM)
            # self.gripper_left_idxs.append(gripper_left_idx)

            # gripper_right_idx = self.gym.find_actor_rigid_body_index(env, ur_handle, "ur10_rightfinger_ft_sensor", gymapi.DOMAIN_SIM)
            # self.gripper_right_idxs.append(gripper_right_idx)

            self.kitchen_env_manager.create_current_assets(env, i)

            # set camera
            for c in range(len(self.cam_positions)):
                self.cam_handles.append(self.gym.create_camera_sensor(env, self.cam_props))
                self.gym.set_camera_location(self.cam_handles[c], env, self.cam_positions[c], self.cam_targets[c])

            # set light
            l_color = gymapi.Vec3(0.8, 0.8, 0.8)
            l_ambient = gymapi.Vec3(0.8,0.8,0.8)
            l_direction = gymapi.Vec3(1.5,-1,10)
            self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)
            self.gym.set_light_parameters(self.sim, 1, gymapi.Vec3(), gymapi.Vec3(), gymapi.Vec3())
            self.gym.set_light_parameters(self.sim, 2, gymapi.Vec3(), gymapi.Vec3(), gymapi.Vec3())

        # self.init_data()

    def init_data(self):
        # get jacobian tensor
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries correspoinding to franka hand
        self.j_eef = jacobian[:, self.ur_gripper_index-1, :, :6]
        self.j_link = jacobian[:, :6, :, :6]

        # get mass matrix tensor
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "ur")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :6, :6]

        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # get actor state tensor
        _act_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.act_states = gymtorch.wrap_tensor(_act_states)

        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        dof_pos = dof_states[:, 0].view(self.num_envs, self.num_ur_dofs, 1)
        dof_vel = dof_states[:, 1].view(self.num_envs, self.num_ur_dofs, 1)
        self.dof_vel_robot_link = dof_vel[:,:6,:]

        # set initial pos action
        self.pos_action = torch.zeros_like(dof_pos).squeeze(-1)
        # print(self.init_pos_list)
        self.init_pos = torch.Tensor(self.init_pos_list).view(self.num_envs, 3).to(self.device)
        self.init_rot = torch.Tensor(self.init_rot_list).view(self.num_envs, 4).to(self.device)

        # get contact force tensor
        net_contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        rb_force_tensor = gymtorch.wrap_tensor(net_contact_force_tensor).view(-1,3)

        gripper_left_rot = self.rb_states[self.gripper_left_idxs, 3:7]
        gripper_left_matrix = tensor_to_rotation_matrix(gripper_left_rot)
        gripper_left_force = local_force_from_global_force(gripper_left_matrix, rb_force_tensor[self.sensor_handles[0], :])

        gripper_right_rot = self.rb_states[self.gripper_right_idxs, 3:7]
        gripper_right_matrix = tensor_to_rotation_matrix(gripper_right_rot)
        gripper_right_force = local_force_from_global_force(gripper_right_matrix, rb_force_tensor[self.sensor_handles[1], :])

        # get dof force tensor
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        dof_force = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_ur_dofs)

        # tensor for the FT sensor
        _fsdata = self.gym.acquire_force_sensor_tensor(self.sim)
        fsdata = gymtorch.wrap_tensor(_fsdata)

        ee_rot_matrix=tensor_to_rotation_matrix(self.rb_states[self.ee_idxs, 3:7])
        ee_local_force=local_force_from_global_force(ee_rot_matrix, fsdata[0,:3])
        ee_local_torque=local_force_from_global_force(ee_rot_matrix, fsdata[0,3:])

        # set control parameters
        self.kp = 5000
        self.kp_orn = 29010
        self.kv = 2*math.sqrt(4950)

        self.u_compensation = torch.zeros((1,self.num_ur_dofs), device=self.device)

        self.err_pos = torch.zeros(1,6, device=self.device)
        self.err_orn = torch.zeros(1,6, device=self.device)

        self.gripper_close = torch.Tensor([[0.012,0.012]]*self.num_envs).to(self.device)
        self.gripper_open = torch.Tensor([[0.0, 0.0]]*self.num_envs).to(self.device)

        self.fsdata=fsdata

        # self.gripper_pos=self.rb_states[self.gripper_idxs, :3]
        # self.gripper_rot=self.rb_states[self.gripper_idxs, 3:7]

        self.ee_pos=self.rb_states[self.ee_idxs, :3]
        self.ee_rot=self.rb_states[self.ee_idxs, 3:7]
        self.dof_pos=dof_pos
        self.dof_vel=dof_vel
        self.dof_force=dof_force
        self.ee_local_force=ee_local_force
        self.ee_local_torque=ee_local_torque
        self.gripper_left_local_force = gripper_left_force
        self.gripper_right_local_force = gripper_right_force

        self.is_save=False

        colums = ['step'] \
            + ['ee_pos_x', 'ee_pos_y', 'ee_pos_z', 'ee_rot_x', 'ee_rot_y', 'ee_rot_z', 'ee_rot_w'] \
            + [f'robot_joint_pos_{i}' for i in range(self.num_ur_dofs)] \
            + [f'robot_joint_vel_{i}' for i in range(self.num_ur_dofs)] \
            + [f'robot_joint_tor_{i}' for i in range(self.num_ur_dofs)] \
            + ['ee_force_x', 'ee_force_y', 'ee_force_z'] \
            + ['ee_torque_x', 'ee_torque_y', 'ee_torque_z'] \
            + ['contact_1_force_x', 'contact_1_force_y', 'contact_1_force_z'] \
            + ['contact_2_force_x', 'contact_2_force_y', 'contact_2_force_z']
            # + ['obj_pos_x', 'obj_pos_y', 'obj_pos_z', 'obj_rot_x', 'obj_rot_y', 'obj_rot_z', 'obj_rot_w'] \

        self.dataframe = pd.DataFrame(columns=colums)

        self.skip_frame = 100
        self.frame = 0

        self.total_frame=700

        self.grasping_frame=0
        self.picking_frame=0
        self.place_frame=0

    def run_sim(self):
        self.create_sim()

        # set environment manager
        self.kitchen_env_manager = KitchenEnvManager(asset_root=self.asset_root, gym=self.gym, sim=self.sim, device=self.device)
        self.kitchen_env_manager.set_env("env1")
        """
        wall_type : ivory, white, green
        """
        self.kitchen_env_manager.set_wall_type("white")
        
        self.create_env()

        self.gym.viewer_camera_look_at(self.viewer, self.envs[0], self.viewer_pos, self.viewer_target)
        self.gym.prepare_sim(self.sim)

        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # refresh tensors
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
            self.gym.refresh_force_sensor_tensor(self.sim)

            """
            Control code here.
            """
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            """
            Data saving code here.
            """
        #     self.frame += 1

        #     if self.is_save:
        #         # list_obj_pos = (cube_pos[0,:].detach().cpu().numpy().copy()).tolist()
        #         # list_obj_rot = (cube_rot[0,:].detach().cpu().numpy().copy()).tolist()
        #         list_ee_pos = (self.gripper_pos[0,:].detach().cpu().numpy().copy()).tolist()
        #         list_ee_rot = (self.gripper_rot[0,:].detach().cpu().numpy().copy()).tolist()
        #         list_dof_pos = (self.dof_pos.squeeze(-1)[0,:].detach().cpu().numpy().copy()).tolist()
        #         list_dof_vel = (self.dof_vel.squeeze(-1)[0,:].detach().cpu().numpy().copy()).tolist()
        #         list_dof_tor = (self.dof_force[0,:].detach().cpu().numpy().copy()).tolist()
        #         #list_ee_force = (ee_local_force.detach().cpu().numpy().copy()).tolist()
        #         list_ee_force = (self.fsdata[0,:3].detach().cpu().numpy().copy()).tolist()
        #         list_ee_angvel = (self.ee_angvel[0,:].detach().cpu().numpy().copy()).tolist()
        #         list_contact_1_force = (self.gripper_left_local_force.detach().cpu().numpy().copy()).tolist()
        #         list_contact_2_force = (self.gripper_right_local_force.detach().cpu().numpy().copy()).tolist()

        #         row = [self.frame] \
        #                 + list_ee_pos + list_ee_rot \
        #                 + list_dof_pos \
        #                 + list_dof_vel \
        #                 + list_dof_tor \
        #                 + list_ee_force \
        #                 + list_ee_angvel \
        #                 + list_contact_1_force \
        #                 + list_contact_2_force
        #                 # + list_obj_pos + list_obj_rot \
        #         self.dataframe.loc[self.frame] = row

        # if self.is_save:
        #     _dir = self.save_dir + 'test_data_cube_no_trans.csv'
        #     self.dataframe.to_csv(_dir, index=False)
        #     print("Data saved to joint_data.csv")
        
        # clean up
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

if __name__ == "__main__":
    # torch.cuda.init()
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["TORCH_USE_CUDA_DSA"] = '1'
    sim_env = SimEnv()
    sim_env.run_sim()