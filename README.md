# Create Simulation Environments using IsaacGym
## 3 Concepts of simulation environments
Utilize objects in [Google Scanned Objects](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research), [YCB Dataset](https://www.ycbbenchmarks.com/object-set/) and [Objaverse Dataset](https://objaverse.allenai.org/objaverse-1.0).

<details>
<summary>Click to see example environments</summary>
<div markdown="1">

* Kitchen

> <img src="https://github.com/user-attachments/assets/b4c4f9b7-a315-4227-9f17-8b04a4d20730" width="340" height="180">  <img src="https://github.com/user-attachments/assets/3c7b6f94-4487-48a8-8624-b26d6ab891fe" width="340" height="180">  <img src="https://github.com/user-attachments/assets/734c14f3-c60d-47eb-b378-097a7b927763" width="340" height="180">  <img src="https://github.com/user-attachments/assets/964b6f9f-36ae-4fee-b011-837a34a2576f" width="340" height="180">  <img src="https://github.com/user-attachments/assets/0a1f6b25-0c41-4f71-8501-acfee4bc67e8" width="340" height="180"> 
  
* Office

> <img src="https://github.com/user-attachments/assets/e8953210-d2a4-4820-b388-ee715c052830" width="340" height="180">  <img src="https://github.com/user-attachments/assets/13c35292-0c3f-4569-a2f1-219e42620f94" width="340" height="180">  <img src="https://github.com/user-attachments/assets/87bb13b4-6704-42e5-aa9a-da40fd7a4a03" width="340" height="180">  <img src="https://github.com/user-attachments/assets/d6c106c1-b964-45ea-9bb5-d20e9accae97" width="340" height="180">  <img src="https://github.com/user-attachments/assets/3ea84cc3-00ef-4da2-9dc0-dc31457ea5c2" width="340" height="180"> 

* Living Room

> <img src="https://github.com/user-attachments/assets/e3e8083a-0ab8-4dbf-a33b-6b5f05190160" width="340" height="180">  <img src="https://github.com/user-attachments/assets/3d6eb4ae-df33-43c5-82e1-eef2ce269e02" width="340" height="180">  <img src="https://github.com/user-attachments/assets/2d824dce-16bc-4454-9cbd-8853b9cfea61" width="340" height="180">  <img src="https://github.com/user-attachments/assets/7dd8086d-7710-4311-b2fd-fad083965851" width="340" height="180">  <img src="https://github.com/user-attachments/assets/8d32a739-7281-4833-a40e-b3496a886545" width="340" height="180">
</div>
</details>

## Domain Transformation

For changing interior of the environment, edit below code in ```run_demo_kitchen.py``` file.

Edit : ```self.kitchen_env_manager.set_env("env5")```
> If users want to change interior "env5" to "env1", just edit the code.
>
> ```ruby
> self.kitchen_env_manager.set_env("env1")
> ```
>
> This function is same for all concepts of environments.
>
> Possible interior name : "env1", "env2", "env3", "env4", "env5".


Additionally, users can change wall color to edit the code in ```run_demo_kitchen.py``` file.

Edit : ```self.kitchen_env_manager.set_wall_type("white")```
> If users want to change wall color "white" to "gray", just edit the code.
>
> ```ruby
> self.kitchen_env_manager.set_wall_type("gray")
> ```
>
> This function is also same for all concepts of environments.
>
> Possible wall types : "white", "ivory", "gray".

There are 3 types of floor and edit the code in ```env_manager.py``` file to change floor type.

Edit : ```floor_file = "urdf/floor/wood1.urdf"``` in ```def set_interior_asset(self)``` function
> If users want to change floor type "wood1" to "wood2", just edit the code.
>
> ```foor_file = "urdf/floor/wood2.urdf"```
>
> This function is same for all concepts of environments.
>
> Possible floor types : "wood1", "wood2", "wood3".

## Run Simulation

### Kitchen

```ruby
python run_demo_kitchen.py
```

### Office

```ruby
python run_demo_office.py
```

### Living Room

```ruby
python run_demo_livingroom.py
```


## Troubleshooting

For Anaconda Users

```
export LD_LIBRARY_PATH=/home/user_name/anaconda3/envs/your_env/lib
```
