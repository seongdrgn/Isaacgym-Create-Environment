# Create Simulation Environments using IsaacGym
## 3 Concepts of simulation environments

* Kitchen

> <img src="https://github.com/user-attachments/assets/b4c4f9b7-a315-4227-9f17-8b04a4d20730" width="340" height="180">  <img src="https://github.com/user-attachments/assets/53387e93-541f-458d-986c-8946d1249a1a" width="340" height="180">  <img src="https://github.com/user-attachments/assets/734c14f3-c60d-47eb-b378-097a7b927763" width="340" height="180">  <img src="https://github.com/user-attachments/assets/964b6f9f-36ae-4fee-b011-837a34a2576f" width="340" height="180">  <img src="https://github.com/user-attachments/assets/0a1f6b25-0c41-4f71-8501-acfee4bc67e8" width="340" height="180"> 
  
* Office

* Living Room

## Domain Transformation

For changing interior of the environment, edit below code in ```run_demo_kitchen.py``` file.

Edit : ```self.kitchen_env_manager.set_env("env5")```
> If users want to change interior "env5" to "env1", just edit the code.
>
> ```self.kitchen_env_manager.set_env("env1")```
>
> This function is same for all concepts of environments.
>
> Possible interior name : "env1", "env2", "env3", "env4", "env5".


Additionally, users can change wall color to edit the code in ```run_demo_kitchen.py``` file.

Edit : ```self.kitchen_env_manager.set_wall_type("white")```
> If users want to change wall color "white" to "gray", just edit the code.
>
> ```self.kitchen_env_manager.set_wall_type("gray")```
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

```python run_demo_kitchen.py```

### Office

```python run_demo_office.py```

### Living Room

```python run_demo_livingroom.py```


## Troubleshooting

For Anaconda Users

```
export LD_LIBRARY_PATH=/home/user_name/anaconda3/envs/your_env/lib
```
