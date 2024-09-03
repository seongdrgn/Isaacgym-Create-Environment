# Create Simulation Environments using IsaacGym
## 3 Concepts of simulation environments

* Kitchen
* Office
* Living Room

## Domain Transformation

For changing interior of the environment, edit below code in ```run_demo_kitchen.py``` file

Edit : ```self.kitchen_env_manager.set_env("env5")```
> If users want to change interior "env5" to "env1", just edit the code
>
> ```self.kitchen_env_manager.set_env("env1")```
>
> This function is same for all concepts of environments
>
> Possible interior name : "env1", "env2", "env3", "env4", "env5"


Additionally, users can change wall color to edit the code in ```run_demo_kitchen.py``` file

Edit : ```self.kitchen_env_manager.set_wall_type("white")```
> If users want to change wall color "white" to "gray", just edit the code
>
> ```self.kitchen_env_manager.set_wall_type("gray")```
>
> This function is also same for all concepts of environments
>
> Possible wall types : "white", "ivory", "gray"

There are 3 types of floor and edit the code in ```env_manager.py``` file to change floor type

Edit : ```floor_file = "urdf/floor/wood1.urdf"``` in ```def set_interior_asset(self)``` function
> If users want to change floor type "wood1" to "wood2", just edit the code
>
> ```foor_file = "urdf/floor/wood2.urdf"```
>
> This function is same for all concepts of environments
>
> Possible floor types : "wood1", "wood2", "wood3"

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
