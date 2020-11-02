

To see demo for goal_to_goal.py, just run:
python goal_to_goal.py

There are two types of agents state update methods:
1. using "vanilla" (simple)
2. using Human Social Force model (almost guarantee non-collision bwtween agents but a little bit slow)
3. training using rl

to use "vanilla":
python goal_to_goal.py --step-name vanilla

to use human social force model:
python goal_to_goal.py --step-name hsf

to use rl training:
python goal_to_goal.py --step-name rl


packages needed: pygame
