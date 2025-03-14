DIR_PATH=`pwd`
# UnrealEnv need absolute path
UnrealEnv=${DIR_PATH}/dataset PYTHONPATH=./ python gym_rescue/example/rescue_demo.py --render --keyboard
