# Behavior Clone Fetch :metal:
### Using imitation learning let Fetch robot to learn a policy of grasping in Gym

- [x] All scripts are run on Python3
- [ ] Python2 not tried

## Request OpenAI GYM
* Recommended use virtual enviroment of python

1. Install gym
```bash
git clone https://github.com/st99315/gym
cd gym
git checkout imitation_fetch
pip install -e '.[all]'
```

2. Install mujoco-py
```bash
git clone https://github.com/st99315/mujoco-py
cd mujoco-py
git checkout imitation_fetch
pip install -e .
```

## How to Use
### Generation data
* First need to [generate training data](./generation_data/README.md)

### Training Network
```bash
python train_im_behavior_clone.py
```

### Testing Network
```bash
python test_im_behavior_clone.py
```

### Data Aggregation
```bash
python dagger.py
```

## Using Virtualenv
1. Install virtualenv to pip3 lib
```bash
pip3 install -U virtualenv
```

2. Create new virtual enviroment with python3
* You can create many enviroments, and using different interpreter
```bash
virtualenv -p python3 envname
```

3. Activate/Deactivate virtual enviroment
```bash
source $SOMEWHERE/envname/bin/activate

# exit virtual enviroment
deactivate
```

4. Install python module to virtual enviroment
```bash
pip install $SOMEMODLUE
```
