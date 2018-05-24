# Behavior Clone Fetch

* All scripts are run on Python3
* Python2 not tried

## Request OpenAI GYM

* Recommended use virtual enviroment of python
```bash
git clone https://github.com/st99315/gym
cd gym
git checkout imitation_fetch
pip install -e '.[all]'
```

### Generation data
* First need to [generate training data](./generation_data/README.md)

### Training NN
```bash
python train_im_move_behavior_clone.py
```

### Testing NN
```bash
python test_im_move_behavior_clone.py
```

## Using Virtualenv
1. Install virtualenv to pip3 lib
```bash
pip3 install -U virtualenv
```

2. Create new virtual enviroment using Python3
```bash
virtualenv -p python3 envname
```

3. Activate/Deactivate virtual enviroment
* you can create many enviroments
```bash
source $SOMEWHERE/envname/bin/activate
# exit virtual enviroment
deactivate
```

4. Install Python module to virtual enviroment
```bash
pip install $SOMEMODLUE
```
