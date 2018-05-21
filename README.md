# Behavior Clone Fetch

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
