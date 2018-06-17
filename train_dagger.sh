#!/bin/bash

# activate python virtualenv
# if you don't use virtualenv, then comment it
source $VIRTUALENV_PATH/bin/activate

# getting interrupt to stop this script
trap - INT

for i in $(seq 10)
do
    echo ---------- dagger iter $i ----------
    python train_im_behavior_clone.py
    python dagger.py
    python dagger_valid.py
done
