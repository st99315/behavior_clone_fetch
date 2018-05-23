#!/bin/bash

# dataset directory
DIRECTORY_SET="train_data_same_color_0522"
# log directory
DIRECTORY="train_data_same_color"

if [  -d "$DIRECTORY" ]; then
    rm -rf "$DIRECTORY"
fi
if [ ! -d "$DIRECTORY" ]; then
    mkdir "$DIRECTORY"
fi

# activate python virtualenv
source $HOME/youjun/remote/test_env/bin/activate

let "batch = 1"
for i in $(seq 10)
do
    let "s = (i-1) * batch"
    let "e = i * batch"
    
    # this appoarch request install fetch_remote to pip
    # -si: start of index, -ei: end of index, -r: random texture and light, -dir: saving directory
    python -m fetch_remote.fetch_pick_object_data -si $s -ei $e -r True -s True -dir ../$DIRECTORY_SET/train_data > $DIRECTORY/gen_$s_$e.log 2>&1 &

    # try following in ssh or docker
    #xvfb-run -a -s "-screen 0 1400x900x24"  python3 gen_move_train_data.py -s $s -e $e > $DIRECTORY/g$s_$e.log 2>&1 &
done
