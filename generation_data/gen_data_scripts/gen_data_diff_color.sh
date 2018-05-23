#!/bin/bash

# dataset directory
DIRECTORY_SET="train_data_diff_color_0523"
# log directory
DIRECTORY="train_data"

if [  -d "$DIRECTORY" ]; then
    rm -rf "$DIRECTORY"
fi
if [ ! -d "$DIRECTORY" ]; then
    mkdir "$DIRECTORY"
fi

# activate python virtualenv
source $HOME/youjun/remote/test_env/bin/activate

let "batch = 2000"
for i in $(seq 10)
do
    let "s = (i-1) * batch"
    let "e = i * batch"
    
    python $HOME/youjun/remote/fetch_remote/fetch_pick_object_data.py -si $s -ei $e -r True -s True -dir ../$DIRECTORY_SET/train_data > $DIRECTORY/gen_$s_$e.log 2>&1 &
    
    # try following in ssh or docker
    #xvfb-run -a -s "-screen 0 1400x900x24"  python3 gen_move_train_data.py -s $s -e $e > $DIRECTORY/g$s_$e.log 2>&1 &
done
