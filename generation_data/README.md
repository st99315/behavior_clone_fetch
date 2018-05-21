# Generation Training Data

* Run the Script
```
./gen_data_scripts/gen_data_diff_color.sh
```

## Maybe Need
* Install fetch_remote module
```
python setup.py -e .
```

* This appoarch request install fetch_remote to pip
* -si: start of index, -ei: end of index, -r: random texture and light, -dir: saving directory
```
python -m fetch_remote.fetch_pick_object_data -si 0 -ei 100 -r True -s True -dir $SOMEWHERE
```
