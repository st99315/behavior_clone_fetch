# Generation Training Data

* Run the Script
```
cd gen_data_scripts
chmod +x gen_data_diff_color.sh
./gen_data_diff_color.sh
```

## Maybe Need
* Install fetch_remote module
```
pip install -e .
```

### This appoarch request install fetch_remote to pip
```
python -m fetch_remote.fetch_pick_object_data -si 0 -ei 100 -r True -s True -dir $SOMEWHERE
```
* -si : type int,  start of index.
* -ei : type int,  end of index.
* -r  : type bool, random texture and light.
* -s  : type bool, saving.
* -dir: type str,  saving directory.
