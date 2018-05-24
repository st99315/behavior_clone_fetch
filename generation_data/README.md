# Generation Training Data

## Run the Script
* Generation training Data
```bash
cd gen_data_scripts
chmod +x gen_data_diff_color.sh
./gen_data_diff_color.sh
```

* Generation validation Data
```bash
./gen_data_diff_color_valid.sh
```

## Maybe Need
* Install fetch_remote module
```bash
# in generation_data directory
pip install -e .
```

### This appoarch request install fetch_remote
```bash
python -m fetch_remote.fetch_pick_object_data -si 0 -ei 100 -r True -s True -dir $SOMEWHERE
```
* -si : type int,  start of index.
* -ei : type int,  end of index.
* -r  : type bool, random texture and light.
* -s  : type bool, saving data.
* -dir: type str,  directory for saving.
