"""
    Generation XML file from Template file
        - aim to change object texture
"""

import os
import glob
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import random
from data_save import check_dir

_XML_PATH = '/home/iclab/youjun/gym_env/lib/python3.5/site-packages/gym/envs/robotics/assets/fetch'
_OBJ_PATH = '/home/iclab/youjun/gym_env/lib/python3.5/site-packages/gym/envs/robotics/assets/textures/obj_textures'
_DST_FOLDER = 'myenvs'


def get_all_textures():
    os.chdir(_OBJ_PATH)
    # get all of files
    all_files = glob.glob(os.path.join(_OBJ_PATH, '*.png'))
    # shuffle
    random.shuffle(all_files)
    half_size = int(len(all_files) / 2.)

    os.chdir(_XML_PATH)
    return all_files[:half_size], all_files[half_size:]


def change_texture(root, obj_file, dis_file):
    obj_path = os.path.join('./obj_textures', obj_file)
    dis_path = os.path.join('./obj_textures', dis_file)
    for texture in root.findall('./asset/texture'):
        texture.attrib['file'] = obj_path if texture.attrib['name'] == 'object' else dis_path


# get template
os.chdir(_XML_PATH)
tree = ET.ElementTree(file='pick_and_place_template.xml')
root = tree.getroot()

# get all of textures
obj_files, dis_files = get_all_textures()
check_dir(_DST_FOLDER)

while len(obj_files):
    obj_file = obj_files.pop().split('/')[-1]
    dis_file = dis_files.pop().split('/')[-1]
    change_texture(root, obj_file, dis_file)

    xml_name = '{0}_{1}.xml'.format(obj_file.split('.')[0], dis_file.split('.')[0])
    tree.write(os.path.join(_DST_FOLDER, xml_name))

