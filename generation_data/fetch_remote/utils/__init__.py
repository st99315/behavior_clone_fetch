import os
import glob
from argparse import ArgumentParser


def set_env_variable(display):
    if display:
        os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-390/libGL.so'
    else:
        os.environ['LD_PRELOAD'] = ''
    print('LD_PRELOAD:', os.environ['LD_PRELOAD'])


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-d",   dest='display', help="True is Using Default GUI",       type=bool, default=False)
    parser.add_argument("-s",   dest='save',    help="True is Saving data",             type=bool, default=False)
    parser.add_argument("-dir", dest='dir',     help="Directory for Saving Data",       type=str,  default='./train_data')
    parser.add_argument("-si",  dest='start',   help="Start of Number",                 type=int,  default=0)
    parser.add_argument("-ei",  dest='end',     help="End of Number",                   type=int,  default=10)
    parser.add_argument("-r",   dest='random',  help="Random light and object texture", type=bool, default=False)
    args = parser.parse_args()
    print('All argument:', args)
    return args


def get_all_xml(xml_dir):
    all_xmls = glob.glob(os.path.join(xml_dir, '*.xml'))
    # print('envs', len(all_xmls))
    return all_xmls

