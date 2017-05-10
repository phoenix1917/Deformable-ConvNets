# -*- coding: UTF-8 -*-

import os.path as osp
import sys

# 将一个路径加入系统PATH
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# 将lib目录加入系统PATH
this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)
