import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/mateus/mateus_alonso/ros2_ws/src/install/ps_localization'
