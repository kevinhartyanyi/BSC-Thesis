from natsort import natsorted # Run with python3
import os
import numpy as np
import matplotlib.pyplot as plt

def list_directory(dir_name, extension=None):
    """
    List all files with the given extension in the specified directory.
    :param dir_name: name of the directory
    :param extension: filename suffix
    :return: list of file locations  
    """
    if extension is None:
        is_valid = lambda fn: os.path.isfile(fn)
    else:
        is_valid = lambda fn: fn[-len(extension):] == extension
    
    fns = [os.path.join(dir_name, fn) 
            for fn in os.listdir(dir_name) if is_valid(os.path.join(dir_name, fn))]
    fns = natsorted(fns)
    #fns.sort()
    return fns

speeds_dir = list_directory("./", extension='speed.npy')
speeds_dir = natsorted(speeds_dir)
speeds = []
for s in speeds_dir:
    speeds.append(np.load(s))

print(speeds[0])
plt.plot(speeds, "m")
plt.ylabel("Speed in km/h")
plt.xlabel("Frame number")
#plt.xticks(np.arange(0, len(speeds), 1))
plt.grid(axis="y", linestyle="-")
plt.savefig("aaa.png", bbox_inches='tight', dpi=150)

plt.show()
