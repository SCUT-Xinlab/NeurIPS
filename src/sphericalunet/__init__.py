import os

ABSPATH = os.path.dirname(os.path.abspath(__file__))

SURF_PATH = os.path.join(ABSPATH, "surf")

NEIGH_INDICES_PATH = os.path.join(ABSPATH, "neigh_indices")

TEMPLATE_PATH = os.path.join(ABSPATH, "template")

NUM_VOXELS = {
    "1": 12,
    "2": 42,
    "3": 162,
    "4": 642,
    "5": 2562,
    "6": 10242,
    "7": 40962,
    "8": 163842
}