"""
Given a parent directory, return all the paths to
image class paths
"""

import sys
import glob

if __name__ == "__main__":
    parent_dir = sys.argv[1]
    all_dirs = glob.glob(parent_dir + "/*/*/*")
    for i in all_dirs:
        print(i)
