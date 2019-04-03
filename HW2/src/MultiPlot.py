# plot.py wrapper and new style for generating graphs
# Author: Cristobal Pais
# Date: September 2018

# Importations
import os
import sys
import glob
import re
import argparse
    
# Parse command line arguments    
parser = argparse.ArgumentParser()
parser.add_argument('-re', type=str)     # Regular expression
parser.add_argument('-o', type=str)     # Output plot name (.pdf)
args = parser.parse_args()

# Current working directory (inside the root of the HW2 folder)
cwd = os.getcwd()
files = os.listdir(cwd + "/data")

# Filter directories based on regular expression
x = args.re
filteredFiles = []
for i in range(len(files)):
    if len(re.findall(string=files[i], pattern=x)) > 0:
        filteredFiles.append(files[i])

# Call plot.py with the filtered arguments
cmdline = "python plot.py"
for i in range(len(filteredFiles)):
    cmdline += " data/" + filteredFiles[i]
cmdline += " --plotName="+args.o
os.system(cmdline)