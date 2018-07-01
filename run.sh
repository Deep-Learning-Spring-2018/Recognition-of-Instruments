#!/bin/bash
#########################################################################
# File Name: run.sh
# Author:  Tiangang Zhou
# mail: tg_zhou@pku.edu.cn
#########################################################################

# Prepared data
python3 ./aiff_process.py

# Run the net
python3 ./cnn_network.py
