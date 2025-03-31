#!/bin/sh
python3.10 generate_data.py
#python3.10 train.py -na medium_BOSZ --autocast False --compile True
python3.10 train_NDE.py -nl medium_BOSZ -na medium_BOSZ_Z_mix --autocast False --compile True
