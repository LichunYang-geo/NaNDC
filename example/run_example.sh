#!/bin/bash

event_list="./events.list"
data_dir="./data/"
save_dir="./invs/"

weight="1/1/1"
vmodel="./velocity_example.nd"

python ../nandc/Inversion.py --event_list="$event_list" --data_dir="$data_dir" --vmodel="$vmodel" --save_dir="$save_dir" --weight="$weight"
