#!/bin/bash

event_list="./example/events.list"
data_dir="./example/"
save_dir="./example/invs/"

weight="1/1/1"
vmodel="./example/toc2me.nd"

python ./nandc/Inversion.py --event_list="$event_list" --data_dir="$data_dir" --vmodel="$vmodel" --save_dir="$save_dir" --weight="$weight"
