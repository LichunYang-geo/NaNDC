import os

event_list="D:/Projects/NaNDC/nandc-v0.0.1/example/events.list"
data_dir="D:/Projects/NaNDC/nandc-v0.0.1/example/data"
save_dir="D:/Projects/NaNDC/nandc-v0.0.1/example/invs"
weight="1/1/1"
vmodel="D:/Projects/NaNDC/nandc-v0.0.1/example/velocity_example.nd"

os.system(f"python D:/Projects/NaNDC/nandc-v0.0.1/nandc/Inversion.py --event_list={event_list} --data_dir={data_dir} --vmodel={vmodel} --save_dir={save_dir} --weight={weight}")