import os

event_list = "../examples/msh/shots1.list"
data_dir = "../examples/msh/input/ratio_snr_5f1-20/"
vmodel = "../examples/msh/MSHvel.nd"
save_dir = "../results/msh"

inv_method = "FMT"
weight_pol, weight_amp, weight_spr = 1, 1, 1
nandc_path = "../src/nandc/runNaNDC.py"
os.system(f"python {nandc_path} --event_list={event_list} --data_dir={data_dir} --save_dir={save_dir} \
                --weight_pol={weight_pol} --weight_amp={weight_amp} --weight_spr={weight_spr} \
                --vmodel={vmodel} --inv_method={inv_method} --normalization --skipzero")
