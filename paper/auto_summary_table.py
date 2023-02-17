import auto_ap_thrs_eval as ap
import auto_pdq_thrs_eval as pdq
import common_settings as s
import model_utils as utils
import os

path_to_models = "E:/models/FINAL_ours"

outputs = []
for dir in s.list_all_dirs_only(path_to_models):
    arch, channels = utils.parse_config_and_channels_from_checkpoint_path(dir)

    pdq_data = pdq.read_pdq_data(os.path.join(path_to_models, dir, pdq.output_file_name), dir)
    pdq_val_max = max(pdq_data[1], key=lambda x:x[1])

    ap_data = ap.read_ap_data(os.path.join(path_to_models, dir, s.score_thrs_file_name), dir)
    ap_val_max = max(ap_data[1], key=lambda x:x[1])
    ap_at_pdq_max = next((ap_val for ap_val in ap_data[1] if ap_val[0] == pdq_val_max[0]), None) 

    # {pdq_data[0]} \t 
    outputs.append([arch, channels, f"{pdq_val_max[1]:.4f} \t {pdq_val_max[0]:.3f} \t {ap_at_pdq_max[1]:.3f} \t {ap_val_max[1]:.3f}"])

outputs.sort(key=lambda x:x[1])
#Arch \t 
print("PDQmax \t Score at PDQmax \t AP at PDQmax \t APmax")
for _, _, output in outputs:
    print(output)