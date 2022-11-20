import torch
import sys

path = sys.argv[1]
out_path = '/'.join(path.split('/')[:-1] + ['pytorch_model.bin'])
print (out_path)

composer_ckpt = torch.load(path, map_location=torch.device('cpu'))
state_dict = composer_ckpt['state']['model']
torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, prefix='model.')
torch.save(state_dict, out_path)
