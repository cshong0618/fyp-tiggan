import torch
import time

from label_noise_gan import G, generate_batch_images

path = "./model/g/_g.pkl"

model = G(11)

state_dict = torch.load(path)

for k in state_dict:
    print(k)

for k in model.state_dict():
    print(k)

state_dict = {k: state_dict[k] for k in model.state_dict()}

model.load_state_dict(state_dict)

generate_batch_images(model, None, 10, figure_path="./batch_generate_%d" % (int(time.time())))
