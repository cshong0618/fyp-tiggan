import torch
import time

from helper import generate_batch_images
import model

path = "./1518261086-model/g/_g.pkl"

g = model.G_rnn(11)

state_dict = torch.load(path)

for k in state_dict:
    print(k)

for k in g.state_dict():
    print(k)

state_dict = {k: state_dict[k] for k in g.state_dict()}

g.load_state_dict(state_dict)
g.cuda()
generate_batch_images(g, None, 10, figure_path="./batch_generate_%d" % (int(time.time())))
