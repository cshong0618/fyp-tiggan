import torch
import time
import argparse
import sys

from helper import generate_batch_images
import model

parser = argparse.ArgumentParser()

parser.add_argument("-M", "--model_path", help='Model path', default='')

args = parser.parse_args(sys.argv[1:])

if args.model_path != "":
    path = args.model_path
else:
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
