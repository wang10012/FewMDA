import torch
from torch.nn import functional as F
import clip
from collections import OrderedDict
from descriptor_strings import *
from load import gpt_descriptions,load_gpt_descriptions

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_description_encodings(model):
    description_encodings = OrderedDict()
    for k, v in gpt_descriptions.items():
        tokens = clip.tokenize(v).to(device)
        description_encodings[k] = F.normalize(model.encode_text(tokens))
    return description_encodings

def compute_label_encodings(model):
    label_encodings = F.normalize(model.encode_text(clip.tokenize(['' + wordify(l) + '' for l in label_to_classname]).to(device)))
    return label_encodings

model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.eval()
model.requires_grad_(False)

for key,value in gpt_descriptions.items():
    print(key,value)
    print("*******************************************")

