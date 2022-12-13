import sys
import os
#Work around for relative import - clean up later
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from gpt_mlp.decoding import Decoding
from gpt_mlp.gen_poem_utils import GenPoemUtils
from tqdm import tqdm

import skimage.io as io
import json
import numpy as np

if __name__ == '__main__':
    # decoding_ut = Decoding("models/gpt_mlp/checkpoints/ki-009.pt")
    gen_poem_ut = GenPoemUtils("models/gpt_mlp/checkpoints/ki-009.pt")

    image = io.imread("data/image/13658/0.png")
    pred_poem = "".join(gen_poem_ut.predict(image))

    print(pred_poem)