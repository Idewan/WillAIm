import sys
import os
#Work around for relative import - clean up later
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from eval.clip_scores import CLIPEval
from eval.distinct_2 import DistinctEval
from eval.imageability import ImageabilityEval
from gpt_mlp.gen_poem_utils import GenPoemUtils

import skimage.io as io
import json

def main():
    with open("data/test_poem2img.json", "r") as file:
        test_data = json.load(file)["poem2img"]
    
    distinct_eval = DistinctEval()
    imageability_eval = ImageabilityEval()
    clip_eval = CLIPEval()

    gen_poem_ut = GenPoemUtils("models/gpt_mlp/checkpoints/ki-009.pt")
    
    image = io.imread(test_data[10]["img_path"])
    pred_poem = "".join(gen_poem_ut.predict(image))

    print("Poem {1}")
    print(pred_poem)
    print("==== CLIP Score ====")
    print(clip_eval.score_poem(pred_poem, test_data[10]["poem"], image))
    print("==== Distinct-2 Score ====")
    print(distinct_eval.score_poem(pred_poem[10]))
    print("==== Imageability Score ====")
    print(imageability_eval.score_poen(pred_poem[10]))

if __name__ == "__main__":
    sys.exit(main())


