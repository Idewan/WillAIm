import sys
import os
#Work around for relative import - clean up later
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from eval.clip_scores import CLIPEval
from eval.distinct_2 import DistinctEval
from eval.imageability import ImageabilityEval
from gpt_mlp.gen_poem_utils import GenPoemUtils
from tqdm import tqdm

import skimage.io as io
import json

def main():
    with open("data/test_poem2img.json", "r") as file:
        test_data = json.load(file)["poem2img"]
    
    distinct_eval = DistinctEval()
    imageability_eval = ImageabilityEval()
    clip_eval = CLIPEval()

    gen_poem_ut = GenPoemUtils("models/gpt_mlp/checkpoints/ki-009.pt")

    results = {
        "Mean CLIP Score" : 0,
        "Median CLIP Score" : 0,
        "Max CLIP Score" : 0,
        "Min CLIP Score" : 0,
        "Mean Distinct-2 Score" : 0,
        "Median Distinct-2 Score" : 0,
        "Max Distinct-2 Score" : 0,
        "Min Distinct-2 Score" : 0,
        "Mean Imageability Score" : 0,
        "Median Imageability Score" : 0,
        "Max Imageability Score" : 0,
        "Min Imageability Score" : 0,
        "Poems" : [],
        "CLIP Scores" : [],
        "Distinct-2 Scores" : [],
        "Imageability Scores" : []
    }
    
    for i in tqdm(range(10)):
        image = io.imread(test_data[i]["img_path"])
        pred_poem = "".join(gen_poem_ut.predict(image))

        print(pred_poem)
        print(len(pred_poem))
        print(test_data[i]["poem"])
        print(len(test_data[i]["poem"]))

        # print("Poem {1}")
        results["Poems"].append(pred_poem)
        # print("==== CLIP Score ====")
        clip_score = clip_eval.score_poem(pred_poem, test_data[i]["poem"], image)
        results["CLIP Scores"].append(clip_score[0])
        # print("==== Distinct-2 Score ====")
        distinct_score = distinct_eval.score_poem(pred_poem)
        results["Distinct-2 Scores"].append(distinct_score)
        # print("==== Imageability Score ====")
        imageability_score = imageability_eval.score_poen(pred_poem)
        results["Imageability Scores"].append(imageability_score)
    
    with open("models/gpt_mlp/scores", "w") as f:
        json.dump(results, f)
    
    # results["Mean CLIP Score"] = 

if __name__ == "__main__":
    sys.exit(main())


