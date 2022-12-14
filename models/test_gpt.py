import sys
import os
#Work around for relative import - clean up later
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from eval.clip_scores import CLIPEval
from eval.distinct_2 import DistinctEval
from eval.imageability import ImageabilityEval
from gpt_mlp.decoding import Decoding
from tqdm import tqdm

import skimage.io as io
import json
import numpy as np

def main():
    with open("data/test_poem2img.json", "r") as file:
        test_data = json.load(file)["poem2img"]
    
    distinct_eval = DistinctEval()
    imageability_eval = ImageabilityEval()
    clip_eval = CLIPEval()

    gen_poem_ut = Decoding("models/gpt_mlp/checkpoints/ki-009.pt")

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

    clip_pred = []
    
    for i in tqdm(range(len(test_data))):
        image = io.imread(test_data[i]["img_path"])
        pred_poem = "".join(gen_poem_ut.predict(image))

        # print("Poem {1}")
        results["Poems"].append(pred_poem)

        # print("==== CLIP Score ====")
        clip_score = clip_eval.score_poem(pred_poem, test_data[i]["poem"], image)
        results["CLIP Scores"].append(clip_score)
        clip_pred.append(clip_score["Predicted"])

        # print("==== Distinct-2 Score ====")
        distinct_score = distinct_eval.score_poem(pred_poem)
        results["Distinct-2 Scores"].append(distinct_score)

        # print("==== Imageability Score ====")
        imageability_score = imageability_eval.score_poem(pred_poem)
        results["Imageability Scores"].append(imageability_score)
    
    with open("models/gpt_mlp/scores/score_final.json", "w") as f:
        json.dump(results, f)
    
    results["Mean CLIP Score"] = np.mean(np.array(clip_pred))
    results["Median CLIP Score"] = np.median(np.array(clip_pred))
    results["Min CLIP Score"] = np.min(np.array(clip_pred))
    results["Max CLIP Score"] = np.max(np.array(clip_pred))

    results["Mean Distinct-2 Score"] = np.mean(np.array(results["Distinct-2 Scores"]))
    results["Median Distinct-2 Score"] = np.median(np.array(results["Distinct-2 Scores"]))
    results["Min Distinct-2 Score"] = np.min(np.array(results["Distinct-2 Scores"]))
    results["Max Distinct-2 Score"] = np.max(np.array(results["Distinct-2 Scores"]))

    results["Mean Imageability Score"] = np.mean(np.array(results["Imageability Scores"]))
    results["Median Imageability Score"] = np.median(np.array(results["Imageability Scores"]))
    results["Min Imageability Score"] = np.min(np.array(results["Imageability Scores"]))
    results["Max Imageability Score"] = np.max(np.array(results["Imageability Scores"]))

    with open("models/gpt_mlp/scores/score_final.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    sys.exit(main())


