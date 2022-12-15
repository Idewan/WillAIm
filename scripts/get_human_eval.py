import json
import sys
import os


def main():
    eval_ = {
        "gpt" : [],
        "multim" : [],
        "transformer" : [],
        "mlp" : []
    }

    #Open all relevant scores
    indices = [4, 5, 34, 113, 224, 288, 301, 355, 385,
                505, 538, 541, 554, 575, 685]
    
    gpt_poems = []
    multi_m_poems = []
    transformer_poems = []
    mlp_poems = []

    #Get GPT poems
    with open("models/gpt3/scores/score_final.json", "r") as f:
        gpt = json.load(f)["Poems"]
    with open("models/m_adv/scores/score_final.json", "r") as f:
        multim = json.load(f)["Poems"]
    with open("models/gpt_mlp/scores/score_final.json", "r") as f:
        mlp = json.load(f)["Poems"]
    with open("models/gpt_transformer/scores/score_nws_final.json", "r") as f:
        transformer = json.load(f)["Poems"]

    for i in indices:
        gpt_poems.append(gpt[i])
        multi_m_poems.append(multim[i])
        mlp_poems.append(mlp[i])
        transformer_poems.append(transformer[i])

    eval_["gpt"] = gpt_poems
    eval_["multim"] = multi_m_poems
    eval_["transformer"] = transformer_poems
    eval_["mlp"] = mlp_poems

    with open("data/eval/poems.json", "w") as f:
        json.dump(eval_, f, ensure_ascii=False)
    
    print("Done!")

if __name__ == "__main__":
    sys.exit(main())