import sys
import codecs
import json

def main():
    """
    
    """
    data = {
        "info" : "Dictionary of imageability scores for all words for which ytsvetko had vector embeddings for \
                    based on the MRC psycholinguistic database obtained experimentally by Wilson in 1988. Contains \
                    scores for 150,114 words",
        "words" : {}
    }

    for line in codecs.open("data/eval/raw_imageability.predictions", "r", "utf-8"):
        word, _, preds = line.strip().split("\t")
        preds_dict = eval(preds)

        data['words'][word] = preds_dict['A']

    with open("data/eval/imageability.json", "w") as file:
        json.dump(data, file)

if __name__ == "__main__":
    sys.exit(main())