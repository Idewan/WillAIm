import json

class ImageabilityEval():
    """
        Similarly to Tsvetkov et al. (2013),
        I will use the dictionary created by them to classify metaphors. 
        Originally they built a lgositic regression model to to propagate
        abstractness and imageability scores from MRC
        ratings to all words for which we have vector space
        representations. More specifically, we calculate
        the degree of abstractness and imageability of all
        English items that have a vector space representation,
        using vector elements as features.
    """
    def __len__(self):
        return len(self.dict)

    def __init__(self):
        with open("data/eval/imageability.json", "r") as file:
            self.dict = json.load(file)
    
    def score_poem(self, poem):
        n = len(poem.split(" "))
        
        if n == 0:
            return 0

        sum_score = 0

        for w in poem:
            sum_score += self.dict.get(w, default=0)
        
        return sum_score / n