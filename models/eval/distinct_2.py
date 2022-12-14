from paddlenlp.metrics import Distinct

import re

class DistinctEval():

    def __init__(self):
        self.distinct = Distinct()

    def reset_distinct(self):
        self.distinct = Distinct()
    
    def cleanPoem(self, poem):
        poem = poem.replace('\r', '')
        poem = poem.replace('\n', '')
        poem = re.sub(' +', ' ', poem.lower())
        return poem
    
    def score_poem(self, poem):
        """
            Scores the poem using Distinct-2 score:
            https://arxiv.org/abs/1510.03055
            :type poem: List[String]
            :return score: float
        """
        clean_poem = self.cleanPoem(poem)

        if clean_poem.isspace() or clean_poem == "" or clean_poem == "1":
            return 0

        self.distinct.add_inst(clean_poem)
        score = self.distinct.score()
        self.reset_distinct()

        return score