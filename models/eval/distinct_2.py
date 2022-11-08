from paddlenlp.metrics import Distinct


class DistinctEval():

    def __init__(self):
        self.distinct = Distinct()

    def reset_distinct(self):
        self.distinct = Distinct()
    
    def score_poem(self, poem):
        """
            Scores the poem using Distinct-2 score:
            https://arxiv.org/abs/1510.03055
            :type poem: List[String]
            :return score: float
        """
        self.distinct.add_inst(poem)
        score = self.distinct.score()
        self.reset_distinct()

        return score