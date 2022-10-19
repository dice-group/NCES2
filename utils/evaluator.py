from .metrics import Accuracy, F1

class Evaluator:
    def __init__(self, kb):
        self.kb = kb
        
    def evaluate(self, prediction, pos_examples, neg_examples):
        all_individuals = set(self.kb.individuals())
        instances ={ind.get_iri().as_str().split("/")[-1] for ind in self.kb.individuals(prediction)}
        f1 = F1.score(pos_examples, instances)
        acc = Accuracy.score(pos_examples, neg_examples, instances, all_individuals)
        return 100*acc, 100*f1