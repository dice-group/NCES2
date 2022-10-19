from typing import Final


class F1:
    __slots__ = ()

    name: Final = 'F1'
    
    @staticmethod
    def score(pos_examples:set = {}, instances:set = {}):
        if len(instances) == 0:
            return 0.

        tp = len(pos_examples.intersection(instances))
        # tn = len(pos_examples.difference(instances))

        fp = len(instances.difference(pos_examples))
        fn = len(pos_examples.difference(instances))

        try:
            recall = float(tp) / (tp + fn)
        except ZeroDivisionError:
            return 0.

        try:
            precision = float(tp) / (tp + fp)
        except ZeroDivisionError:
            return 0.

        if precision == 0. or recall == 0.:
            return 0.

        f_1 = 2 * ((precision * recall) / (precision + recall))
        return round(f_1, 5)
    
    
class Accuracy:
    """
    Accuracy is          acc = (tp + tn) / (tp + tn + fp+ fn). However,
    Concept learning papers (e.g. Learning OWL Class expression) appear to invent their own accuracy metrics.

    In OCEL =>    Accuracy of a concept = 1 - ( \\|E^+ \ R(C)\\|+ \\|E^- AND R(C)\\|) / \\|E\\|)


    In CELOE  =>    Accuracy of a concept C = 1 - ( \\|R(A) \ R(C)\\| + \\|R(C) \ R(A)\\|)/n

    1) R(.) is the retrieval function, A is the class to describe and C in CELOE.

    2) E^+ and E^- are the positive and negative examples provided. E = E^+ OR E^- .
    """
    __slots__ = ()

    name: Final = 'Accuracy'
    
    @staticmethod
    def score(pos_examples:set = {}, neg_examples = {}, instances:set = {}, all_individuals:set = {}) -> float:
        
        if len(instances) == 0:
            return 0.

        tp = len(pos_examples.intersection(instances))
        tn = len(neg_examples.intersection(all_individuals-instances))
        fp = len(instances.difference(pos_examples))
        fn = len(pos_examples.difference(instances))

        acc = (tp + tn) / (tp + tn + fp + fn)

        return round(acc, 5)