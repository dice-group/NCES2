import pandas as pd

class Data:
    def __init__(self, kwargs):
        
        """
        Read triples into a list of lists
        """
        self.info = {'dataset': kwargs.path_to_triples}

        train_data = self.load_data(kwargs.path_to_triples, data_type="train")
        #self.valid_data = self.load_data(kwargs.path_to_triples, data_type="valid")
        #self.test_data = self.load_data(kwargs.path_to_triples, data_type="test")
        self.data_triples = train_data # + self.valid_data + self.test_data
        # The order of entities is important
        self.entities = self.get_entities(self.data_triples)
        train_relations = self.get_relations(self.data_triples)
        #self.valid_relations = self.get_relations(self.valid_data)
        #self.test_relations = self.get_relations(self.test_data)
        # The order of entities is important
        self.relations = train_relations #+ [i for i in self.valid_relations \
                                               #  if i not in self.train_relations] + [i for i in self.test_relations \
                                               #                                       if i not in self.train_relations]
        self.entity2idx = pd.DataFrame(list(range(len(self.entities))), index=self.entities)
        self.relation2idx = pd.DataFrame(list(range(len(self.relations))), index=self.relations)

    @staticmethod
    def load_data(data_dir, data_type):
        try:
            with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                data = f.read().strip().split("\n")
                data = [i.split("\t") for i in data if len(i.split("\t"))==3]
        except FileNotFoundError as e:
            data = []
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities
