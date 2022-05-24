class Data:
    def __init__(self, kwargs):
        """
        Load data from triples.
        """
        self.info = {'dataset': kwargs['path_to_triples']}

        self.data = self.load_data(kwargs['path_to_triples'], data_type="train")
        self.entities = self.get_entities(self.data)
        self.relations = self.get_relations(self.data)
        self.entity_to_idx = self.__entity_to_idx()
        self.relation_to_idx = self.__relation_to_idx()

    @staticmethod
    def load_data(data_dir, data_type):
        try:
            with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                data = f.read().strip().split("\n")
                data = [i.split("\t") for i in data if len(i.split("\t"))==3]
        except FileNotFoundError as e:
            print(e)
            print('Add empty.')
            data = []
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(set([d[1] for d in data]))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(set([d[0] for d in data] + [d[2] for d in data]))
        return entities
    
    def __entity_to_idx(self):
        return {e: idx for idx,e in enumerate(self.entities)}
    
    def __relation_to_idx(self):
        return {r: idx for idx,r in enumerate(self.relations)}
        
