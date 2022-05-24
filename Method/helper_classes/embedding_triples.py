from rdflib import graph
import os

class RDFTriples:
    """The knowledge graph/base is converted into triples of the form: individual_i ---role_j---> concept_k and stored in a txt file for the computation of embeddings."""
    
    def __init__(self, source_kg_path):
        self.Graph = graph.Graph()
        self.Graph.load(source_kg_path)
        self.source_kg_path = source_kg_path
              
    def export_triples(self, export_folder_name='Triples'):
        if not os.path.exists(os.path.join(self.source_kg_path[:self.source_kg_path.rfind("/")], export_folder_name)):
            os.mkdir(os.path.join(self.source_kg_path[:self.source_kg_path.rfind("/")], export_folder_name))
        if os.path.isfile(os.path.join(self.source_kg_path[:self.source_kg_path.rfind("/")], export_folder_name, "train.txt")):
            return
        train_file = open("%s/train.txt" % os.path.join(self.source_kg_path[:self.source_kg_path.rfind("/")], export_folder_name), mode="w")
        for s,p,o in self.Graph:
            s = s.expandtabs()[s.expandtabs().rfind("/")+1:]
            p = p.expandtabs()[p.expandtabs().rfind("/")+1:]
            o = o.expandtabs()[o.expandtabs().rfind("/")+1:]
            if s and p and o:
                train_file.write(s+"\t"+p+"\t"+o+"\n")
        train_file.close()
        print("*********************Finished exporting triples*********************\n")