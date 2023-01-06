import argparse
from embedding_triples import RDFTriples

parser = argparse.ArgumentParser()

parser.add_argument('--kbs', type=str, nargs='+', default=['carcinogenesis'], help='Knowledge base name')

for kb in parser.parse_args().kbs:
    triples = RDFTriples(source_kb_path=f'../datasets/{kb}/{kb}.owl')
    triples.export_triples()