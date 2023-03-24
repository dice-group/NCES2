from typing import Final
import copy, sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('helper_classes')[0])
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from owlapy.render import DLSyntaxObjectRenderer

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit
from transformers import AutoTokenizer, PreTrainedTokenizerFast
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SimpleSolution:
    
    name: Final[str] = 'SimpleSolution'
    
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.renderer = DLSyntaxObjectRenderer()
        atomic_concepts = frozenset()
        self.atomic_concept_names = frozenset([self.renderer.render(a) for a in knowledge_base.ontology().classes_in_signature()])
        self.role_names = frozenset([r.get_iri().get_remainder() for r in knowledge_base.ontology().object_properties_in_signature()])
        Vocab = ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', ')', '(', '.'] + list(self.atomic_concept_names) + list(self.role_names)
        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.train_from_iterator(Vocab, trainer)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        self.tokenizer.pad_token = "[PAD]"
            
    def predict(self, expression: str):
        atomic_classes = [atm for atm in self.tokenizer.tokenize(expression) if atm in self.atomic_concept_names]
        if atomic_classes == []:
            atomic_classes =['⊤']
        return " ⊔ ".join(atomic_classes)
    
    
    
