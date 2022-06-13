from typing import Final
import copy, sys, os, numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('helper_classes')[0])
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from owlapy.render import DLSyntaxObjectRenderer
from metrics import Accuracy, F1

class SyntaxChecker:
    '''-Python class for checking and validating/correcting/suggesting DL class expressions
       -Includes methods that can query the instances of a class expression and compute the f-measure w.r.t. positive and negative examples    
       -Requires a KnowledgeBase object (see ontolearn (a.k.a OntoPy) library)
    '''
    
    name: Final[str] = 'Syntax Checker'
    
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.renderer = DLSyntaxObjectRenderer()
        atomic_concepts = frozenset(knowledge_base.ontology().classes_in_signature())
        self.concept_str_to_concept = {self.renderer.render(a): a for a in atomic_concepts}
        self.concept_str_to_concept.update({'⊤': knowledge_base.thing, '⊥': knowledge_base.nothing})
        self.role_str_to_role = {r.get_iri().get_remainder(): r for r in knowledge_base.ontology().object_properties_in_signature()}
        self.role_names = frozenset(self.role_str_to_role.keys())
        self.atomic_concept_names = frozenset([self.renderer.render(a) for a in atomic_concepts])
        self.atoms = self.role_names.union(self.atomic_concept_names).union({'⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' '})
        
    def split(self, atom):
        if atom in self.atoms:
            return [atom]
        piece = ''
        for j in range(len(atom)):
            if not piece in self.atoms:
                piece += atom[j]
            else:
                return [piece]+split(atom[j:])
        return (False, [atom])
    
    def split2(self, expression):
        def longer_atom_exists(a, expression, lookahead = 20, start_index=0):
            lst = [a+expression[start_index:start_index+i+1] for i in range(lookahead)]
            booleans = [a+expression[start_index:start_index+i+1] in self.atoms for i in range(lookahead)]
            return len(set(lst))>1 and any(booleans)
        pieces = []
        piece = ''
        for i in range(len(expression)):
            a = expression[i]
            piece += a
            if piece in self.atoms:
                try:
                    if not longer_atom_exists(piece, expression, lookahead=30, start_index=i+1):
                        pieces.append(piece); piece = ''
                except IndexError:
                    pieces.append(piece)
                    piece = ''
        return pieces

    def preprocess(self, expression:str):    
        spec_chars = ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.']
        for char in spec_chars:
            expression = expression.replace(char+char, char)
            expression = expression.replace(char+' '+char, char)
            expression = expression.replace(char+'.', char)
            expression = expression.replace(char+' .', char)
            if not char in ['⊤', '⊥']:
                expression = expression.replace('.'+char, char)
                expression = expression.replace('. '+char, char)
        expression = expression.replace('⊔⊓', '⊔'); expression = expression.replace('⊓⊔', '⊓')
        expression = expression.replace('⊔ ⊓', '⊔'); expression = expression.replace('⊓ ⊔', '⊓')
        expression = expression.replace('⊔ ⊔', '⊔'); expression = expression.replace('⊓ ⊓', '⊓')
        expression = expression.replace('∃∀', '∃'); expression = expression.replace('∀∃', '∀')
        expression = expression.replace('∃ ∀', '∃'); expression = expression.replace('∀ ∃', '∀')
        dec_expression = self.split2(expression)
        dec_expression = [atm for atm in dec_expression if atm != ' ']
        filtered_atoms = []
        for i in range(len(dec_expression)):
            if i == len(dec_expression)-1 and dec_expression[i] in set(spec_chars)-{'⊤', '⊥', '.'}:
                continue
            if i == 0 and dec_expression[i] in {'⊔', '⊓', '⊥', '.'}:
                continue
            atom = dec_expression[i]
            if filtered_atoms == []:
                if atom in ['∃', '∀'] and i < len(dec_expression)-1 and not dec_expression[i+1] in self.role_names:
                    continue
                elif atom in self.role_names and i < len(dec_expression)-1 and dec_expression[i+1] != '.':
                    filtered_atoms.extend(['∃', atom, '.'])
                elif atom in self.role_names and i < len(dec_expression)-1 and dec_expression[i+1] == '.':
                    filtered_atoms.extend(['∃', atom])
                elif atom in self.role_names and i == len(dec_expression)-1:
                    filtered_atoms.extend(['∃', atom, '.', '⊤'])
                else:
                    filtered_atoms.append(atom)
            elif atom in self.role_names:
                try:
                    if not filtered_atoms[-2] in ['⊔', '⊓']:
                        continue
                except IndexError:
                    pass
                if not filtered_atoms[-1] in ['∃', '∀']:
                    continue
                if filtered_atoms[-1] in ['∃', '∀']:
                    filtered_atoms.append(atom)
                if i < len(dec_expression)-1:
                    if not dec_expression[i+1] in self.atomic_concept_names:
                        if dec_expression[i+1] == '.':
                            continue
                        elif dec_expression[i+1] in ['∃', '∀', '¬', '⊤', '⊥']:
                            filtered_atoms.append('.')
                        else:
                            filtered_atoms.extend(['.', '⊤'])
                    else:
                        filtered_atoms.append('.')
                elif i == len(dec_expression)-1:
                    if filtered_atoms == []:
                        filtered_atoms.append('⊤')
                    else:
                        filtered_atoms.extend(['.', '⊤'])
            elif atom in ['⊔', '⊓']:
                if i==len(dec_expression)-1 or not filtered_atoms[-1] in self.atomic_concept_names.union({'⊤', '⊥'}):
                    continue
                if i<len(dec_expression)-1 and not dec_expression[i+1] in self.atomic_concept_names.union({'∃', '∀'}):
                    continue
                filtered_atoms.append(atom)
            elif atom in self.atomic_concept_names.union({'⊤', '⊥'}):
                if filtered_atoms[-1] in self.role_names:
                    filtered_atoms.extend(['.', atom])
                elif filtered_atoms[-1] in ['.', '⊔', '⊓']:
                    filtered_atoms.append(atom)
            elif atom == '¬':
                if i<len(dec_expression)-1 and not dec_expression[i+1] in self.atomic_concept_names:
                    continue
                else:
                    filtered_atoms.append(atom)
            elif atom in ['∃', '∀']:
                if not dec_expression[i+1] in self.role_names:
                    continue
                if not filtered_atoms[-1] in ['⊔', '⊓']:
                    continue
                if i<len(dec_expression)-1:
                    filtered_atoms.append(atom)
            elif atom == '.':
                if not filtered_atoms[-1] in self.role_names:
                    continue
                if i==len(dec_expression)-1:
                    filtered_atoms.extend(['.','⊤']); continue
                else:
                    filtered_atoms.append(atom)
                    if (i<len(dec_expression)-1 and not dec_expression[i+1] in self.atomic_concept_names.union({'⊤', '⊥', '∃', '∀', '¬'})) or i == len(dec_expression)-1:
                        filtered_atoms.append('⊤')
        return filtered_atoms
    
    def correct(self, expression: str, max_num_trials = 5):
#        print('start: ', expression)
        if set(self.split2(expression)).issubset({'⊔', '⊓', '∃', '∀', '¬', '.', ' '}):
            expression = '⊤'
        seq_prev = ''
        seq = expression
        count = 0
        while seq != seq_prev:
            seq_prev = seq
            seq = ' '.join(self.preprocess(seq))
            count += 1
            if count == max_num_trials: break
        return seq.split()
    
    def get_suggestions(self, expression_list):
        simple_expr = all([any([not expression_list[i-1] in self.atomic_concept_names.union({'⊤', '⊥'}), not expression_list[i+1] in\
                        self.atomic_concept_names.union({'⊤', '⊥'})]) for i in range(len(expression_list)) if expression_list[i] in ['⊔', '⊓']]) or\
        (set(expression_list).issubset(self.atomic_concept_names.union({'⊔', '⊓'})) and expression_list.count('⊔')*expression_list.count('⊓') == 0) # if there is no need to
#       add parentheses, e.g union and intersection only appear in the form A ⊔ ∃ r.C or there is not mix of ⊔ and ⊓
        if len(expression_list)<=4 or simple_expr:
            yield expression_list
            return
        def put_parentheses(expr, rng=3):
            return ['(']+[expr[i] for i in range(rng)]+[')']
        
        possible_exprs = [expression_list]
# left priority
        expr = []
        Expr = []
        i = 0
        while i < len(expression_list):
            if not expression_list[i] in ['⊔', '⊓']:
                if expression_list[i] in self.atomic_concept_names.union({'⊤', '⊥'}) and i<len(expression_list)-1:
                    expr.append(expression_list[i])
                else:
                    Expr.append(expression_list[i])
                i += 1
            else:
                if  Expr and Expr[-1] == ')' or expr==[]:
                    Expr.append(expression_list[i])
                    expr = []
                    i += 1
                elif i < len(expression_list)-1 and not expression_list[i+1] in self.atomic_concept_names.union({'⊤', '⊥'}):
                    Expr.extend(expr+[expression_list[i]])
                    expr = []
                    i += 1
                elif i < len(expression_list)-1:
                    expr.extend([expression_list[i], expression_list[i+1]])
                    Expr.extend(put_parentheses(expr))
                    i += 2
        if Expr != expression_list:
            possible_exprs.append(Expr)

# right priority
        expr = []
        Expr = []
        i = 0
        while i < len(expression_list):
            if not expression_list[i] in ['⊔', '⊓']:
                if expression_list[i] in self.atomic_concept_names.union({'⊤', '⊥'}) and 0<i<len(expression_list)-1:
                    expr.append(expression_list[i])
                else:
                    Expr.append(expression_list[i])
                i += 1
            else:
                if Expr and Expr[-1] == ')' or expr==[]:
                    Expr.append(expression_list[i])
                    expr = []
                    i += 1
                elif i < len(expression_list)-1 and not expression_list[i+1] in self.atomic_concept_names.union({'⊤', '⊥'}):
                    Expr.extend(expr+[expression_list[i]])
                    expr = []
                    i += 1
                elif i < len(expression_list)-1:
                    expr.extend([expression_list[i], expression_list[i+1]])
                    Expr.extend(put_parentheses(expr))
                    i += 2
        if not Expr in possible_exprs:
            possible_exprs.append(Expr)
        yield from possible_exprs

    def recursive_concept_constructor(self, expression_list:list):
        if len(expression_list) == 1:
            assert expression_list[0] in self.atomic_concept_names.union({'⊤', '⊥'})
            return [self.concept_str_to_concept[expression_list[0]]]
        if expression_list[0] == '∃':
            assert expression_list[1] in self.role_names and expression_list[2] == '.' and expression_list[3] in self.atomic_concept_names.union({'⊤', '⊥', '¬', '∃', '∀'}), 'Invalid class expression'
            if expression_list[3] in ['∃', '∀']:
                filler = self.recursive_concept_constructor(expression_list[3:7])[0]
            elif expression_list[3] == '¬':
                filler = self.recursive_concept_constructor(expression_list[3:5])[0]
            else:
                filler = self.concept_str_to_concept[expression_list[3]]
            prop = self.role_str_to_role[expression_list[1]]
            if len(expression_list)>4 and not expression_list[3] in ['∃', '∀']:
                return [self.knowledge_base.existential_restriction(filler, prop)] + self.recursive_concept_constructor(expression_list[4:])
            elif len(expression_list)<=4:
                return [self.knowledge_base.existential_restriction(filler, prop)]
            elif expression_list[3] in ['∃', '∀']:
                return [self.knowledge_base.existential_restriction(filler, prop)] + self.recursive_concept_constructor(expression_list[7:])
        if expression_list[0] == '∀':
            assert expression_list[1] in self.role_names and expression_list[2] == '.' and expression_list[3] in self.atomic_concept_names.union({'⊤', '⊥'}), 'Invalid class expression'
            if expression_list[3] in ['∃', '∀']:
                filler = self.recursive_concept_constructor(expression_list[3:7])[0]
            elif expression_list[3] == '¬':
                filler = self.recursive_concept_constructor(expression_list[3:5])[0]
            else:
                filler = self.concept_str_to_concept[expression_list[3]]
            prop = self.role_str_to_role[expression_list[1]]
            if len(expression_list)>4 and not expression_list[3] in ['∃', '∀']:
                return [self.knowledge_base.universal_restriction(filler, prop)] + self.recursive_concept_constructor(expression_list[4:])
            elif len(expression_list)<=4:
                return [self.knowledge_base.universal_restriction(filler, prop)]
            elif expression_list[3] in ['∃', '∀']:
                return [self.knowledge_base.universal_restriction(filler, prop)] + self.recursive_concept_constructor(expression_list[7:])
        if expression_list[0] in ['⊔', '⊓']:
            assert len(expression_list) > 1, 'Invalid class expression'
            return self.recursive_concept_constructor(expression_list[1:])
        if len(expression_list)>1 and expression_list[0] in self.atomic_concept_names.union({'⊤', '⊥'}):
            return [self.concept_str_to_concept[expression_list[0]]] + self.recursive_concept_constructor(expression_list[1:])
        if expression_list[0] == '¬':
            assert len(expression_list)>1 and expression_list[1] in self.atomic_concept_names.union({'⊤', '⊥'}), 'Invalid class expression'
            return [self.knowledge_base.negation(expression_list[1])] + self.recursive_concept_constructor(expression_list[2:])
        raise ValueError
        
    def get_concept(self, expression_list:list):
        def disj_conj_concept_builder(atom_list):
            assert len(atom_list) == 5 and ('⊓' in atom_list or '⊔' in atom_list) , 'Incorrect form'
            if '⊓' in atom_list:
                return self.knowledge_base.intersection([self.concept_str_to_concept[atom_list[1]], self.concept_str_to_concept[atom_list[3]]])
            return self.knowledge_base.union([self.concept_str_to_concept[atom_list[1]], self.concept_str_to_concept[atom_list[3]]])
        def get_block_concept(start_idx, expr):
            assert expr[start_idx] == '(', 'Missing parentheses, invalid block'
            out = []
            j = start_idx
            while expr[j] != ')':
                out.append(expr[j])
                j += 1
            return disj_conj_concept_builder(out + [')'])
        i = 0
        current_expr = None
        join = None
        while i < len(expression_list):
            if expression_list[i] in ['⊔', '⊓']:
                join = expression_list[i]
                i += 1
            elif expression_list[i] in self.atomic_concept_names.union({'⊤', '⊥'}):
                c = self.concept_str_to_concept[expression_list[i]]
                if current_expr is None:
                    current_expr = c
                else:
                    assert join, 'No join'
                    current_expr = self.knowledge_base.union([current_expr, c]) if join=='⊔' else self.knowledge_base.intersection([current_expr, c])
                i += 1
            elif expression_list[i] == '(':
                ce = get_block_concept(i, expression_list) # ce: class expression
                if current_expr is None:
                    current_expr = ce
                else:
                    assert join, 'No join'
                    current_expr = self.knowledge_base.union([current_expr, ce]) if join == '⊔' else self.knowledge_base.intersection([current_expr, ce])
                i += 5
            elif expression_list[i] in ['∃', '∀']:
                if expression_list[i+3] in self.atomic_concept_names.union({'⊤', '⊥'}):
                    filler = self.concept_str_to_concept[expression_list[i+3]]
                    prop = self.role_str_to_role[expression_list[i+1]]
                    ce = self.knowledge_base.existential_restriction(filler, prop) if expression_list[i] == '∃' else self.knowledge_base.universal_restriction(filler, prop)
                    if current_expr is None:
                        current_expr = ce
                    else:
                        assert join, 'No join'
                        current_expr = self.knowledge_base.union([current_expr, ce]) if join == '⊔' else self.knowledge_base.intersection([current_expr, ce])
                    i += 4
                elif expression_list[i+3] == '¬':
                    filler = self.knowledge_base.negation(self.concept_str_to_concept[expression_list[i+4]])
                    prop = self.role_str_to_role[expression_list[i+1]]
                    ce = self.knowledge_base.existential_restriction(filler, prop) if expression_list[i] == '∃' else self.knowledge_base.universal_restriction(filler, prop)
                    if current_expr is None:
                        current_expr = ce
                    else:
                        assert join, 'No join'
                        current_expr = self.knowledge_base.union([current_expr, ce]) if join == '⊔' else self.knowledge_base.intersection([current_expr, ce])
                    i += 5
                elif expression_list[i+3] == '(':
                    filler = get_block_concept(i+3, expression_list)
                    prop = self.role_str_to_role[expression_list[i+1]]
                    ce = self.knowledge_base.existential_restriction(filler, prop) if expression_list[i] == '∃' else self.knowledge_base.universal_restriction(filler, prop)
                    if current_expr is None:
                        current_expr = ce
                    else:
                        assert join, 'No join'
                        current_expr = self.knowledge_base.union([current_expr, ce]) if join == '⊔' else self.knowledge_base.intersection([current_expr, ce])
                    i += 8
                else:
                    raise ValueError
            else:
                raise ValueError
        return current_expr
                    
            
    def concept(self, expression: str):
#        def get_conjunctions(components_list, joins):
#            i = 0
#            conjunctions = []
#            while i<len(joins)+1:
#                j = i
#                conj = [components_list[j]]
#                while j<len(joins) and joins[j] == '⊓':
#                    conj += [components_list[j+1]]
#                    j += 1
#                if len(conj) == 1:
#                    conjunctions.append(conj[0])
#                else:
#                    conjunctions.append(self.knowledge_base.intersection(conj))
#                i = j+1
#            return conjunctions
#        def get_disjunctions(components_list, joins):
#            i = 0
#            disjunctions = []
#            while i<len(joins)+1:
#                j = i
#                disj = [components_list[j]]
#                while j<len(joins) and joins[j] == '⊔':
#                    disj += [components_list[j+1]]
#                    j += 1
#                if len(disj) == 1:
#                    disjunctions.append(disj[0])
#                else:
#                    disjunctions.append(self.knowledge_base.union(disj))
#                i = j+1
#            return disjunctions

        list_atoms = self.correct(expression)
        possible_concepts = list(self.get_suggestions(list_atoms))
#        print('Possible CEs: ', possible_concepts)
        self.class_expressions = []
        for c in possible_concepts:
            ce = self.get_concept(c)
            self.class_expressions += [ce]
#            else:
#                concept_components = self.recursive_concept_constructor(list_atoms)
#                joins = [j for j in list_atoms if j in ['⊔', '⊓']]
#                assert len(joins) == len(concept_components) - 1, 'Cannot perform joins'
#                self.class_expressions += [self.knowledge_base.union(get_conjunctions(concept_components, joins))]
        return self
    
    def evaluate(self, pos_examples, verbose = True):
        Instances = []
        all_individuals = set(self.knowledge_base.individuals())
        for ce in self.class_expressions:
            Instances += [{ind.get_iri().as_str().split("/")[-1] for ind in self.knowledge_base.individuals(ce)}]
        #instances += [{ind.get_iri().as_str().split("/")[-1] for ind in self.knowledge_base.individuals(self.class_expression[1])}]
        f1 = [F1.score(set(pos_examples), instances) for instances in Instances]
        acc = [Accuracy.score(set(pos_examples), instances, all_individuals) for instances in Instances]
        best_concept = self.renderer.render(self.class_expressions[np.argmax(f1)])
        if verbose:
            print("Accuracy: {}%".format([100*s for s in acc]))
            print("F1 score: {}%".format([100*s for s in f1]))
        return best_concept, 100*max(acc), 100*max(f1)
        
                          
                          
                          
                          
        
        
        
        