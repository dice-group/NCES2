def decompose(concept_name: str) -> list:
    """ Decomposes a class expression into a sequence of tokens (atoms) """
    def is_number(char):
        """ Checks if a character can be converted into a number """
        try:
            int(char)
            return True
        except:
            return False
    specials = ['⊔', '⊓', '∃', '∀', '¬', ' ', '(', ')',\
                '⁻', '≤', '≥', '{', '}', ':', '[', ']',
                'double', 'integer', 'xsd']
    list_ordered_pieces = []
    i = 0
    while i < len(concept_name):
        concept = ''
        while i < len(concept_name) and not concept_name[i] in specials:
            if concept_name[i] == '.' and not is_number(concept_name[i-1]):
                break
            concept += concept_name[i]
            i += 1
        if concept:
            list_ordered_pieces.append(concept)
        i += 1
    return list_ordered_pieces

def concept_length(concept_string):
    spec_chars = ['⊔', '⊓', '∃', '∀', '¬', '⁻', '≤', '≥']
    return len(decompose(concept_string)) + sum(map(concept_string.count, spec_chars))