import spacy

nlp = spacy.load('en_core_web_sm')


def spacy_tokenizer(doc):
    result = []

    for token in nlp(doc):
        if not token.is_punct and not token.is_space:
            lower_lemma = token.lemma_.lower()
            try:
                # ignore numbers
                int(lower_lemma)
            except ValueError:
                if all(c.isalnum() for c in lower_lemma):
                    result.append(lower_lemma)
    return result
