import spacy

nlp = spacy.load('en_core_web_sm')


def spacy_tokenizer(doc):
    result = []
    for token in nlp(doc):
        if not token.is_punct and not token.is_space:
            result.append(token.lemma_.lower())
    return result
