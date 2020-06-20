from backend import docprovider
from backend.docsummarizer import DocSummarizer

def get_document():
    # Returns paragraphs of a doc
    doc = docprovider.get_from_page()
    return doc

def summarize_doc(doc, method, per_paragraph=True):
    summarizer = DocSummarizer(method)
    summary = []
    
    if per_paragraph:
        # Summarize per paragraph
        i = 0
        for p in doc:
            summary.append(summarizer.summarize(p))
            print('\r', str(i), end='')
            i += 1
    else:
        # Summarize as a whole
        total_doc = ''
        for p in doc:
            total_doc += (p + '\n')
        summary = summarizer.summarize(total_doc)
    return summary

doc = get_document()

# sum_method = 'simple'
# summary = summarize_doc(doc, sum_method)
# print('Using ', sum_method, ' method.')
# print('Here is the summary: ', summary)


# sum_method = 'GPT2'
# summary = summarize_doc(doc, sum_method, per_paragraph=False)
# print('Using ', sum_method, ' method.')
# print('Here is the summary: ', summary)

sum_method = 't5'
summary = summarize_doc(doc, sum_method, per_paragraph=False)
print('Using ', sum_method, ' method.')
print('Here is the summary: ', summary)


# sum_method = 'xlnet'
# summary = summarize_doc(doc, sum_method, per_paragraph=False)
# print('Using ', sum_method, ' method.')
# print('Here is the summary: ', summary)

# sum_method = 'bart'
# summary = summarize_doc(doc[8], sum_method)
# print('Using ', sum_method, ' method.')
# print('Here is the summary: ', summary)

# actions = get_actions(doc, act_method)

