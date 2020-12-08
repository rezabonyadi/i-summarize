from backend import docprovider
from backend.docsummarizer import DocSummarizer

def get_document():
    # address='data/text_data/singularity.txt'
    # address='data/text_data/employment.txt'
    # address='data/text_data/human_robot_interaction.txt'
    # address='data/text_data/bias.txt'
    # address='data/text_data/opacity.txt'
    # address='data/text_data/behaviour.txt'
    address='data/text_data/privacy.txt'
    
    # Returns paragraphs of a doc
    # doc = docprovider.get_from_page()
    doc = docprovider.get_from_txt(address)
    
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
per_paragraph = True

# sum_method = 'simple'
# summary = summarize_doc(doc, sum_method, per_paragraph=per_paragraph)

# sum_method = 'GPT2'
# summary = summarize_doc(doc, sum_method, per_paragraph=per_paragraph)

sum_method = 't5'
summary = summarize_doc(doc, sum_method, per_paragraph=per_paragraph)

# sum_method = 'xlnet'
# summary = summarize_doc(doc, sum_method, per_paragraph=per_paragraph)

# sum_method = 'bart'
# summary = summarize_doc(doc, sum_method, per_paragraph=per_paragraph)


print('Using ', sum_method, ' method.')
print('Here is the summary: ')

if per_paragraph:
    for s in summary:
        print(s)
        print()
else:
    print(summary)


# actions = get_actions(doc, act_method)

