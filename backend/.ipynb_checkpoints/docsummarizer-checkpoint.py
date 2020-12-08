import nltk
import heapq
import re
from summarizer import Summarizer,TransformerSummarizer
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np


class DocSummarizer:
    def __init__(self, sum_method):
        super().__init__()
        self.summarization_methods = {'simple': self.simple_summarizer, 
                                 'GPT2': self.gpt2_summarizer, 
                                 'xlnet': self.xlnet_summarizer, 
                                 'bart': self.bart_summarizer, 
                                 't5': self.t5_summarizer}

        
        # Not using dictionary to avoid creating models that are not going to be used
        if sum_method == 'GPT2':
            self.model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
        if sum_method == 'xlnet':
            self.model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
        if sum_method == 'bart':
            self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        if sum_method == 't5':
            self.model = T5ForConditionalGeneration.from_pretrained('t5-small')

        self.sum_method_name = sum_method
        self.summarization_method = self.summarization_methods[sum_method]


    def summarize(self, doc: str, sum_strength=10.0):        
        num_sentences = len(nltk.sent_tokenize(doc))
        min_n_sen_summary = int(num_sentences*sum_strength/100.0) + 1
        num_words = len(re.findall(r'\w+', doc)) 

        min_n_words_summary = int(np.min([(num_words*(sum_strength-5.0)/100.0)+5, 30]))
        max_n_sen_summary = int(np.min([(num_words*(sum_strength+5.0)/100.0)+5, 500]))

        settings = {'simple': {'n_sentences': min_n_sen_summary}, 
                    'GPT2': {'min_length': min_n_words_summary, 'max_length': max_n_sen_summary}, 
                    'xlnet': {'min_length': min_n_words_summary, 'max_length': max_n_sen_summary},
                    'bart': {'min_length': min_n_words_summary, 'max_length': max_n_sen_summary},
                    't5': {'min_length': min_n_words_summary, 'max_length': max_n_sen_summary},
                    }

        self.summary = self.summarization_method(doc, settings[self.sum_method_name])
        return self.summary        

    def simple_summarizer(self, article_text, settings):
        n_sentences = settings['n_sentences']
        # Removing special characters and digits
        formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

        # Convert text to sentences
        sentence_list = nltk.sent_tokenize(article_text)


        # Find Weighted Frequency of Occurrence

        stopwords = nltk.corpus.stopwords.words('english')

        word_frequencies = {}
        for word in nltk.word_tokenize(formatted_article_text):
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        # Finally, to find the weighted frequency, we can simply divide the number of occurances of all the words by 
        # the frequency of the most occurring word, as shown below:            
        maximum_frequncy = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
        
        # Calculating Sentence Scores
        sentence_scores = {}
        for sent in sentence_list:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]

        # Getting the Summary

        summary_sentences = heapq.nlargest(n_sentences, sentence_scores, key=sentence_scores.get)

        summary = ' '.join(summary_sentences)
        return summary
    
    def gpt2_summarizer(self, doc, settings):
        # min_length = settings['min_length'] 
        max_length = settings['max_length']        
        full = ''.join(self.model(doc, min_length=20, max_length=max_length))
        return full

    def xlnet_summarizer(self, doc, settings):
        min_length = settings['min_length'] 
        max_length = settings['max_length']
        full = ''.join(self.model(doc, min_length=20, max_length=max_length))
        return full
    
    def bart_summarizer(self, doc, settings):
        min_length = settings['min_length'] 
        max_length = settings['max_length']
        # see ``examples/summarization/bart/evaluate_cnn.py`` for a longer example
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        inputs = tokenizer.batch_encode_plus([doc], max_length=max_length, return_tensors='pt')
        # Generate Summary
        summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        summary = ([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
        return summary

    def t5_summarizer(self, doc, settings):
        min_length = settings['min_length'] 
        max_length = settings['max_length']

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        
        tokenized_text = tokenizer.encode(''.join(['summarize: ', doc]) , return_tensors="pt")  # Batch size 1
        # outputs = model.generate(input_ids)
        summary_ids = self.model.generate(tokenized_text,
                                            num_beams=4,
                                            no_repeat_ngram_size=2,
                                            min_length=30,
                                            max_length=100,
                                            early_stopping=True)

        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return output


