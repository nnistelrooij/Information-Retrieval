"""
A script that normalizes and tokenizes queries from a test collection.
"""

import re, string
from nltk.corpus import stopwords

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
re_punct = re.compile('[%s]' % re.escape(string.punctuation))
re_split = re.compile(r'[\-\/]')

fname_in = 'data/queries.orig.txt'
fname_out = 'data/test_queries.csv'
with open(fname_in, 'r') as input_file, open(fname_out, 'w') as output_file:
    for line in input_file:
        query_id, full_query = line.split('|')
        # normalize and tokenize query
        full_query = full_query.strip().lower()        
        partial_query = full_query.split()
        # filter out stop words
        partial_query = [re_split.split(term) for term in partial_query if term not in stop_words]
        # flatten list and remove punctuation
        partial_query = [re_punct.sub('', term) for split_terms in partial_query for term in split_terms]
        query_len = len(partial_query)
        # write to file
        for term in partial_query:
            output_file.write("{}|{}|{}\n".format(query_id, term, query_len))

if __name__ == "__main__":
    pass
