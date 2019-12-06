"""
Some query and document pairs have graded relevance judgements.
To compute Mean Average Precision (MAP), we need to have 
binary relevance judgements. This script converts any graded relevance
judgements to binary.
"""

fname_in = 'data/qrels.csv'
fname_out = 'data/qrels_binary.csv'
with open(fname_in, 'r') as input_file, open(fname_out, 'w') as output_file:
    for line in  input_file:
        line = line.strip()
        query_id, doc_id, rel = line.split('|')
        if rel == '2':
            rel = '1'
        output_file.write("{}|{}|{}\n".format(query_id, doc_id, rel))
    