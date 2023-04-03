 
This is a Inforamtion Retrieval projcet and porpuse is to compare the accuracy of different models. A typical application is searching databases of abstracts of scientific documents. For instance, in data collection of the Centre for Inventions and Scientific Information (‘CISI’) which consist of text data about 1,460 documents (file ‘CISI.ALL’) and 112 associated queries (file ‘CISI.QRY’). The file ‘CISI.REL’ contains the correct list (ie. "gold standard" or "ground proof") of querydocument matching and our model can be compared against this "gold standard" to see how it has performed. A given query will return a list of document IDs relevant to the query and rank according to relevance. The performance of model will be evaluated by means of two measures, Recall and Precision.
Following models have been implemented:

- Full Vector Matrix
- Latent Semantic Structure
- Clustering
- Nonnegative Matrix Factorization (NMF)
- LGK Bidiagonalization

A library called "[utils.py](https://github.com/PouyaRepos/Information_Retrieval/utils.py)" has been created; for having an overview about it, please refer to [library description](https://github.com/PouyaRepos/Information_Retrieval/library_description.pdf).
