---
layout: post
title: Topic Model Types
description: the various types of topic models with some python
date: 2019-02-13
permalink: /:title/
---
  
**Please note**, some sections of this post are incomplete, particularly the extensions to LDA.  
I would like to come back and finish them, as well as add code examples for most of the remaining topic models (H-NMF, GuidedLda, HDP, DTM, and possibly the remaining using the PyMC3 library), but I'm not finding the time right now as work is pressing elsewhere.

--- 

<br />
There is a cornucopia of topic models currently available, all suited to different goals and allowing different extensions and customisations.  
Here I'll try to describe the ones I've encounted.

Generally, the topic model types can be loosely divided into what I consider **linear algebra first** and **probability first**, but this division is neither strict nor complete (see [linear topic modeling](http://qpleple.com/linear-topic-modeling/) and [probabilistic topic modeling](http://qpleple.com/probabilistic-topic-modeling/) and other neatly explained stuff on Pleplé's blog).

At their core, all topic modeling techniques are a form of lossy compression and conversely many lossy compression techniques can be considered a form of topic modeling. What is important however is the goal: to extract latent features which describe the documents in a corpus.

Some considerations which factor into determining a topic model are:

* speed of computation of topics
* size of fitted model
* interpretability of topics
* use of topics for information retrieval indexing

Other features, such as automatic clustering, can be useful too.

## Linear Algebra First Models

These models typically arise as matrix decomposition techniques. Given a document-term matrix, which here I'll denote $D$, one wishes to find matrices $X$, $Y$, and possibly $Z$ such that $D = XY$ or $D = XYZ$.

### Latent Semantic Analysis/Indexing

Possibly the earliest of topic models, LSA is also used "in reverse" to construct word embeddings.

The matrix factorization behind latent semantic analysis (also known as latent semantic indexing) is singular value decomposition. Any of the standard matrices (word counts, tf-idf, log-entropy) representing the corpus can be used.

Because the latent factors (singular vectors) can have negative values, these topic models are often difficult to interpret. However, SVD computes a global optimum, minimizing $L_2$ error, which (I believe) ties in with its excellent performance as an indexing tool.

LSA can be performed both with popular sklearn and with Gensim, as well as any tool performing singular value decomposition (such as numpy). Gensim however handles well too-large-for-memory datasets as well as building the tf-idf _or_ log-entropy models (sklearn at this point has not log-entropy model) and as such is my first choice tool for this method.

Example code:

```python
>>> from gensim.models import TfidfModel, LsiModel
>>> from gensim.corpora import Dictionary
>>> # a corpus which fits in memory
>>> corpus = [
...     "this is my tokenized corpus".split(),
...     "this is a document".split(),
...     "i need to keep come up with another document".split(),
...     "this is another contribution to the corpus for luck".split()
... ]
>>> id2word = Dictionary(corpus)
>>> corpus_bows = [id2word.doc2bow(doc) for doc in corpus]
>>> tfidf = TfidfModel(corpus=corpus_bows)
>>> corpus_tfidf = tfidf[corpus_bows]
>>> lsa = LsiModel(corpus_tfidf, id2word=id2word)
>>> # get topics for a document
>>> lsa[id2word.doc2bow("this is a document".split())]
[(0, -1.0068498278621802),
(1, -0.5152191886994592),
(2, 1.1421074914364997),
(3, -0.461346825201285)]
>>> # inspect topics
>>> lsa.show_topic(0)  # there are infact multiple ways to do this
[('a', -0.36841266325684113),
('document', -0.27238851341891934),
('my', -0.2705574517425453),
('tokenized', -0.2705574517425453),
('corpus', -0.2567762926193342),
('contribution', -0.24299513349612328),
('luck', -0.24299513349612328),
('for', -0.24299513349612328),
('the', -0.24299513349612328),
('to', -0.2096797485385604)]
```

### Non-Negative Matrix Factorization

This is a matrix factorization technique used with much success in topic modeling. While there are several libraries which do implement NMF and its derivative techniques in python (such as nmflib, nimfa, and sklearn to name a few), gensim does not.  

The problem statement can be expressed as finding $W, H$ such that they minimize some error function $F_D(W, H)$. Typically $F$ is the Frobenius norm of $D - WH$, but can also be the generalised Kullback-Leibler divergence between $D$ and $WH$, or even a sparse $L_1$ norm.

Depending on the implementation and optimization method, NMF can be very rapid, and can be adapted to find hierarchical topics ([hierarchical-NMF](#hierarchical-nmf) or H-NMF). The topics NMF finds are more interpretable than [LSA](#latent-semantic-analysis/indexing), as they are non-negative vectors over the words.

NMF has a rich history (through _positive matrix factorization_, for example) and has had great success in many domains. To the best of my knowledge, it is currently at or near state-of-the-art for general topic models in terms of coherence scores.

Downsides are such that, on a square matrix, its optimal performance is bested by Principal Component Analysis in terms of $L_2$ error. On the other hand, it does not typically overfit the data as the number of latent factors/components approaches the rank of the matrix (possibly because the rank of the matrix is less-or-equal-to the positive rank of the matrix).

NMF also implicitly computes clusters: the largest component in the topic vector of a document provides its cluster (much like assigning samples to the class corresponding to the highest probability in a mixture model). This allows for extensions such as [H-NMF](#hierarchical-nmf).

Furthermore, the solution is not unique, since for any invertible matrix $B$ of correct size, if $W$ and $H$ are solutions, then so too are $WB^{-1}$ and $BH$. Also, the solution found is typically from a local minimum of the error function - not a globally optimal solution.

Measuring similarity between the NMF projections can be performed in a multitude of ways, although cosine similarity works quite well on faces (for a comparison, see: [""Evaluation of Distance Measures for NMF-Based Face Image Applications" (pdf)](https://pdfs.semanticscholar.org/8af9/5e945986df9d12cd61cd061016af3f612b20.pdf))

Example code:

```python
>>> from gensim.models import TfidfModel
>>> from gensim.corpora import Dictionary
>>> from gensim.matutils import corpus2dense
>>> from sklearn.decomposition import NMF
>>> # a corpus which fits in memory
>>> corpus = [
...     "this is my tokenized corpus".split(),
...     "this is a document".split(),
...     "i need to keep come up with another document".split(),
...     "this is another contribution to the corpus for luck".split()
... ]
>>> id2word = Dictionary(corpus)
>>> corpus_bows = [id2word.doc2bow(doc) for doc in corpus]
>>> # create a num_docs x num_terms matrix for sklearn NMF
>>> document_term_matrix = corpus2dense(corpus_bows, num_terms=len(id2word)).T
>>> nmf = NMF(n_components=2)  # 2 topics
>>> topic_vectors_matrix = nmf.fit_transform(document_term_matrix)
>>> # get topics for new documents
>>> new_docs = [
...     "here is another document".split(),
...     "let us add to the corpus once more for luck".split()
... ]
>>> new_docs_matrix = corpus2dense(
>>>     [id2word.doc2bow(new_doc) for new_doc in new_docs],
>>>     num_terms=len(id2word)).T
>>> new_topics_vectors_matrix = nmf.transform(new_docs_matrix)
```

#### Hierarchical NMF

I found the paper [here](https://smallk.github.io/papers/hierNMF2.pdf).

This is a derived technique. The idea here is to iteratively select two topics, then split up the "broadest" topic. In this way, one finds a binary tree of hierarchical topics.

More specifically, one iteratively performs NMF with 2 topics. After each iteration, all the documents are assigned to a cluster corresponding to their dominant topic. Then each cluster in turn has NMF with 2 topics performed on it to find subtopics. The split - or 2 new subtopics - which maximises the modified normalised cumulative discounted gain is kept, the others dropped.

#### Matrix Update Rule

The matrix update rule is the simplest implementation of NMF. However, it requires a relatively long computation time.

#### Alternating Non-Negative Least Squares

The idea with ANLS is to fix $W$ and minimize the error function for $H$, then fix $H$ and minimize for $W$. Then repeat.

#### Generalised KL-Divergence

Kullback-Leibler divergence can be generalised to account for non-negative matrices. Known as I-Divergence, it is formulated for two matrices $X$ and $Y$ as  

$$D_I(X||Y) = \sum_{i,j}X_{i,j} \log\frac{X_{i,j}}{Y_{i,j}} - X_{i,j} + Y_{i,j}$$
It has been proven ([pdf](https://pdfs.semanticscholar.org/0062/b9ff8522498b34f467e36af218d87fcf5d9a.pdf)) that NMF with this objective function is equivalent to solving [PLSI](#probabilistic-latent-semantic-indexing/analysis) with maximimum likelihood. Since both expectation-maximisation (to solve PLSI) and the techniques for solving NMF can succumb to local minima, both techniques have been bolstered by this observation as a new technique has arisen - one simply alternates between solving PLSI with EM, and solving NMF with any typical gradient technique.
<br />

---

<br />
## Probability First Models

These models are typically generative models, at least given the documents in the corpus. That is, they learn the probabilites of the words given the topics, and the probabilities of the topics given the documents. They may also learn the probabilities of the documents themselves. Topics here are now _probability distributions over the words_.

These techniques are inherently Bayesian. They typically model the counts of distinct words in a document with a multinomial distribution, and as such the input term-document matrix is that of word counts. However, one can also model the _presence_ of words with a binary input matrix.

Module 4 in [applied text mining](https://www.coursera.org/learn/python-text-mining) on Coursera is a great resource to learn about these.

### Probabilistic Latent Semantic Indexing/Analysis

The probabilistic cousin of [LSA](#latent-semantic-analysis/indexing) from 1999, this technique is the backbone of many more advanced techniques. The basic idea was to describe a probabilistic model/graph to describe the data, then fit it with expectation maximisation, possibly combined with some other Bayesian method (such as MCMC or Variational Methods). This dependency graph has been extended into [Latent Dirichlet Allocation](#latent-dirichlet-allocation), and as such I've rarely seen PLSI outside of academia.

#### Equivalences

As mentioned above, PLSI is equivalent to [NMF](#non-negative-matrix-factorization) with the generalized Kullback-Leibler divergence error function.  
I've read that it is also equivalent to [LDA](#latent-dirichlet-allocation) with a uniform prior, but I cannot remember where I read that - if anyone knows more about this, please let me know.

### Latent Dirichlet Allocation

Much can be said about LDA. As far as probabilistic topic models go, if this is not a gold standard, it is at least a solid baseline. If you have a few hours, David Blei's lectures at videolectures.net are incredibly insightful.

You can see an implementation of it [in your browser](https://lettier.com/projects/lda-topic-modeling/). Go [here](https://medium.com/@lettier/how-does-lda-work-ill-explain-using-emoji-108abf40fa7d) for the corresponding Medium publication by Lettier.

Strictly speaking, there are _two_ LDA models. Originally, no Dirichlet prior was given to the topic distributions (the distributions over words). Later, this was added.

There have been extensions to LDA to allow it to be supervised. One method is to give the topics' Dirichlet prior a strong asymmetric parameter over the words. Another is to use GuidedLDA.

The downsides of implementing LDA include:  

* Comparing topic vectors of documents requires more than inner product, as the vectors are probability distributions. I've seen Jensen-Shannon divergence used often here.
* Fitting the model is slow relative to the [Linear Algebra First](#linear-algebra-first-models) techniques.  
* It struggles to fit short texts.

Upsides include:

* Being handled by multiple libraries (such as [Gensim](https://radimrehurek.com/gensim/models/ldamodel.html), [sklean](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html), and ldamallet), so one need not implement their own model (using, say, PyMC3 - obligatory shoutout to a great library).
* If one _wants_ to implement their own, their are many source codes to learn from, including less famous libraries (such as [GuidedLDA](#guided-lda)), the python [lda project](https://github.com/lda-project/lda), and more).
* It is very well understood and studied, and has scope for much customisation.  
* Understanding LDA helps in understanding more complex models, such as Hierarchical Dirichlet Process or the Author-Topic Model, or even the Bigram Model for short texts.

Example code:

```python
>>> from gensim.models import LdaModel
>>> from gensim.corpora import Dictionary
>>> # a corpus which fits in memory
>>> corpus = [
...     "this is my tokenized corpus".split(),
...     "this is a document".split(),
...     "i need to keep come up with another document".split(),
...     "this is another contribution to the corpus for luck".split()
... ]
>>> id2word = Dictionary(corpus)
>>> corpus_bows = [id2word.doc2bow(doc) for doc in corpus]
>>> lda = LdaModel(corpus_bows, id2word=id2word, num_topics=2)  # 2 topics
>>> # get coherence on corpus
>>> coherence = sum(t[1] for t in lda.top_topics(corpus_bows))
>>> coherence
-27.85701662947573
>>> # get topics for a document
>>> lda[id2word.doc2bow("this is a document".split())]
[(0, 0.12262865), (1, 0.8773713)]
>>> # inspect topics
>>> lda.show_topic(0)
[('to', 0.08377125),
('this', 0.07921152),
('another', 0.0738356),
('corpus', 0.07132132),
('is', 0.06781829),
('for', 0.06778592),
('the', 0.06593237),
('contribution', 0.06553106),
('luck', 0.06541307),
('document', 0.04248052)]
```

### TODO Hierarchical Dirichlet Process

This extends the [LDA](#latent-dirichlet-allocation) model into an infinite topic hierarchical model by modeling topics with a Dirichlet process.

### TODO Dynamic Topic Model

This model includes time variation, under the assumption that your corpus' topics change with time.

### Author-Topic Model

The Author-Topic Model ([pdf here](https://arxiv.org/pdf/1207.4169.pdf)) is an extension of [LDA](#latent-dirichlet-allocation) which associates with each document a tag, or author. Instead of documents, authors will have topics.  
A full Gensim tutorial on using this model can be found [here](www.github.com/rare-technologies/gensim/blob/develop/docs/notebooks/atmodel_tutorial.ipynb). The simple example here extends the LDA example above with a `author2doc` mapping.  

One of the downsides of this model is it does not allow (as far as I can tell) inferring topics of new author or documents without updating the model.
Gensim's implementation does not allow adding new authors to previously seen documents.

Example code:

```python
>>> from gensim.models import AuthorTopicModel
>>> from gensim.corpora import Dictionary
>>> # a corpus which fits in memory
>>> corpus = [
...     "this is my tokenized corpus".split(),
...     "this is a document".split(),
...     "i need to keep come up with another document".split(),
...     "this is another contribution to the corpus for luck".split()
... ]
>>> # create author2doc or doc2author mapping
>>> author2doc = {"bob": [0, 3], "alice": [1, 2]}
>>> id2word = Dictionary(corpus)
>>> corpus_bows = [id2word.doc2bow(doc) for doc in corpus]
>>> at = AuthorTopicModel(
>>>     corpus_bows, id2word=id2word, num_topics=2,
>>>     author2doc=author2doc)  # 2 topics
>>> # get coherence on corpus
>>> coherence = sum(t[1] for t in at.top_topics(corpus_bows))
>>> coherence
-27.777245117351377
>>> # get topics for a author
>>> at["bob"]
[(0, 0.22708675883415316), (1, 0.7729132411658468)]
>>> # inspect topics
>>> at.show_topic(0)
[('document', 0.09309895873992037),
('to', 0.07332999366376926),
('another', 0.07130705074493436),
('is', 0.06510520559996717),
('with', 0.0626504931704415),
('keep', 0.06262040462886577),
('up', 0.06181764432609072),
('i', 0.06170268618412031),
('need', 0.059894102830654675),
('come', 0.05898013336484846)]
>>> # update the model
>>> corpus_new = [
... "a sentence about something tokenized".split(),
... "sentence sentence word tokenize sentence".split(),
... "something entirely different".split()
... ]
>>> at.id2word.add_documents(corpus_new)
>>> corpus_new_bows = [at.id2word.doc2bow(doc) for doc in corpus_new]
>>> # add new author 'charlie'
>>> author2doc_new = {'charlie': [0, 1], 'alice': [2]}
>>> at.update(corpus_new_bows, author2doc=author2doc_new)
>>> # old authors have changed...
>>> at['bob']
[(0, 0.13044923570534722), (1, 0.8695507642946528)]
>>> # ... along with topics
>>> at.show_topic(0)
[('document', 0.10540149751326161),
('to', 0.08591053986615653),
('another', 0.08478336531889338),
('with', 0.07723496192944129),
('keep', 0.07722640838023308),
('up', 0.07699717690326738),
('i', 0.0769641864720344),
('need', 0.07643953979689866),
('come', 0.0761701981655613),
('a', 0.040321355669232034)]
>>> # new author is added
>>> at['charlie']
[(0, 0.20990596601665726), (1, 0.7900940339833428)
```

### TODO HMM-LDA

### TODO Relational Topic Model

### Some Other Probabilistic Models

Not all adaptations of the baseline models gain popularity, but many are made to meet specific needs. Here are some I've encountered:

#### Guided LDA

Documentation for the python library can be found [here](https://guidedlda.readthedocs.io/en/latest/), along with some clear code examples.

This library extends [LDA](#latent-dirichlet-allocation) to allow one to _seed_ topics with a `seed_confidence`, so the model effectively generalises your already-in-place intuitions about the topics in a corpus.

#### Tag-LDA Model

Tag-LDA extends [LDA](#latent-dirichlet-allocation) so that each document has tags as well as words. Each topic has both a tag distribution and a word distribution. This is a technique which can be used to predict keywords or folksonomies.

The research text can be found [here](http://manu35.magtech.com.cn/Jwk_ics/EN/abstract/abstract3438.shtml).

#### Labeled LDA

I haven't looked into this, but there's a paper [here](https://nlp.stanford.edu/pubs/llda-emnlp09.pdf).

#### Bigram Model

This is an extension to [LDA](#latent-dirichlet-allocation) focusing on short texts. The authors tested on massive twitter datasets, and I have no idea how well it generalises to smaller datasets. I cannot find any references to it at this time.
<br />

---

<br />
## Other Types

There are new topic models coming out regularly, including ones based on neural networks (for example, see [Neural-Relational-Topic-Models](https://github/com/zbchern/Neural-Relational-Topic-Models) - I haven't been able to access the paper yet, but I'm sure it's great!)
