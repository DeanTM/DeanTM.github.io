---
layout: post
title: Memory Profiling in Python
description: a look at the memory_profiling tool with LDA
date: 2019-02-13
permalink: /:title/
---

### Installation

Occassionally we find ourselves desperately trying to minimize memory consumption. Python now has a tool for that: `memory_profiler`. It performs line-based memory usage tracking. It's _wonderful_!

Here I'll show how to use it in a Jupyter environment with a machine learning pipeline. You can also look at great tutorials [here](https://ipython-books.github.io/44-profiling-the-memory-usage-of-your-code-with-memory_profiler/) and in [sklearn's documentation](https://scikit-learn.org/stable/developers/performance.html).  

The original project, including API info in the readme, can be found [here](https://github.com/pythonprofilers/memory_profiler).

You can install it with
```
pip install -U memory_profiler
```
or 
```
conda install memory_profiler
```
whichever you prefer.

### A simple example

Next we'll need to load the extension


```python
%load_ext memory_profiler
```

Following the first tutorial, as a starting point we'll create a simple function which uses excessive memory.  
We'll write it to a file `big_memory.py` and run it, saving the profile to `mprof0`.


```python
%%writefile big_memory.py
def big_memory():
    a = [1] * 1000000
    b = [2] * 9000000  # load memory
    del b  # clear memory
    return a
```

    Overwriting big_memory.py


^ It says _Overwriting_ cos I've done this before


```python
from big_memory import big_memory
%mprun -T mprof0 -f big_memory big_memory()
```

    
    
    *** Profile printout saved to text file mprof0. 



```python
with open('mprof0', 'r') as f:
    print(f.read())
```

    Filename: /home/dean/projects/memory-profiler/big_memory.py
    
    Line #    Mem usage    Increment   Line Contents
    ================================================
         1     44.0 MiB     44.0 MiB   def big_memory():
         2     51.5 MiB      7.5 MiB       a = [1] * 1000000
         3    120.1 MiB     68.6 MiB       b = [2] * 9000000  # load memory
         4     51.7 MiB      0.0 MiB       del b  # clear memory
         5     51.7 MiB      0.0 MiB       return a


Surprisingly for me, the _decrement_ in memory at line 4 doesn't present negatively in the increment column.  

Okay, let's extend this to take a variable. Since I'm saving on redundant files, we'll have to reload the module so python knows the function has changed.


```python
%%writefile big_memory.py
def big_memory(c_count):
    a = [1] * 1000000
    b = [2] * 9000000  # load memory
    del b  # clear memory
    c = [3] * c_count  # load more
    return a
```

    Overwriting big_memory.py



```python
import sys
import importlib as imp
imp.reload(sys.modules['big_memory'])
from big_memory import big_memory

%mprun -T mprof0 -f big_memory big_memory(4500000)
```

    
    
    *** Profile printout saved to text file mprof0. 



```python
with open('mprof0', 'r') as f:
    print(f.read())
```

    Filename: /home/dean/projects/memory-profiler/big_memory.py
    
    Line #    Mem usage    Increment   Line Contents
    ================================================
         1     53.0 MiB     53.0 MiB   def big_memory(c_count):
         2     53.0 MiB      0.0 MiB       a = [1] * 1000000
         3    121.6 MiB     68.6 MiB       b = [2] * 9000000  # load memory
         4     53.0 MiB      0.0 MiB       del b  # clear memory
         5     87.3 MiB     34.3 MiB       c = [3] * c_count  # load more
         6     87.3 MiB      0.0 MiB       return a


We can also test the memory of a single line or cell with the IPython magic `%memit`


```python
%memit [0] * 4500000
```

    peak memory: 105.10 MiB, increment: 34.30 MiB



```python
%%memit import numpy as np
A = np.random.rand(1000, 1000)
A ** 2
```

    peak memory: 70.82 MiB, increment: 0.02 MiB


### Usage within a script

Following the documentation, let's create a script which does several things, but also memory profiles a function in its main loop


```python
%%writefile mem_script.py

@profile
def memory_func():
    a = [1] * (10**6)
    b = [2] * (2* 10**7)
    del b
    return a

def redundant_length(a):
    i = 0
    for x in a:
        i += 1
    return i

if __name__ == '__main__':
    a = memory_func()
    print(redundant_length(a))
    print('Done script!\n')
```

    Overwriting mem_script.py



```python
!python -m memory_profiler mem_script.py
```

    1000000
    Done script!
    
    Filename: mem_script.py
    
    Line #    Mem usage    Increment   Line Contents
    ================================================
         2   32.699 MiB   32.699 MiB   @profile
         3                             def memory_func():
         4   40.285 MiB    7.586 MiB       a = [1] * (10**6)
         5  192.910 MiB  152.625 MiB       b = [2] * (2* 10**7)
         6   40.453 MiB    0.000 MiB       del b
         7   40.453 MiB    0.000 MiB       return a
    
    


Provided we're running a script (and not a cell in Jupyter) we can also do it without calling `memory_profiler` in the command line, by importing the decorator instead. This exposes a precision parameter we can use.


```python
%%writefile mem_script.py
from memory_profiler import profile

# `@profile` works too, 
# the precision is not needed
@profile(precision=4)
def memory_func():
    a = [1] * (10**6)
    b = [2] * (2* 10**7)
    del b
    return a

def redundant_length(a):
    i = 0
    for x in a:
        i += 1
    return i

if __name__ == '__main__':
    a = memory_func()
    print(redundant_length(a))
    print('Done script!\n')
```

    Overwriting mem_script.py



```python
!python mem_script.py
```

    Filename: mem_script.py
    
    Line #    Mem usage    Increment   Line Contents
    ================================================
         5  32.8086 MiB  32.8086 MiB   @profile(precision=4)
         6                             def memory_func():
         7  40.2031 MiB   7.3945 MiB       a = [1] * (10**6)
         8 192.8281 MiB 152.6250 MiB       b = [2] * (2* 10**7)
         9  40.4414 MiB   0.0000 MiB       del b
        10  40.4414 MiB   0.0000 MiB       return a
    
    
    1000000
    Done script!
    


There's also some stuff using `mprof run <executable>` and `mprof plot` to track memory usage agaisnt time, but it didn't seem to display in this notebook.

Instead of pursuing that, I'm moving onto something more fun..

### Comparing the memory of Latent Dirichlet Allocation implementations

Here I will fit LDA models to NLTK's Brown corpus, 15 topics, one for each category.
I'll compare scikit-learn's implementation with Gensim's. Scikit-learn is usually regarded as faster - let's see which uses less memory!

It should be said I'm rooting for Gensim here. I've enjoyed using Gensim.

This test itself is a bit tongue-in-cheek, as Gensim has tools for streaming corpora too large for memory, and scikit-learn's LDA can also do online batch updates.


```python
%%writefile lda_memory.py
from nltk.corpus import brown
docs = [brown.words(fid) for fid in brown.fileids()]
docs_sklearn = [' '.join(doc) for doc in docs]

def gensim_test():
    from gensim.models import LdaModel
    from gensim.corpora import Dictionary
    id2word = Dictionary(docs)
    bows = [id2word.doc2bow(doc) for doc in docs]
    lda = LdaModel(corpus=bows, id2word=id2word, num_topics=15)
    return lda

def sklearn_test():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    docterm_matrix = CountVectorizer().fit_transform(docs_sklearn)
    lda = LatentDirichletAllocation(n_components=15)
    lda.fit(docterm_matrix)
    return lda
```

    Writing lda_memory.py



```python
from lda_memory import gensim_test
%mprun -T mprof0 -f gensim_test gensim_test()

with open('mprof0', 'r') as f:
    print(f.read())
```

    
    
    *** Profile printout saved to text file mprof0. 
    Filename: /home/dean/projects/memory-profiler/lda_memory.py
    
    Line #    Mem usage    Increment   Line Contents
    ================================================
         5    196.7 MiB    196.7 MiB   def gensim_test():
         6    196.7 MiB      0.0 MiB       from gensim.models import LdaModel
         7    196.7 MiB      0.0 MiB       from gensim.corpora import Dictionary
         8    191.7 MiB      0.0 MiB       id2word = Dictionary(docs)
         9    213.2 MiB      0.3 MiB       bows = [id2word.doc2bow(doc) for doc in docs]
        10    238.2 MiB     25.0 MiB       lda = LdaModel(corpus=bows, id2word=id2word, num_topics=15)
        11    238.2 MiB      0.0 MiB       return lda



```python
from lda_memory import sklearn_test
%mprun -T mprof0 -f sklearn_test sklearn_test()

with open('mprof0', 'r') as f:
    print(f.read())
```

    
    
    *** Profile printout saved to text file mprof0. 
    Filename: /home/dean/projects/memory-profiler/lda_memory.py
    
    Line #    Mem usage    Increment   Line Contents
    ================================================
        13    214.4 MiB    214.4 MiB   def sklearn_test():
        14    214.4 MiB      0.0 MiB       from sklearn.feature_extraction.text import CountVectorizer
        15    214.4 MiB      0.0 MiB       from sklearn.decomposition import LatentDirichletAllocation
        16    222.1 MiB      7.6 MiB       docterm_matrix = CountVectorizer().fit_transform(docs_sklearn)
        17    222.1 MiB      0.0 MiB       lda = LatentDirichletAllocation(n_components=15)
        18    249.8 MiB     27.8 MiB       lda.fit(docterm_matrix)
        19    249.8 MiB      0.0 MiB       return lda


### Results

The input to sklearn's method, the `docterm_matrix`, is definitely less efficiently represented than the `bows` from gensim. But the differences in the memory of the lda models _themselves_ seems too small to comment.
