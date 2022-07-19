# LDA Numpy Implementation with Baby Example

This is a tiny LDA implementation from Boyuan Chen and Yi Wei. The goal is to help first-time topic-modeling learners to acquire the knowledge faster. The corpus looks like:

<pre>
X = np.array([
    [0, 0, 1, 2, 2],
    [0, 0, 1, 1, 1],
    [0, 1, 2, 2, 2],
    [2, 2, 1, 1, 4],
    [4, 4, 4, 4, 4],
    [3, 3, 4, 4, 4],
    [3, 4, 4, 4, 4],
    [3, 3, 3, 4, 1],
    [4, 4, 3, 3, 2],
]) </pre>

Apparently, the first four documents should be in one topic, and the other five should be in the other topic. The goal is to output a document-topic probability distribution like below:
<pre>
[[0.14285714 0.85714286]
 [0.14285714 0.85714286]
 [0.28571429 0.71428571]
 [0.42857143 0.57142857]
 [0.85714286 0.14285714]
 [0.71428571 0.28571429]
 [0.85714286 0.14285714]
 [0.85714286 0.14285714]
 [0.71428571 0.28571429]]</pre>
We mainly took reference from from Agustinus Kristiadi's Blog: https://agustinus.kristia.de/techblog/2017/09/07/lda-gibbs/. All the parameters correspond to Finding scientific topics 2004. 
