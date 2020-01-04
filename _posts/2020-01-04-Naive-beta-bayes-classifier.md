---
layout: default
title: A bayesian naive bayes classifier [in progress]
---

# A bayesian naive bayes classifier

Naive bayes classifiers are commonly-used tools in subject classification of text as well as in target identification in chemoinformatics (aka ligand-based virtual screening). See [Jake VanderPlas' python data science handbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html) for a practical example if the topic is unfamiliar. One issue with naive bayes classifiers is the 'naive' independence assumption - i.e. assuming that each feature occurs independently, when in reality features can be highly correlated - leading to very high probability predictions. It's hard to see how to fix that without increasing the cost of fitting an NB classifier greatly.

Another issue I've come across is the availability of evidence - some protein targets have many thousands of known positives and known negatives, while some have only 10<sup>0</sup> - 10<sup>2</sup>. A target with few known positives will lead to highly confident predictions simply because not all of the feature space has been sampled. As a result, when ranking the predictions from a naive bayes classifier by probability, it's hard to tell how confident one can be about each of the top-ranked predictions. 

To show why this is a problem, let's use an example. 
