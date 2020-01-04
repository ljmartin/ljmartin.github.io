---
layout: default
title: A bayesian naive bayes classifier [in progress]
---

# A bayesian naive bayes classifier

Naive bayes classifiers are commonly-used tools in subject classification of text as well as in target identification in chemoinformatics (aka ligand-based virtual screening). See [Jake VanderPlas' python data science handbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html) for a practical example if the topic is unfamiliar. One issue I've come across is the availability of evidence - some targets have many thousands of known positives and known negatives, while some targets have on the order of 10<sup>0</sup> - 10<sup>2</sup>. A target with few known positives will lead to highly confident predictions simply because not all of the feature space has been sampled.
