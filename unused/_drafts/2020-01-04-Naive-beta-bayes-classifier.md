---
layout: default
title: A bayesian naive bayes classifier [in progress]
---

# A bayesian naive bayes classifier

Here I'll walk through the process of creating a Naive bayes classifier that takes into account the amount of available data when calculating probabilities. Doing this adds implicit confidence values to the predicted probability values, which helps when ranking predictions.


Naive bayes classifiers are commonly-used tools in subject classification of text as well as in target identification in chemoinformatics (aka ligand-based virtual screening). See [Jake VanderPlas' python data science handbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html) for a practical example if these are unfamiliar. One issue with naive bayes classifiers is the 'naive' independence assumption - i.e. assuming that each feature occurs independently, when in reality features can be highly correlated - leading to very high/very low probability predictions (the [sklearn docs](https://scikit-learn.org/stable/modules/calibration.html) demonstrate this nicely). It's hard to see how to fix that without greatly increasing the cost of fitting a naive bayes classifier.

Another issue I've come across is the availability of evidence - some protein targets have many thousands of known positives and known negatives, while some have only 10<sup>0</sup> - 10<sup>2</sup>. A target with few known positives will lead to highly confident predictions simply because not all of the feature space has been sampled adequately. As a result, when ranking the predictions from a naive bayes classifier by probability, it's hard to tell how _confident_ one can be about each of the top-ranked predictions. 

To show why this is a problem, let's use an extreme example. 
