# Explanation
## `evidence.csv`
This contains known evidence for cannabinoid binding, intended to be used for scoring LBVS models.
At the moment, only including publicly available data.
Initial source for data is GuideToPharmacology.
Later sources include literature search (in particular morales2017 paper).
Attempts will be made to keep pchembl value above 5 (equivalent to 10uM.) 
for instance, mu opioid receptor paper uses 100uM as evidence and is therefore not included. 
Rat or mouse evidence uses the human chembl here.

## `cannabinoids.csv`
Just a list of cannabs and chemblIDs
