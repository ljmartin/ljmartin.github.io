---
title: Drug-like LACAN mols on csvbase
date: 2026-08-18
---

# Drug-like LACAN mols on csvbase

Lots of posts on the to-do list require a demo dataset of drug-like molecules. ChEMBL is a common source for these, but ChEMBL molecules require some kind of data cleaning. Also, since ChEMBL molecules are 'real', then eventually such a dataset would get out of date. 

Instead, here is a public dataset of random drug-like molecules, which look very much like ChEMBL molecules, using two tools I've wanted to try out: 
- Wim Dehean's [LACAN toolkit](https://github.com/dehaenw/lacan/)
- [csvbase](https://csvbase.com)

LACAN is a software package to generate molecules and calculate a goodness score, with no deep learning required. The generation occurs by combining fragments just like BRICS and RECAP might do. The score comes from comparing the neighbourhood of a new bond to some corpus, and the ChEMBL corpus comes pre-installed. The trick to make this work seems to be using ECFP bits to define atomic environments, which to me makes it a variant of [Wave Function Collapse](https://robertheaton.com/2018/12/17/wavefunction-collapse-algorithm/). There's more details on LACAN from Wim [here](https://github.com/rdkit/UGM_2024/blob/main/Presentations/Dehaen_LACAN.pdf). 

csvbase is a online database for storing data snippets in csv format. The FAQ really says it all: https://csvbase.com/faq

I want a million molecules in this demo dataset, and LACAN would take over an hour to produce that on my machine. Instead, I'll parallelise it using Modal. This job cost a few dollars, which would be within Modal's free monthly allowance. 

```python
import modal
import os

app = modal.App("lacan-calc")
image = (
    modal.Image.debian_slim()
    .pip_install("lacan", "rdkit", "pandas")
)

@app.function(image=image, cpu=2)
def calculate(i: int) -> dict:
    from lacan import lacan, gen
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    profile = lacan.load_profile("chembl")
    mols = gen.generate_filtered_molecules(
        profile, 
        n_molecules=5000, 
        n_jobs=8, 
        max_atoms=45
    )
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    inchi_keys = [
	    Chem.MolToInchiKey(mol).split('-')[0] for mol in mols
	]
    df = pd.DataFrame({
        'smiles': smiles,
        'idn': inchi_keys
    })
    return {
	    'id': i, 
	    'csvdata': df.to_csv(index=False)
	}

@app.local_entrypoint()
def main():
    inputs = [i for i in range(200) if not os.path.isfile(f'./csvs/{i}.csv')]
    function_calls = [calculate.spawn(i) for i in inputs]
    for fc in function_calls:
        result = fc.get()
        data = result
        with open(f'./csvs/{data["id"]}.csv', 'w') as f:
            f.write(data['csvdata'])
```


I collated these outputs, and uploaded them to csvbase under the URL https://csvbase.com/ljmartin/random-lacan-1e6. So, you can pull some data with, e.g.:
```bash
curl -s https://csvbase.com/ljmartin/random-lacan-1e6 | head -n 11
```
output: 
```
csvbase_row_id,smiles,idn
1,CC(C)CCC1CCC(CCCO)CC1,ZRKJEVSWBMZXPL
2,Cc1ccc(S(N)(=O)=O)cc1CCCN(C)C,PJCHVYLLOCNIMR
3,CC(C)(C)Cc1cc(S(C)(=O)=O)c2ncn(-c3cnccn3)c2c1,RSOIQJINNUZINP
4,CS(=O)(=O)CCc1nc2ccccc2s1,UIJPVYOESDHXNV
5,CCN(CC1CCN(C)CC1)C(=O)c1ccccc1,ACGXCBGARJPUDB
6,OCC(O)Cc1cc2ccccc2cn1,LGLOTIFSBXMEQR
7,CN(C(=O)c1nc2ccccc2[nH]1)c1nccs1,QQPQBITVTSGHMN
8,C(=Cc1cccc2ccccc12)c1ccc2c(c1)CCC2,YKNVIPLCOMCWTK
9,CCN(CC)c1ccc(N2CCc3ccccc3C2)cc1,RIGMTWTURVKYEH
10,CC(C)(C)c1nc2cc(OCC(F)(F)F)ccc2s1,QUHBUWCDRHXXPZ
```

or with duckdb: 
```
select * from read_csv("https://csvbase.com/ljmartin/random-lacan-1e6");
```