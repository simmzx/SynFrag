from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import pandas as pd
from sklearn.cluster import KMeans


def morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)


def cluster_smiles(smiles, n_clusters=50):
    fps = []
    for s in smiles:
        fp = morgan_fingerprint(s)
        fps.append(fp)

    dists = []
    nfps = len(fps)
    for i in range(nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        dists.append(sims)

    mol_dist = pd.DataFrame(dists)
    k_means = KMeans(n_clusters=n_clusters)
    k1 = k_means.fit_predict(mol_dist)
    output = []
    having_clustered_index = []
    assert(len(k1) == len(smiles))
    for k, s in zip(k1, smiles):
        if k in having_clustered_index:
            continue
        else:
            output.append(s)
            having_clustered_index.append(k)
    return output

