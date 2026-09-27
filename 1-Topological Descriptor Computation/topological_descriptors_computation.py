 
import os
from collections import Counter, defaultdict

import pandas as pd
from rdkit import Chem

def graph_from_smiles(smiles: str):
    """Parse a SMILES string into a heavy-atom adjacency structure.

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    adj = defaultdict(set)
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (
            mol.GetAtomWithIdx(u).GetAtomicNum() == 1
            or mol.GetAtomWithIdx(v).GetAtomicNum() == 1
        ):
            continue
        adj[u].add(v)
        adj[v].add(u)
    deg = {v: len(neis) for v, neis in adj.items()}
    return adj, deg


def m_polynomial_counts(adj, deg):
    """Build the M-polynomial edge partition {m_ij}: for each unordered
    degree pair (i, j) with i <= j, count the edges joining a degree-i
    vertex to a degree-j vertex."""
    counts = Counter()
    for u in adj:
        for v in adj[u]:
            if u < v:
                i, j = deg[u], deg[v]
                if i > j:
                    i, j = j, i
                counts[(i, j)] += 1
    return counts

def Dx(M):
    return sum(i * c for (i, j), c in M.items())


def Dy(M):
    return sum(j * c for (i, j), c in M.items())


def DxDy(M):
    return sum(i * j * c for (i, j), c in M.items())


def Dx2(M):
    return sum((i**2) * c for (i, j), c in M.items())


def Dy2(M):
    return sum((j**2) * c for (i, j), c in M.items())


def M1_from_M(M):
    """First Zagreb index: sum_{uv in E} (d(u) + d(v))."""
    return Dx(M) + Dy(M)


def M2_from_M(M):
    """Second Zagreb index: sum_{uv in E} d(u) d(v)."""
    return DxDy(M)


def mM2_from_M(M):
    """Modified second Zagreb index: sum_{uv in E} 1 / (d(u) d(v))."""
    return sum(c * (1.0 / (i * j)) for (i, j), c in M.items() if i * j > 0)


def ReM3_from_M(M):
    """Redefined third Zagreb index: sum_{uv in E} d(u) d(v) (d(u)+d(v))."""
    return sum(c * (i * j * (i + j)) for (i, j), c in M.items())


def H_from_M(M):
    """Harmonic index: sum_{uv in E} 2 / (d(u) + d(v))."""
    return 2.0 * sum(c / (i + j) for (i, j), c in M.items())


def F_from_M(M):
    """Forgotten index: sum_{uv in E} (d(u)^2 + d(v)^2)."""
    return Dx2(M) + Dy2(M)


def AZ_from_M(M):
    """Augmented Zagreb index: sum_{uv in E} [d(u)d(v) / (d(u)+d(v)-2)]^3."""
    total = 0.0
    for (i, j), c in M.items():
        denom = i + j - 2
        if denom > 0:
            total += c * ((i * j) / denom) ** 3
    return total


def ISI_from_M(M):
    """Inverse sum indeg index: sum_{uv in E} d(u) d(v) / (d(u)+d(v))."""
    return sum(c * ((i * j) / (i + j)) for (i, j), c in M.items())


def SDD_from_M(M):
    """Symmetric division degree index: sum_{uv in E} (d(u)/d(v) + d(v)/d(u))."""
    return sum(c * ((i / j) + (j / i)) for (i, j), c in M.items() if i * j > 0)


DESCRIPTOR_FUNCS = [
    ("M1 (First Zagreb)", M1_from_M),
    ("M2 (Second Zagreb)", M2_from_M),
    ("mM2 (Modified Second Zagreb)", mM2_from_M),
    ("ReM3 (Redefined M3)", ReM3_from_M),
    ("Harmonic Index", H_from_M),
    ("Forgotten Index", F_from_M),
    ("AZ (Augmented Zagreb)", AZ_from_M),
    ("ISI (Inverse Sum Indeg)", ISI_from_M),
    ("SDD (Symmetric Division Degree)", SDD_from_M),
]


def main():
 
    smiles_df = pd.read_csv(IN_SMILES)
    smiles_df.columns = smiles_df.columns.astype(str).str.strip()

    rows = []
    for _, r in smiles_df.iterrows():
        drug, smiles = str(r["Drug"]).strip(), str(r["SMILES"]).strip()
        try:
            adj, deg = graph_from_smiles(smiles)
            M = m_polynomial_counts(adj, deg)
            row = {"Drug": drug}
            for name, fn in DESCRIPTOR_FUNCS:
                row[name] = fn(M)
            rows.append(row)
        except Exception as e:
            print(f"[WARN] Could not process {drug}: {e}")

    df = pd.DataFrame(rows)
    df.insert(0, "Index", range(1, len(df) + 1))
    df.to_csv(OUT_CSV, index=False)

    print(f"\nTopological calculated and saved to: {OUT_CSV}")
    print(f"Processed {len(df)} / {len(smiles_df)} compounds.")


if __name__ == "__main__":
    main()
