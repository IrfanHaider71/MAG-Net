
import numpy as np
import torch
import torch.nn as nn

import baseline_harness as H


PAD_ID = 0
UNK_ID = 1


def build_vocab(smiles_list):
   Character-level vocab built from TRAIN SMILES ONLY.
    Index 0 = PAD, index 1 = UNK, indices 2.. = observed training chars.
    chars = sorted({c for s in smiles_list for c in s})
    vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
    for c in chars:
        vocab[c] = len(vocab)
    return vocab


def tokenize(s, vocab, max_len):
    ids = [vocab.get(c, UNK_ID) for c in s]
    ids = ids[:max_len]
    ids = ids + [PAD_ID] * (max_len - len(ids))
    return ids

class SmilesCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, n_out=7):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)

        def conv_block(c_in, c_out, k):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
            )

        self.conv1 = conv_block(embed_dim, 64, 7)
        self.conv2 = conv_block(64, 128, 5)
        self.conv3 = conv_block(128, 128, 3)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_out),
        )

    def forward(self, x):
        # x: [B, L] long token ids
        h = self.embed(x)                 # [B, L, E]
        h = h.transpose(1, 2)             # [B, E, L]
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)                 # [B, 128, L]
        h = self.pool(h).squeeze(-1)      # [B, 128]
        out = self.head(h)                # [B, 7]
        return out

def make_collate_fn():
    def collate_fn(items):
        xs = torch.stack([it[0] for it in items], dim=0)
        ys = torch.stack([it[1] for it in items], dim=0)
        ms = torch.stack([it[2] for it in items], dim=0)
        return xs, ys, ms
    return collate_fn


def main():
    H.set_seed(H.SEED)

    df = H.load_merged()
    df_train, df_test = H.canonical_split(df)

    train_smiles = df_train["SMILES"].tolist()
    test_smiles = df_test["SMILES"].tolist()

    vocab = build_vocab(train_smiles)
    vocab_size = len(vocab)
    max_len = max(len(s) for s in train_smiles)
    print(f"Vocab size (incl. PAD/UNK): {vocab_size} | Train max SMILES length: {max_len}")

    Ytr_raw, Mtr = H.targets_tensor(df_train)
    Yte_raw, Mte = H.targets_tensor(df_test)
    y_mu, y_sd = H.mask_aware_target_stats(Ytr_raw, Mtr)

    train_items = []
    for i, s in enumerate(train_smiles):
        ids = torch.tensor(tokenize(s, vocab, max_len), dtype=torch.long)
        train_items.append((ids, Ytr_raw[i], Mtr[i]))

    tr_items, val_items = H.make_val_split(train_items, val_frac=0.15, seed=H.SEED)

    test_items = []
    for i, s in enumerate(test_smiles):
        ids = torch.tensor(tokenize(s, vocab, max_len), dtype=torch.long)
        test_items.append((ids, Yte_raw[i], Mte[i]))

    collate_fn = make_collate_fn()

    model = SmilesCNN(vocab_size=vocab_size, embed_dim=32, n_out=len(H.PROP_COLS))

    model, best_val_rmse = H.train_generic(
        model, tr_items, val_items, collate_fn, y_mu, y_sd,
        epochs=500, batch=16, lr=2e-3, weight_decay=1e-4, patience=60,
        seed=H.SEED, verbose=True,
    )
    print(f"Best validation RMSE (standardized-space target scale): {best_val_rmse:.5f}")

    model.eval()
    with torch.no_grad():
        Xte, _, _ = collate_fn(test_items)
        yhat_std = model(Xte)
        yhat_test = (yhat_std * y_sd + y_mu).numpy()

    if not np.isfinite(yhat_test).all():
        print("WARNING: non-finite predictions detected in test set!")

    metrics = H.evaluate_and_save(
        "SMILES-CNN", df_test, Yte_raw.numpy(), yhat_test, Mte.numpy(),
        out_prefix="baseline_smiles_cnn",
    )

    print("\nFinal per-property test metrics:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
