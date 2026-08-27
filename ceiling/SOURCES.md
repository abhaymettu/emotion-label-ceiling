# Where every cited number comes from

`ceiling/sota.csv` holds published numbers. Rules used to build it:

- Every value was read out of the paper's own abstract or results table, fetched
  directly. Nothing was taken from a search-engine summary.
- `primary_verified = yes` means the number was read from the paper itself.
  `no` means it was read from another paper's comparison table and the primary is
  paywalled. `abstract only` means the publisher abstract, full text closed.
- Numbers that could not be verified were left out rather than approximated.

## Rejected numbers

- **"MSAF 74.86% on CREMA-D" and "MMTM 73.12% on CREMA-D"** — false. MSAF
  ([arXiv:2012.07175](https://arxiv.org/abs/2012.07175)) never mentions CREMA-D;
  it evaluates on RAVDESS, CMU-MOSEI and NTU RGB+D. Those two values are the
  **RAVDESS** column of MAE-DFER's Table 8, adjacent to the CREMA-D column. A
  search summary asserted both as CREMA-D results. They are not in our table.
- **emotion2vec** ([arXiv:2312.15185](https://arxiv.org/abs/2312.15185)) does not
  evaluate on CREMA-D. Excluded.
- **Lei & Cao visual-only 64.68 UAR** appears in secondary tables cited to
  "Lei and Cao 2023" but we could not confirm it comes from the same TAFFC paper.
  Left out.
- Papers With Code's CREMA-D leaderboard is **gone** — the URL now 302s to
  Hugging Face. The table here was reassembled from the comparison tables inside
  HiCMAE, SVFAP, MAE-DFER, VQ-MAE-AV and Koo et al., which is the best available
  substitute and is not a leaderboard.

## The metric-name problem

S5 and S6 report **F1-micro**, not accuracy. For single-label multiclass with no
abstentions F1-micro equals accuracy, so they are comparable — but S9's own
comparison table silently relabels VAVL's and DE-III's F1-micro as "Acc". If you
see 82.60 attributed to VAVL as accuracy, that relabelling is where it came from.

## Not chased

TA-AVN (84.00, 2021), LADDER (80.30), Goncalves & Busso RAVER (77.30), AttA-Net.
All appear in secondary tables; all paywalled. Add only after primary verification.
