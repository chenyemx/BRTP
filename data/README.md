# BRTP evaluation counts

`aggregate_counts.csv` contains 16 BRTP records covering the task, source, target and class subset. Each row gives the number of correct predictions and total evaluations summed over three training seeds. Within a row, all three seeds used the same test set, so this ratio equals the arithmetic mean of the three accuracies. BRTP used CLIP ViT-B/16 and 16 labeled examples per training class.

Run `python data/summarize.py` from the repository root to calculate the 25 BRTP percentages to two decimals. The script uses the Python standard library.

- **B2N** evaluates base and new classes separately. For the overall harmonic mean, average the four domain accuracies for each subset first, then take the harmonic mean of those two averages.
- **CD** evaluates cross-dataset transfer from PatternNet. The target mean averages RSICD, NWPU45 and MLRSNet equally.
- **DG** evaluates domain generalization on the aligned class vocabulary. It uses the same source and target averaging rule as CD. NWPU45 denotes NWPU-RESISC45.

[Dataset sources](datasets.csv) lists provider or distribution pages and the original publications. Obtain imagery under each provider's terms.
