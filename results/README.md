# BRTP: experimental data accompanying the manuscript

This package contains numerical results and protocol information for the BRTP manuscript. It contains no dataset images, model weights, third-party publications, or private storage locations. It records aggregate and per-seed results; it does not contain the complete archive of per-image test predictions.

Run **python verify.py** from any directory. The script needs only Python 3 and the files alongside it. It checks the file digests and individual panel accuracies, rebuilds the main tables for the five locally run methods, and checks the ten-seed summaries. All reported percentages are on a 0–100 scale unless specified.

| File | Contents |
| --- | --- |
| protocol.txt | Source/test split counts, support sampling, training settings, and how to compute each transfer score. |
| tables.csv | Numeric values in Tables 1–11 and A.1–A.4. Each entry gives its displayed value, the value checked against source evidence, and its evidence type. |
| test-panels.csv | Counts rebuilt from saved test predictions. Each row is one model/task/source/target/seed/test-vocabulary panel. n: test images; k: competing labels; correct: correct predictions; oa: accuracy (%). For joint B2N, joint_base and joint_new are subgroup accuracies (%) under the common vocabulary; n_joint_base and n_joint_new are their denominators. Empty fields are inapplicable. |
| fig4-mechanisms.csv | Figure 4: outcomes of matched model configurations by seed, before rounding. |
| fig5-sensitivity.csv | Figure 5: results by shot budget and radius, at manuscript reporting precision. |
| fig6-ten-seeds.csv / fig6-summary.csv | Figure 6: ten individual scores per method and metric, plus the plotted estimate and sample SD. |
| class-geometry.csv | Table 6: per-class, per-seed diagnostics for F00, including coverage (g), pre-bound residual length, threshold indicators, and final angle (degrees). |
| external-targets.csv | Appendix A.3: individual scores on EuroSAT, UC Merced, and AID. |
| calibration.csv | Appendix A.4: expected calibration error (ECE, fraction from 0–1) and negative log likelihood (NLL), by method/protocol/seed. Multiply ECE by 100 to compare with the table. |

**Reading the comparisons.** Full denotes BRTP. Matched CoOp uses the same training settings as the local models and is distinct from the published CoOp row. The external benchmark rows are numerical comparisons from the [FrogDogNet paper](https://arxiv.org/abs/2504.16433), not independent reruns; their per-seed results are not supplied here. For locally trained models, overall B2N H is computed from the averaged base/new accuracies; it generally differs from both the mean of dataset H and the mean of per-seed H. Target averages for CD and DG exclude the training domain. A v2 suffix identifies a shared-16-class dataset view used for DG.

**Obtaining the public datasets.** Use the providers' own distribution conditions and instructions. The code project's protocol archive contains the split/support definitions for the main experiments.

| Dataset | Provider or distribution record |
| --- | --- |
| PatternNet | [Dataset author's page](https://sites.google.com/view/zhouwx/dataset) |
| RSICD | [Dataset repository](https://github.com/201528014227051/RSICD_optimal) |
| NWPU-RESISC45 | [Distribution record and source information](https://www.tensorflow.org/datasets/catalog/resisc45) |
| MLRSNet | [Dataset repository](https://github.com/cugbrs/MLRSNet) |
| EuroSAT | [Zenodo record](https://zenodo.org/records/7711810) |
| UC Merced Land Use | [UC Merced dataset page](https://vision.ucmerced.edu/datasets/) |
| AID | [Wuhan University dataset page](https://captain-whu.github.io/AID/) |
