# TimeCite Dataset for RecSys 2024

## Temporal Citation Prediction

Welcome to the TimeCite project! This repository provides the TimeCite dataset and a script for evaluating temporal citation prediction models. The primary focus is on calculating per-bin accuracy scores based on similarity scores provided in TSV files.

### Dataset

We provide the raw dataset in the file `TimeCite.tsv`.

### File Structure

- `TimeCite.tsv`: The raw dataset containing citation information.
- `preds/`: Directory where you will place your similarity prediction files. We provide Specter2 and Prone-S (50) scores from the paper.
- `timecite.py`: The script for processing the dataset and evaluating the models.
- `results.tsv`: Output file containing the per-bin accuracy scores for each method.

### Usage

1. **Prepare Similarity Files:**
   - Use the IDs from `TimeCite.tsv` to calculate similarity scores ranging from `-1` to `1`.
   - Save the results as a TSV file with columns `id0`, `id1`, and `sim` in the `preds/` directory.
   - For missing values, set the similarity to `-1`.

2. **Run the Evaluation Script:**
   - Execute the script using the following command:
     ```sh
     python timecite.py
     ```
   - The script will produce a file `results.tsv` containing the per-bin accuracy scores for each method.

### Example Similarity File Format

An example of a similarity file (e.g., `preds/your_model_preds.tsv`) should look like this:

```
id0    id1    sim
248518397    1041744    0.7416329
248518397    23848439    0.092053235
248518397    4235810    0.116437435
248518397    82079949    -1
3374228    140728989    0.64704543
```

### Evaluation Script Details

The evaluation script performs the following steps:

1. Reads the `TimeCite.tsv` file to get the dataset.
2. Reads each TSV file in the `preds/` directory containing similarity scores.
3. Merges the similarity scores with the `TimeCite` dataset.
4. For each bin defined in the `TimeCite` dataset, it:
   - Finds the optimal threshold for citation prediction using a logistic regression classifier on the validation set.
   - Predicts citations in the test set based on the calculated threshold.
   - Handles missing similarity values (`sim == -1`) by setting 30% of the predicted citations to `1`.
   - Calculates accuracy for each bin.
5. Saves the results to `results.tsv` with bin-wise accuracy scores.

### Conclusion

This project provides a dataset and code for evaluating temporal citation prediction using the TimeCite dataset.  If you have any questions or need further assistance, please feel free to reach out.
