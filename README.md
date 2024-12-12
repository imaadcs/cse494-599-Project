# Modified-TEPCAM
Prediction of T cell receptor-epitope binding specificity via interpretable deep learning

## Class Project: Model Variants and Results

For this project, we modified TEPCAM to use a Huber Loss function rather than the original Cross Entropy Loss and trained it on **TCR** and **EPI** splits using different variants of hyperparameters. The trained models can be found in the `model/` directory, and the results are available in the `results/metrics/` directory.

Below is a summary of the performance metrics and hyperparameters for each model from the independent test set:

---

| Split  | Model Name                          | Attention Heads | Epochs | Learning Rate | Accuracy | AUC   | AUPR  | Recall | Precision | F1    |
|--------|-------------------------------------|-----------------|--------|---------------|----------|-------|-------|--------|-----------|-------|
| TCR    | TEPCAM_tcr_1.pt            | 6               | 50    | 5e-4           | 0.573    | 0.613 | 0.701 | 0.686  | 0.559     | 0.616 |
| TCR    | TEPCAM_tcr_2.pt             | 6               | 50     | 1e-4           | 0.576    | 0.631 | 0.734 | 0.836  | 0.550     | 0.663 |
| TCR    | TEPCAM_tcr_3.pt             | 6               | 50     | 1e-3           | 0.549    | 0.581 | 0.698 | 0.722  | 0.535     | 0.615 |
| EPI    | TEPCAM_epi_1.pt            | 6               | 50    | 5e-4           | 0.523    | 0.549 | 0.705 | 0.791  | 0.515     | 0.624 |
| EPI    | TEPCAM_epi_2.pt             | 6               | 50     | 1e-4           | 0.525    | 0.576 | 0.722 | 0.855  | 0.516     | 0.643 |
| EPI    | TEPCAM_epi_3.pt             | 6               | 50     | 1e-3           | 0.550    | 0.575 | 0.685 | 0.653  | 0.542     | 0.593 |

---

# TEPCAM
Prediction of T cell receptor-epitope binding specificity via interpretable deep learning
![image](pics/model.png)

## Requirements
TEPCAM is constructed using python 3.8.16. The detail dependencies are recorded in `requirements.txt`.    

To install from the [requirements.txt](requirements.txt), using:     

```bash
pip install -r requirements.txt
```   

Using a conda virtual environment is highly recommended.

``` console
conda create -n TEPCAM
conda activate TEPCAM
conda install -r requirements.txt
```

## Model Training
For training TEPCAM, run command below:
```commandline
python ./scripts/train.py --input_file="./Data/TEP-merge.csv" --model_name="TEPCAM" --epoch=30 --learning_rate=5e-4 --GPU_num=0
```
## Model Evaluation
Model evalutation using [test.py](./scripts/test.py), run command below:
```commandline
python ./scripts/test.py 
--file_path="./Data/ImmuneCODE.csv" 
--model_path="./model/TEPCAM_test.pt" 
--output_file="./output.csv" 
--metric_file="./metric_file.csv"
```
