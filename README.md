<h1 align="center">
    ATM-TCR
</h1>

ATM-TCR demonstrates how a multi-head self-attention based model can be utilized to learn structural information from protein sequences to make binding affinity predictions.

## Class Project: Model Variants and Results

For this project, we trained a modified variant of ATM-TCR on **EPI** and **TCR** splits using four different variants of hyperparameters. The trained models can be found in the release and their raw performance stats can be seen in the `results/` folder or in the table below.

The major modification we did here compared to the regular ATM-TCR model was use context-aware embeddings through the catELMO embedding model.

You can download the trained models directly from the release section of this repository:  
[**Download Trained Models**](https://github.com/imaad-uni/cse494-599-Project/releases/tag/v1.0.0-Modified-ATM-TCR)

Below is a summary of the performance metrics and hyperparameters for each model from the independent test set (INDEP):

---

| Split | Model Name  | Epoch | Learning Rate | Batch Size | Drop Rate | Accuracy | AUC   | F1 Macro | F1 Micro | Loss       | Precision0 | Precision1 | Recall0 | Recall1 |
|-------|-------------|-------|---------------|------------|-----------|----------|-------|----------|----------|------------|------------|------------|---------|---------|
| EPI   | EPITest1    | 100   | 0.00005       | 32         | 0.25      | 0.8335   | 0.9373| 0.831    | 0.8335   | 47896.0302 | 0.7688     | 0.9391     | 0.9537  | 0.7132  |
| EPI   | EPITest2    | 150   | 0.00005       | 32         | 0.25      | 0.8453   | 0.9289| 0.8445   | 0.8453   | 75092.6377 | 0.8014     | 0.9041     | 0.9181  | 0.7725  |
| EPI   | EPITest3    | 200   | 0.00005       | 32         | 0.25      | 0.8459   | 0.9247| 0.8456   | 0.8459   | 132178.5278| 0.8168     | 0.881      | 0.8919  | 0.7999  |
| EPI   | EPITest4    | 150   | 0.00001       | 64         | 0.3       | 0.8067   | 0.9387| 0.8011   | 0.8067   | 89171.7366 | 0.7296     | 0.9616     | 0.9745  | 0.6388  |
| EPI   | EPITest5    | 150   | 0.0001        | 64         | 0.2       | 0.8289   | 0.9088| 0.8287   | 0.8289   | 180212.2307| 0.8469     | 0.8126     | 0.8029  | 0.8548  |
| EPI   | EPITest6    | 175   | 0.00005       | 32         | 0.3       | 0.8025   | 0.9318| 0.7969   | 0.8025   | 84192.5049 | 0.727      | 0.9534     | 0.9689  | 0.6361  |
| EPI   | EPITest7    | 175   | 0.00005       | 32         | 0.2       | 0.8472   | 0.9241| 0.8472   | 0.8472   | 147628.785 | 0.8418     | 0.8527     | 0.855   | 0.8394  |
| TCR   | TCRTest1    | 150   | 0.00005       | 32         | 0.25      | 0.8561   | 0.954 | 0.8543   | 0.8561   | 34807.2945 | 0.7934     | 0.9535     | 0.9637  | 0.748   |
| TCR   | TCRTest2    | 200   | 0.00005       | 32         | 0.25      | 0.8684   | 0.9547| 0.8675   | 0.8684   | 37458.4415 | 0.8181     | 0.938      | 0.9481  | 0.7883  |
| TCR   | TCRTest3    | 175   | 0.00005       | 32         | 0.2       | 0.8785   | 0.9517| 0.8782   | 0.8785   | 42981.433  | 0.8473     | 0.916      | 0.9239  | 0.8328  |

---

## Usage on SOL
`module load mamba/latest`  
`mamba create --name modified_atm_tcr python=3.8.10`  
`cd /path/to/repo`  
`pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`  
`CUDA_VISIBLE_DEVICES=0 python main.py --infile /path/to/train/pickle --indepfile /path/to/test/pickle --save_model True --cuda True --model_name MODELNAME --mode=CHANGE_TO_TRAIN_OR_TEST_OR_BLINDTEST --epoch=100 --lr=0.00005 --batch_size=32 --drop_rate=0.2`

## Citation

**Context-Aware Amino Acid Embedding Advances Analysis of TCR-Epitope Interactions**  
Pengfei Zhang, Seojin Bang, Michael Cai, Heewook Lee  
*bioRxiv*  
DOI: [10.1101/2023.04.12.536635](https://doi.org/10.1101/2023.04.12.536635)  
Published by Cold Spring Harbor Laboratory.

## Publication
<b>ATM-TCR: TCR-Epitope Binding Affinity Prediction Using a Multi-Head Self-Attention Model</b> <br/>
[Michael Cai](https://github.com/cai-michael)<sup>1,2</sup>, [Seojin Bang](https://github.com/SeojinBang)<sup>2</sup>, [Pengfei Zhang](https://github.com/pzhang84)<sup>1,2</sup>, [Heewook Lee](https://scai.engineering.asu.edu/faculty/computer-science-and-engineering/heewook-lee/)<sup>1,2</sup><br/>
<sup>1 </sup>School of Computing and Augmented Intelligence, Arizona State University, <sup>2 </sup>Biodesign Institute, Arizona State University <br/>
Published in: [**Frontiers in Immunology, 2022.**](https://www.frontiersin.org/articles/10.3389/fimmu.2022.893247/full)

## Model Structure

The model takes a pair epitope and TCR sequences as input and returns the binding affinity between the two. The sequences are processing through an embedding layer before reaching the mutli-head self-attention layer. The outputs of these layers are then concatenated and fed through a linear decoder layer to receive the final binding affinity score.

<img src="data/fig/model.png" alt="drawing" width="500"/>

## Requirements
Written using Python 3.8.10

The pip package dependencies are detailed in ```requirements.txt```

To install directly from the requirements list
```
pip install -r requirements.txt
```
It is recommended you utilize a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

## Input File Formatting Format

The input file should be a CSV with the following format:
```
Epitope,TCR,Binding Affinity
```

Where epitope and TCR are the linear protein sequences and binding affinity is either 0 or 1.

```
# Example
GLCTLVAML,CASSEGQVSPGELF,1
GLCTLVAML,CSATGTSGRVETQYF,0
```

If your data is unlabeled and you are only interested in the predictions, simply put either all 0's or all 1's as the label. The performance statistics can be ignored in this case and the predicted binding affinity scores can be collected from the output file.

## Training
To train the model on our dataset using the default settings and on the first GPU
```
CUDA_VISIBLE_DEVICES=0 python main.py --infile data/combined_dataset.csv
```

To change the device to be utilized for training change the ```CUDA_VISIBLE_DEVICES``` to the device number as indicated by ```nvidia-smi```.

The default model name utilized by the program is  ```original.ckpt```. To change the outputted/read model name utilize the following optional argument:
```
--model_name my_custom_model_name
```

After training has finished the model will appear under the ```models``` folder under ```model_name.ckpt``` and two csv files will appear in the ```result``` folder. These files will be called ```perf_model_name.csv``` and ```pred_model_name.csv``` respectively.

```perf_model_name.csv``` contains the a description of performance metrics throughout training. Each line of the csv is the performance of the training model on the validation set in that particular epoch. The last line of the file contains the final performance statistics.
```
# Example
Loss        Accuracy Precision1 Precision0 Recall1 Recall0 F1Macro F1Micro AUC
37814.6235	0.6101	 0.6241	    0.5988	   0.5542  0.666   0.6089  0.6101  0.6749
```

```pred_model_name.csv``` contains the predictions of the model on the validation set of data. Each line is a pair from the validation set along with the label and prediction made by the model. The calculated score from the model is also included.
```
# Example
Epitope     TCR	        Actual Prediction Binding Affinity
GLCTLVAML	CASCWNYEQYF	1	   1	      0.9996516704559326
```
## Testing
To make a prediction using a pre-trained model
```
python main.py --infile data/combined_dataset.csv --indepfile data/covid19_data.txt --model_name my_custom_model_name --mode test
```

The predictions will be saved into the ```result``` folder under the name ```pred_model_name_indep_test_data.csv```. These will be displayed similarly to the validation set predictions made during training.

## Optional Arguments

For more information on optional hyperparameter and training arguments
```
python main.py --help
```

## Data

See the README inside of the data folder for additional information.

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
