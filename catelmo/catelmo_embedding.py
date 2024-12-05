import pandas as pd
from pathlib import Path
import torch
from allennlp.commands.elmo import ElmoEmbedder
from tqdm import tqdm  # Import tqdm

# Define the paths to the model files
model_dir = Path('./catelmo/model')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'

# Initialize the ElmoEmbedder
embedder = ElmoEmbedder(options, weights, cuda_device=0)  # cuda_device=-1 for CPU / 0 for GPU

def catELMo_embedding(x):
    """Generate catELMo embedding for a given sequence."""
    return torch.tensor(embedder.embed_sentence(list(x))).sum(dim=0).mean(dim=0).tolist()

# Read the CSV file without headers
# dat = pd.read_csv('./catelmo/data/train_subset.csv', header=None)
dat = pd.read_csv('./catelmo/data/test_subset.csv', header=None)

# Rename columns to maintain consistency (assuming columns: epi, tcr, binding)
dat.columns = ['epi', 'tcr', 'binding']

# Initialize new columns for embeddings
dat['tcr_embeds'] = None
dat['epi_embeds'] = None

# Add a progress bar using tqdm
tqdm.pandas(desc="Generating embeddings")  # Set description for the progress bar

# Generate embeddings with a progress bar
dat['epi_embeds'] = dat['epi'].progress_apply(lambda x: catELMo_embedding(x))
dat['tcr_embeds'] = dat['tcr'].progress_apply(lambda x: catELMo_embedding(x))

# Save the embeddings to a pickle file
# dat.to_pickle("./catelmo/pickles/train_subset.pkl")
dat.to_pickle("./catelmo/pickles/test_subset.pkl")