# Binding Affinity Prediction Models

This repository contains the code and resources for four different models used in our experiments on binding affinity prediction:

1. **ATM-TCR**: Baseline multi-head self-attention model using BLOSUM embeddings.
2. **Modified ATM-TCR**: ATM-TCR with context-aware embeddings (catELMO).
3. **TEPCAM**: Baseline TEPCAM model.
4. **Modified TEPCAM**: TEPCAM model with geometric loss.

---

## Repository Structure

This repository is organized into the following branches, each dedicated to a specific model:

- **`main`**: Contains this README and general information about the repository.
- **`ATM-TCR`**: Code for the original ATM-TCR model. Includes instructions for running and training the model.
- **`Modified-ATM-TCR`**: Code for the ATM-TCR model with context-aware embeddings (catELMO).
- **`TEPCAM`**: Code for the TEPCAM model with standard loss.
- **`Modified-TEPCAM`**: Code for the TEPCAM model modified to use geometric loss.

Each branch includes:
- The model-specific codebase.
- Instructions for setting up the environment, training, and evaluating the model.

---

## Releases

This repository includes four releases, each corresponding to the trained models and results for one of the four branches:

1. [**ATM-TCR Release**](https://github.com/imaad-uni/cse494-599-Project/releases/tag/v1.0.0-ATM-TCR)  
   Includes trained models and results for ATM-TCR with various hyperparameter configurations.

2. [**Modified ATM-TCR Release**](https://github.com/imaad-uni/cse494-599-Project/releases/tag/v1.0.0-Modified-ATM-TCR)  
   Includes trained models and results for Modified ATM-TCR with catELMO embeddings.

3. [**TEPCAM Release**](https://github.com/imaad-uni/cse494-599-Project/releases/tag/v1.0.0-TEPCAM)  
   Includes trained models and results for TEPCAM with standard loss.

4. [**Modified TEPCAM Release**](https://github.com/imaad-uni/cse494-599-Project/releases/tag/v1.0.0-Modified-TEPCAM)  
   Includes trained models and results for TEPCAM with geometric loss.
