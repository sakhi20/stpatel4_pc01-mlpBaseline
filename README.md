# PC1 - MLP Baseline for Acoustic Classification

The notebooks in this repo will walk you through a basic exploratory data analysis (`ExploreData.ipynb`), the extraction of Mel‑Frequency Cepstral Coefficients (MFCCs) from audio data (`ExtractMFCC.ipynb`), and the training and evaluation of a basic MLP model (`MLPModel.ipynb`). Make sure you go over each notebook in the order specified. You may need to make some minor modifications to the them (e.g., changing directory paths or changing the names of the functions called). The scripts are set up to run in the JupyterHub given that the training data is placed in `/home/jovyan/Data/Train/`.

Your main tasks are the following as part of the MLP model training:

1. Modify the `MLP_Advanced` class in the `models.py` file so the new model has three layers (including the output layer), with 64 neurons on the hidden layers and dropout layers after each hidden layer (i.e., including affine transformation and non-linear activations) with a rate of 0.2. Use ReLU activation functions, and build on the `Sequential` architecture from the `MLP_Baseline` class.

2. Modify the `loss_Advanced` function in the `models.py` file so it returns a cross-entropy loss with class weights `[1.0, 4.0, 4.0, 1.0]`. Build on the `loss_Baseline` function.

Complete these tasks after you completed the exploratory data analysis and the MFCC extraction portions. Note that you are welcome to modify the notebooks as needed to help you debug your code, but you should only modify the python scripts between the delineated sections starting with
```
### <--- START OF YOUR CODE
```
and ending with
```
### END OF YOUR CODE --->
```
