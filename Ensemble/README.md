# Gated Graph Transformer Ensemble Model: Modified Transformer based property prediction model for chemical structures

*Chiaki Nakamori* & *Tore Eriksson*

## Training
```
python train_ensemble.py --df training_data --save save_weights_folder --log save_log_folder --gpu gpu_number  
```

## Prediction

Standalone python code for property predicion is not provided at the moment. You have to write your own prediction code according to the following example.

```
from EnsembleModel import get_esemble_model
import pickle

with open(f"{save_weights_folder}/data/info", "rb") as pf:
  reg_tasks, class_tasks, d_model, dff = pickle.load(pf)

ensemble_model = get_esemble_model(f"{save_weights_folder}/result", training_data, reg_tasks, class_tasks) 
pred = ensemble_model.predict(smiles)  #smiles : a list of SMILES 
```

## Architecture

### Inputs
<img src="Image/Inputs.png" width=720>


### Encoder
<img src="Image/Encoder.png" width=880>


### Ensemble
<img src="Image/Ensemble.png" width=360>
