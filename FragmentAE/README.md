# FragmentAE: Side chain generation using transformer-based models

*Chiaki Nakamori* & *Tore Eriksson*

## Training
```
python train_FragmentAE.py --df training_data --save save_weights_folder --gpu gpu_number  
```

## Generation

Python code for side chain generation is not provided at the moment. You have to write your own generation code according to the following example.

```
from  FragmentAPI import get_model, get_fragments

transformer = get_model(f"{save_weights_folder}/weights_file")
fragment_smiles = get_fragments(source_smiles)
```

## Architecture

### Inputs
<img src="https://github.com/TaishoPharmaceutical/MolDesignPlatform/blob/main/Ensemble/Image/Inputs.png" width=720>

### Encoder
<img src="https://github.com/TaishoPharmaceutical/MolDesignPlatform/blob/main/Ensemble/Image/Encoder.png" width=880>

### FragmentAE
<img src="Image/FragmentAE.png" width=360>
