# VQVAE-Chem : Graph-based molecular generative model

*Chiaki Nakamori* & *Tore Eriksson*

This model converts a ring (aromatic ring or aliphatic ring) into one token to simplify the complex graph of a compound and make it easier to learn for AI. 

## Training
```
python train_VQVAE.py --df training_data --save save_weights_folder --gpu gpu_number  
```

## Generation
```
from VQVAEChemGen import VQVAEGen

chemgen = VQVAEGen(f"{save_weights_folder}/VQVAE"))
mols, indices = chemgen.input_smiles_base_sampling(smiles, n = 200, e = 0.32)
```

## Architecture
### 1) Training
<img src="VQVAE_Image/GraphConv.png" width=800>
<img src="VQVAE_Image/Vector-Quantized.png" width=720>
<img src="VQVAE_Image/Reconstruct.png" width=800>

### 2) Generation
<img src="VQVAE_Image/Sampling.png" width=400>
