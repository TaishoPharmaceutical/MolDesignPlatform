# Integrated web system for deep learning-assisted chemical compound design

*Chiaki Nakamori* & *Tore Eriksson*

This is a web-based system for deep learning-assisted design of chemical molecules
that is implemented in Python using Dash and Flask. The system is suitable for deployment
in small to medium size organizations doing drug development. The following functions
are implemented:

- SMILES input and URL requests with SMILES in the query string (`/smiles_input?smiles=smiles1,smiles2,smiles3...`)
- Compound editing using a web-based chemical editor (using the [ChemDoodle Web Components](https://web.chemdoodle.com/) library).
- Compound 2-class classification and regression prediction using graph transformer-based models.
- Compound generation by applying chemical reactions *in silico*.
  + Condensation with amine
  + Condensation with carboxylic
  + Suzuki coupling with boron
  + Suzuki coupling with arylhalide
  + Reductive amination with amine
  + Reductive amination with aldehyde or ketone
  + Buchwald amination with amine
  + Buchwald amination with arylhalide
- Peripheral compund generation using VQVAE models.
- Side chain generation using transformer-based models.
- Structure-based filtering of compounds.

All models are to be trained by the user. As the system is based on Web APIs, we
believe it to be easily expandable if necessary.

## Setup

See instructions in requirement.txt

## Preprocessing and Training

- Data has to be prepared in the format of example.csv (see files in Ensemble, FragmentAE, Reactor and VQVAE) for learning.
- Edit config.json in Settings and run the next command.

```
python preprocessing_and_training.py
```

- If training is insufficient, you can train a specific model using the "--RunType" argument.

```
python preprocessing_and_training.py --RunType train_ensemble
python preprocessing_and_training.py --RunType train_fragment
python preprocessing_and_training.py --RunType train_VQVAE
```

## Run

```
python run.py
```
