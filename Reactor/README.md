# ReactorAPI

*Chiaki Nakamori* & *Tore Eriksson*

- Compound generation by applying chemical reactions *in silico*.
  + Condensation with amine
  + Condensation with carboxylic
  + Suzuki coupling with boron
  + Suzuki coupling with arylhalide
  + Reductive amination with amine
  + Reductive amination with aldehyde or ketone
  + Buchwald amination with amine
  + Buchwald amination with arylhalide

## Run
```
python make_reactants_table.py --df example.csv
```
Change host and port in ReactorAPI.py and run the following code.
```
python ReactorAPI
```

## JSON file setting for POST-HTTP
{  
&emsp;&emsp;smiles: SMILES  
&emsp;&emsp;reaction: "condensation-with-amine", "condensation-with-carboxylic", "suzuki-with-boron",  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;"suzuki-with-arylhalide", "reductive-amination-with-amine", "reductive-amination-with-aldehyde-or-ketone",  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;"buchwald-amination-with-amine" or "buchwald-amination-with-arylhalide"  
&emsp;&emsp;MolWt_min: minimum mol weight (if needed)  
&emsp;&emsp;MolWt_max: maximum mol weight (if needed)  
&emsp;&emsp;ArNum_min: minimum number of aromatic ring (if needed)  
&emsp;&emsp;ArNum_max: maximum number of aromatic ring (if needed)  
}
