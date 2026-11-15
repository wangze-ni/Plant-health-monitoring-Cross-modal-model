# Cross-modal model for plant stress recognition

Companion code for the manuscript *Transferable dielectric-excited artificial olfaction enables early non-contact plant stress monitoring*.

The model combines eight-channel gas sensor responses with molecular and electronic descriptors of seven volatile organic compounds (VOCs). For each sensor response, it predicts the stress type (drought, salt or heat) and the stress stage (0, 3, 6, 9 or 12 hours). The chemical descriptors are shared across samples. Each new sensor response is compared with the same seven compounds.

This release provides the model architecture, sensor preprocessing functions and density functional theory (DFT) inputs. The example below runs a forward pass with randomly initialised weights. Reproducing the manuscript’s accuracy estimates requires the training and evaluation workflow and the data splits used in the study.

## Run the model on a sample input

The example was tested on a CPU under Windows with Python 3.10.8, PyTorch 2.1.0, RDKit 2025.9.3, NumPy 1.26.1, pandas 2.1.2 and openpyxl 3.1.5. From the repository root, install these packages in a Python 3.10 environment.

```sh
python -m pip install torch==2.1.0 rdkit==2025.9.3 numpy==1.26.1 pandas==2.1.2 openpyxl==3.1.5
```

Run the following in Python from the same directory. It reads the supplied DFT files, builds the molecular descriptors from SMILES strings and processes two synthetic sensor responses. The shape check verifies that all three groups of DFT features were loaded.

```python
import torch

from config import DFT_BAND_DIR, DFT_DOS_DIR, D_MODEL, MOL_DIM, TARGET_LEN
from dft_bank import build_dft_feature_bank
from model_fusion import CrossModalNetwork
from voc_registry import build_knowledge_bank_tensors

graphs, fingerprints = build_knowledge_bank_tensors()
dft = build_dft_feature_bank(band_dir=DFT_BAND_DIR, dos_dir=DFT_DOS_DIR)
assert dft.shape == (7, 41), f"Expected 7 compounds and 41 DFT features, got {dft.shape}"

torch.manual_seed(42)
model = CrossModalNetwork(d_sensor=D_MODEL, d_mol=MOL_DIM, d_dft=dft.shape[1])
model.eval()
sensor = torch.randn(2, 8, TARGET_LEN)
with torch.inference_mode():
    outputs = model(sensor, graphs, fingerprints, dft)

for name, tensor in outputs.items():
    print(name, tuple(tensor.shape))
```

| Output | Shape for two responses | Meaning |
|---|---|---|
| `logits_stress` | `(2, 3)` | Class scores in the order drought, salt, heat |
| `logits_period_cls` | `(2, 5)` | Stage scores in the order 0, 3, 6, 9, 12 hours |
| `period_reg` | `(2,)` | Continuous estimate on the stage-index scale, where indices 0 through 4 correspond to the five stages |
| `attn_voc` | `(2, 7)` | Attention weights over the seven VOCs in registry order |

## Model and chemical inputs

The sensor encoder uses multiscale temporal convolutions, a bidirectional LSTM and attention pooling to summarise each response. The molecular encoder combines a graph representation with a 2,215-element fingerprint composed of Morgan fingerprint bits and MACCS keys. A separate encoder processes 41 DFT features per compound. Cross-attention uses the sensor representation as the query and the seven compound representations as keys and values. The stress probabilities then modulate the sensor features used to predict the stage.

The compound order used by the fingerprints, DFT features and attention weights is **acetic acid, nonanal, decanoic acid, heptanal, hexanal, decanal and 2-octanone**.

| DFT feature group | Features per compound | Source |
|---|---:|---|
| Adsorption | 6 | Three adsorption energies and three derived quantities in [dft_bank.py](dft_bank.py) |
| Band structure | 15 | One band table per compound in [band_structure/](dft_inputs/band_structure/) |
| Density of states | 20 | Four descriptors each for total DOS and the C, H, O and Sn projected DOS in [DOS/](dft_inputs/DOS/) |

## Using sensor recordings

[The sensor loader](data_loader_sensor.py) accepts a ZIP archive of CSV recordings. Each CSV must have eight columns whose names start with `Chan`. The loader preserves their order in the header. It reads the stress type, stage in hours and replicate number from filenames such as `Drought_3h_1.csv`.

```python
from config import START_ROW, TARGET_LEN
from data_loader_sensor import load_records_from_zip

records = load_records_from_zip("path/to/recordings.zip", START_ROW, TARGET_LEN)
```

The loader divides each recording into five equal segments. Within each segment it extracts overlapping windows, applies a channel-wise baseline correction and resamples each window to 256 time points. Each record contains an `x` array of shape `(8, 256)` and its labels. For evaluation with a trained model, use the training subset’s per-channel mean and standard deviation when constructing `GasResponseDataset` for validation or test records.

## Source files

| Files | Role |
|---|---|
| [model_sensor.py](model_sensor.py), [model_molecule.py](model_molecule.py), [model_dft.py](model_dft.py) | Encoders for sensor responses, molecular structures and DFT features |
| [model_fusion.py](model_fusion.py) | Cross-modal attention, stress classification and stage prediction |
| [voc_registry.py](voc_registry.py) | Compound order, SMILES strings, molecular fingerprints and stress-to-VOC associations |
| [dft_bank.py](dft_bank.py), [dft_band.py](dft_band.py), [dft_dos.py](dft_dos.py) | Construction of the DFT feature matrix |
| [data_loader_sensor.py](data_loader_sensor.py), [config.py](config.py) | Sensor preprocessing, input dimensions and label mappings |
