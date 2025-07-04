# EcoPerceiver

EcoPerceiver is a multimodal transformer model based on the Perceiver architecture designed to predict carbon fluxes from biophysical data. The model integrates multiple data modalities including eddy covariance measurements, MODIS satellite imagery, phenocam data, and site characteristics to forecast ecosystem carbon dynamics.

## Features

- **Multimodal Architecture**: Integrates eddy covariance data, satellite imagery (MODIS), phenocam data, and geographical information
- **Perceiver-based Design**: Uses the Perceiver architecture for efficient cross-attention between different data modalities
- **Flexible Configuration**: Highly configurable model architecture with support for weight sharing and custom layer configurations
- **Carbon Flux Prediction**: Predicts key carbon cycle variables including Net Ecosystem Exchange (NEE), Gross Primary Productivity (GPP), ecosystem respiration (RECO), and methane (FCH4)
- **Distributed Training**: Supports distributed training for large-scale experiments

## Installation

### Prerequisites
- Python ≥ 3.9
- PyTorch 2.0
- CUDA-compatible GPU (recommended for training)
- Appropriate data - training is done with the [CarbonSense dataset](https://zenodo.org/records/15586300)

### Install from source
```bash
git clone https://github.com/mjfortier/EcoPerceiver.git
cd EcoPerceiver
pip install -e .
```

## Quick Start

### Basic Usage

```python
from ecoperceiver.model import EcoPerceiver
from ecoperceiver.components import EcoPerceiverConfig
from ecoperceiver.dataset import EcoPerceiverDataset, EcoPerceiverLoaderConfig

# Configure the model
config = EcoPerceiverConfig(
    latent_space_dim=128,
    num_frequencies=12,
    input_embedding_dim=32,
    context_length=32,
    targets=['NEE']
)

# Initialize the model
model = EcoPerceiver(config)

# Configure the dataset
dataset_config = EcoPerceiverLoaderConfig(
    targets=['NEE'],
    context_length=32,
    use_modis=True,
    use_phenocam=True
)

# Load data
dataset = EcoPerceiverDataset(path_to_data_dir, dataset_config)
```

From here you can create a standard PyTorch dataloader from the dataset object, and begin feeding batches into the model.

## Model Architecture

EcoPerceiver is based on the Perceiver architecture and processes multiple input modalities:

### Input Modalities
1. **Eddy Covariance Data**: Time series of meteorological and flux measurements
2. **MODIS Satellite Data**: Remote sensing imagery for vegetation indices and land surface properties
3. **Phenocam Data**: RGB and infrared imagery from ground-based phenological cameras
4. **Geographical Information**: Site coordinates, elevation, and IGBP land cover classification

### Architecture Components
- **Input Modules**: Specialized encoders for each data modality
- **Cross-Attention Layers**: Perceiver-style attention between latent space and input modalities (including a windowed variety)
- **Self-Attention Layers**: Processing within the latent space
- **Output Module**: Linear projection to target carbon flux variables
## Project Structure

```
ecoperceiver/
├── ecoperceiver/           # Main package
│   ├── __init__.py
│   ├── model.py           # EcoPerceiver model implementation
│   ├── components.py      # Model components and configurations
│   ├── dataset.py         # Data loading and preprocessing
│   └── constants.py       # Variable definitions and constants
├── experiments/           # Experiment scripts and configurations
│   ├── config.yml        # Default configuration
│   ├── run_experiment.py # Main training script
│   ├── submit_CC.py      # Cluster submission script
│   └── runs/             # Experiment outputs
├── setup.py              # Package installation
└── README.md             # This file
```

## License

This project uses the CC-BY 4.0 license