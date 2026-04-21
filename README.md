# FedSurrogate-Robust-Backdoor-Defense-in-Federated-Learning
This is an implementation of the paper FedSurrogate: Robust Backdoor Defense in Federated Learning via Anomaly Detection and Model Replacement


- **Attack Implementations:**
  - Standard backdoor attacks
  - Distributed Backdoor Attack (DBA)
  - Neurotoxin

- **Defense Mechanisms:**
  - FedSurrogate (our proposed method)
  - FoolsGold
  - FLAME
  - FedGrad
  - FLSHIELD
  - Snowball
  - AlignIns  
  - SPMC

- **Datasets:**
  - MNIST
  - Fashion-MNIST
  - CIFAR-10
  - CIFAR-100  
  

## Installation

### Requirements
```bash
pip install torch torchvision numpy matplotlib pyyaml scikit-learn scipy hdbscan
```

### Python Version
Python 3.8 or higher recommended.

## Project Structure
```
.
├── main.py                       # Main training loop
├── server.py                     # Server and aggregation logic
├── config/
│   └── default_config.yaml       # Experiment configuration
├── client/
│   └── client.py                 # Client implementations (benign and malicious)
├── models/                       # Neural network architectures
│   ├── simple_cnn.py
│   ├── cifarnet.py
│   └── 
├── utils/
│   ├── backdoor_utils.py         # Trigger generation utilities
│   ├── metric.py.py             # Metrics
│   └── visualization.py          # Plotting and analysis tools
├── defenses/                     # Defense implementations
│   ├── fedsurrogate.py
│   ├── foolsgold.py
│   ├── flame.py
│   ├── fedgrad.py
│   ├── alignins.py
│  
└── results/                      # Output directory for plots and logs
```

## Usage

### Basic Training
```bash
python main.py --config config/default_config.yaml
```

### Configuration

Edit `config.yaml` to customize experiments:
```yaml
# Select dataset
dataset_name: "cifar10"  # Options: mnist | cifar10 | fashion_mnist 

# Select model
model: "CIFAR10Model"    # Options: simple_cnn | CIFAR10Model |  resnet8

# Configure defense
defense:
  enabled: true
  type: "fedsurrogate"   # Options: fedsurrogate | | foolsgold | 
                         #          flame |  fedgrad | alignins

# Configure attack
apply_attack: true
attack_type: "backdoor"
client_settings:
  model_replacement: false
  distributed_backdoor: true
```

### Running Different Defenses

**FedSurrogate:**
```yaml
defense:
  enabled: true
  type: "fedsurrogate"
  shrink_replace: 0.3
  zeta: 0.4
```



**FoolsGold:**
```yaml
defense:
  enabled: true
  type: "foolsgold"
  recompute_mask: true
```


## Data Partitioning

The framework supports both IID and non-IID data distributions:
```yaml
data_partition:
  strategy: "dirichlet"     # Options: iid | dirichlet
  dirichlet_alpha: 0.5      # Lower = more heterogeneous
  num_classes: 10
  min_require_size: 20
```

## Attack Configuration

### Centralized Backdoor Attack
```yaml
client_settings:
  poison_data_ratio: 0.3
  backdoor_target: 1
  trigger_type: "3x3"
  backdoor_location: "bottom_right"
```

### Distributed Backdoor Attack (DBA)
```yaml
client_settings:
  distributed_backdoor: true
  num_dba_clients: 4
  pixels_per_client: 3
```


## Output

Results are saved in the `results/` directory:
- Training metrics plots
- Class distribution visualizations
- Backdoor trigger examples
- Performance logs

## Evaluation Metrics

The framework tracks:
- **Main Task Accuracy:** Clean test set accuracy
- **Attack Success Rate:** Attack effectiveness on poisoned samples
- **Detection Metrics:** True Positive Rate (TPR), False Positive Rate (FPR)



## License

MIT

## Contact

For questions or issues, please open an issue on this repository.
