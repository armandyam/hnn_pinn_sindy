# AI Safety and Physics: Neural Networks for Equation Discovery

This project implements a comprehensive pipeline for comparing different neural network approaches to learning physics and extracting interpretable equations. The goal is to understand how different inductive biases affect the long-term stability and interpretability of learned dynamical systems.

## Project Overview

The project compares three neural network approaches:

1. **Baseline Neural Network**: Standard neural network that fits data without physics constraints
2. **Physics-Informed Neural Network (PINN)**: Incorporates physics constraints into the loss function
3. **Hamiltonian Neural Network (HNN)**: Learns a Hamiltonian function and enforces energy conservation

## Key Research Questions

- How do different inductive biases affect long-term stability?
- Which approach yields the most interpretable discovered equations?
- Do physics-informed models lead to better symbolic regression results?
- How does energy conservation affect equation discovery?

## Project Structure

```
apart_hamiltonian/
├── data_generation.py          # Synthetic data generation
├── models/
│   ├── nn_baseline.py         # Baseline neural network
│   ├── pinn.py                # Physics-Informed Neural Network
│   └── hnn.py                 # Hamiltonian Neural Network
├── regression/
│   └── symbolic_regression.py # PySINDy-based equation discovery
├── validation/
│   └── compare_long_term.py   # Long-term dynamics comparison
├── main.py                    # Complete pipeline execution
├── requirements.txt           # Dependencies
└── README.md                 # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd apart_hamiltonian
```

2. Set up virtual environment and install dependencies:
```bash
./setup_venv.sh
```

This will:
- Create a Python virtual environment
- Install all required dependencies
- Create necessary project directories

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

Or use the convenience script:
```bash
./activate.sh
```

## Usage

### Quick Test (Recommended for first run)
```bash
source venv/bin/activate
python main.py test
```
This runs the complete pipeline with just the damped harmonic oscillator system.

### Full Pipeline
```bash
source venv/bin/activate
python main.py full
```
This runs the complete pipeline with both damped oscillator and simple pendulum systems.

### Using the convenience script
```bash
./activate.sh python main.py test
./activate.sh python main.py full
```

### Individual Steps

You can also run individual components:

1. **Data Generation**:
```bash
source venv/bin/activate
python data_generation.py
```

2. **Train Individual Models**:
```bash
source venv/bin/activate
python models/nn_baseline.py
python models/pinn.py
python models/hnn.py
```

3. **Symbolic Regression**:
```bash
source venv/bin/activate
python regression/symbolic_regression.py
```

4. **Long-term Validation**:
```bash
source venv/bin/activate
python validation/compare_long_term.py
```

Or using the convenience script:
```bash
./activate.sh python data_generation.py
./activate.sh python models/nn_baseline.py
./activate.sh python models/pinn.py
./activate.sh python models/hnn.py
./activate.sh python regression/symbolic_regression.py
./activate.sh python validation/compare_long_term.py
```

## Methodology

### Step 1: Data Generation
- Generate synthetic data for damped harmonic oscillator and simple pendulum
- Add noise and subsample to simulate real-world constraints
- Save data to CSV files for training

### Step 2: Neural Network Training
- **Baseline NN**: Standard regression network trained on position vs time
- **PINN**: Incorporates physics constraints (second-order ODE form)
- **HNN**: Learns Hamiltonian H(q,p) and enforces energy conservation

### Step 3: Symbolic Regression
- Use PySINDy to extract explicit equations from neural network predictions
- Compare discovered equations with true physics
- Analyze interpretability and accuracy

### Step 4: Long-term Validation
- Integrate discovered equations forward in time
- Compare with true system dynamics
- Analyze stability, energy conservation, and error growth

## Systems Studied

### Damped Harmonic Oscillator
- **True Equation**: `ẍ + 0.1ẋ + x = 0`
- **State Variables**: Position x, Velocity ẋ
- **Characteristics**: Linear, damped, good for comparing conservation

### Simple Pendulum
- **True Equation**: `θ̈ + 0.1θ̇ + 9.81sin(θ) = 0`
- **State Variables**: Angle θ, Angular velocity θ̇
- **Characteristics**: Nonlinear, energy-conserving, ideal for HNN

## Expected Results

### Long-term Stability
- **Baseline NN**: Likely to show energy drift and instability
- **PINN**: Should show improved stability due to physics constraints
- **HNN**: Should maintain energy conservation over long times

### Equation Discovery
- **Baseline NN**: May discover spurious terms due to overfitting
- **PINN**: Should find cleaner equations closer to true physics
- **HNN**: Should discover equations that respect conservation laws

### Interpretability
- HNN-based equations should be more interpretable and physically plausible
- PINN should provide a good balance between accuracy and interpretability
- Baseline NN may provide accurate short-term predictions but poor interpretability

## Output Files

The pipeline generates several output directories:

- `data/`: Generated datasets (CSV files)
- `models/saved/`: Trained neural network models
- `results/`: Symbolic regression results (JSON files)
- `plots/`: Visualization plots (PNG files)

## Key Metrics

1. **RMSE**: Root Mean Square Error vs true trajectory
2. **Energy Variance**: Measure of energy conservation
3. **Equation Complexity**: Number of terms in discovered equations
4. **Long-term Stability**: Error growth over extended time periods

## Dependencies

- **PyTorch**: Neural network training
- **SciPy**: ODE integration and numerical methods
- **PySINDy**: Symbolic regression
- **Matplotlib**: Visualization
- **NumPy/Pandas**: Data manipulation

## Contributing

This project is designed for research in AI safety and physics. Contributions are welcome, particularly:

- Additional physical systems
- New neural network architectures
- Improved symbolic regression methods
- Enhanced validation metrics

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information when published]
```

## Contact

[Add contact information] 