# NSDUH
Exploration of data from the NSDUH

## Setup Instructions

### Prerequisites
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system
- Git (to clone this repository)

### Quick Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/AllenDowney/NSDUH
   cd NSDUH
   ```

2. **Create and activate the conda environment**:
   ```bash
   make create_environment
   conda activate NSDUH
   ```

3. **Install dependencies**:
   ```bash
   make requirements
   ```

### Manual Setup (Alternative)

If you prefer to set up manually instead of using the Makefile:

1. **Create the conda environment**:
   ```bash
   conda create --name NSDUH python=3.11 -y
   ```

2. **Activate the environment**:
   ```bash
   conda activate NSDUH
   ```

3. **Install dependencies**:
   ```bash
   python -m pip install -U pip setuptools wheel
   python -m pip install -r requirements-dev.txt
   ```

### Available Make Commands

- `make create_environment` - Creates the conda environment with Python 3.11
- `make requirements` - Installs all project dependencies
- `make delete_environment` - Removes the conda environment
- `make clean` - Removes Python cache files
- `make lint` - Runs code linting with flake8 and black
- `make format` - Formats code with black
- `make tests` - Runs tests on Jupyter notebooks

### Environment Details

- **Python Version**: 3.11
- **Environment Name**: NSDUH
- **Key Dependencies**: 
  - Data Science: numpy, pandas, matplotlib, seaborn, scipy, statsmodels
  - Jupyter: jupyter, notebook, jupyterlab, nbmake
  - Development: black, flake8, pytest
  - Data Processing: pyreadstat, openpyxl, tables

### Getting Started

After setup, you can:
- Start Jupyter Lab: `jupyter lab`
- Start Jupyter Notebook: `jupyter notebook`
- Run tests: `make tests`
- Format code: `make format`

### Troubleshooting

- If you encounter permission issues, ensure conda is properly installed and in your PATH
- If packages fail to install, try updating conda: `conda update conda`
- For environment activation issues, ensure you're using the correct shell (bash/zsh)
