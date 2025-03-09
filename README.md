# Setup

This repository contains various setup scripts and configurations for different environments and purposes, with a focus on explicit and implicit analysis.

## Project Structure

The repository is organized into the following main directories:

```
Setup/
├── explizite_Analyse/
│   ├── charts/
│   ├── statistics/
│   ├── compute_statistics.py
│   └── fragenkatalog.json
├── implizite_Analyse/
│   ├── 1_create_descriptions/
│   ├── 2_analysis/
│   └── 2_analysis/modells/
├── modells.py
├── modelltester.py
└── README.md
```

### explizite_Analyse

This directory contains scripts and data for explicit analysis.

- **charts/**: Directory for storing generated charts.
- **statistics/**: Directory for storing computed statistics.
- **compute_statistics.py**: Script to compute statistics and generate visualizations.
- **fragenkatalog.json**: JSON file containing questions and their corresponding axes for analysis.

### implizite_Analyse

This directory contains scripts and data for implicit analysis.

- **1_create_descriptions/**: Directory for scripts that create descriptions.
- **2_analysis/**: Directory for analysis scripts.
- **2_analysis/modells/**: Directory for models used in the analysis.

## Languages and Technologies

- **Python**: This repository is primarily composed of Python scripts.

## Getting Started

To get started with the setup scripts in this repository, follow the instructions below.

### Prerequisites

Ensure you have the following installed on your machine:

- Python (version 3.7 or higher)
- pip (Python package installer)
- API Keys for LLMs

### Installation

1. Clone this repository to your local machine:

    ```sh
    git clone https://github.com/alexostgit/Setup.git
    ```

2. Navigate to the repository directory:

    ```sh
    cd Setup
    ```

3. Install the required Python packages

4. Setup API Keys locally 

    ```sh
    export OPENAI_API_KEY="YOUR_API_KEY_HERE"
    ...
    ```

4. Run the scripts

## Usage

### Explicit Analysis

To compute statistics and generate visualizations for explicit analysis:

1. Ensure your data is available at the specified path in the script (`explizite_Analyse/scoring.csv`).
2. Run the script:

    ```sh
    python explizite_Analyse/compute_statistics.py
    ```

### Implicit Analysis

Provide instructions on how to use the scripts or configurations for implicit analysis based on the specific needs of your project.


## Contact

If you have any questions or suggestions, feel free to open an issue or contact the repository owner.
