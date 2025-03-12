# Setup

This repository contains various setup scripts and configurations for different environments and purposes, with a focus on explicit and implicit analysis.

## Project Structure

The repository is organized into the following main directories:

```
Setup/
├── explizite_Analyse/
│   ...
├── implizite_Analyse/
│   ...
├── modells.py
├── modells_tester.py
└── README.md
```

### explizite_Analyse

This directory contains scripts and data for explicit analysis.


### implizite_Analyse

This directory contains scripts and data for implicit analysis.


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
