# RSI-LCCA Pavement Network Tool

This repository contains the code for an RSI- and LCCA-based pavement network optimization framework.
It includes:
- Remaining Service Interval (RSI) performance modeling for flexible pavements.
- Life-Cycle Cost Analysis (LCCA) at project and network level.
- Enumeration of all feasible maintenance, rehabilitation, and reconstruction (MR&R) treatments.
- Optimization of MR&R alternatives under budget constraint.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ElhamAmiri12/rsi-lcca-pavement-optimization-tool.git
   cd rsi-lcca-pavement-optimization-tool

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

After installation and creating/adjusting the input CSV in the data/ folder:
1. Run the per-section RSI/LCCA analysis in batch mode:
   ```bash
   python src/main.py --batch data/example_sections.csv
   •   --batch is the path to a CSV file containing all sections (your input table).
This creates per-section result files in the outputs/ folder.

2. Aggregate the Top-3 alternatives for each section and optimize under a network budget:
   ```bash
   python src/optimization.py --outputs outputs --budget 3500000
   •   --outputs = the same folder where main.py wrote its results.
   •   --budget = your network budget (e.g., 3500000 dollars).
   
The optimization step writes a file _network_summary.xlsx into the outputs/ folder, including:
- Top-3 alternatives per section,
- Total NPV under each rank,
- Coverage information,
- The selected MR&R alternative under the specified budget.
