# RSI-LCCA Pavement Network Tool

This repository contains the code for an RSI- and LCCA-based pavement network optimization framework.
It includes:
- Remaining Service Interval (RSI) performance modeling for flexible pavements.
- Life-Cycle Cost Analysis (LCCA) at project and network level.
- Enumeration of all feasible maintenance, rehabilitation, and reconstruction (MR&R) treatments.
- Optimization of MR&R alternatives under budget constraint.

## Installation

1. Clone the repository:
   ''bash
   git clone https://github.com/ElhamAmiri12/rsi-lcca-pavement-optimization-tool.git
   cd rsi-lcca-pavement-optimization-tool

3. Install dependencies:
   ''bash
   pip install -r requirements.txt

## Usage

After installation and creating/adjusting the input CSV in `data/`:

1. Run the per-section RSI/LCCA analysis in batch mode:
   ''bash
   python src/main.py --batch data/example_sections.csv
   --batch = path to a CSV with all sections (your input table).
   
This creates per-section result files in the outputs/ folder.

3. Aggregate the Top-3 alternatives for each section and optimize under a network budget:
   ''bash
   python src/optimization.py --outputs outputs --budget 3500000
   --outputs = the same folder where main.py wrote its results.
   --budget = your network budget (e.g., 3500000 dollars).
   
This writes _network_summary.xlsx in the outputs/ folder, including:
- All_Top3 (Top-3 alternatives per section)
- Rank_Sums (Total NPV under each rank)
- Optimization (selected MR&R alternative under the given budget)
