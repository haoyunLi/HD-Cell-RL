#!/bin/bash
#SBATCH --job-name=BuildNucleiTable
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=80G
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.err

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL"
cd "$WORK_DIR"

echo "Activating conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

mkdir -p workspace_outputs/colorectal_nucleus_based/intermediate/nuclei

python preprocessing/build_nuclei_table_from_nuclear_bins.py \
  --nuclear-bins-path workspace_outputs/colorectal_nucleus_based/intermediate/ground_truth/colorectal_nucleus_based_square_002um_nuclear_bins.csv.gz \
  --output-path workspace_outputs/colorectal_nucleus_based/intermediate/nuclei/colorectal_nucleus_based_nuclei.parquet \
  --summary-path workspace_outputs/colorectal_nucleus_based/intermediate/nuclei/colorectal_nucleus_based_nuclei.summary.json \
  --bin-size-um 2.0 \
  --microns-per-pixel 0.2737012522439323
