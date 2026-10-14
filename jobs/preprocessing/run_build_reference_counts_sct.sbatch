#!/bin/bash
#SBATCH --job-name=BuildRefCountsSCT
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=08:00:00
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

REFERENCE_DIR="${REFERENCE_DIR:-workspace_outputs/colorectal_nucleus_based/intermediate/reference_scdesign3}"
SYNTHETIC_ROOT="${SYNTHETIC_ROOT:-../Bin2Cell_Validation/outputs/ground_truth/colorectal_nucleus_based/simulation/scdesign3}"
mkdir -p "$REFERENCE_DIR"

python preprocessing/build_hd_gene_allowlist.py \
  --selected-features-tsv workspace_outputs/colorectal_nucleus_based/intermediate/square_002um_filtered_nuclear/colorectal_nucleus_based_square_002um_filtered_rl.selected_features.tsv.gz \
  --output-txt "$REFERENCE_DIR/hd_selected_feature_names_unique.txt"

python preprocessing/build_reference_counts_from_scrna.py \
  --matrix-h5-path "$SYNTHETIC_ROOT/scdesign3_synthetic_counts.h5" \
  --metadata-tsv-path "$SYNTHETIC_ROOT/synthetic_metadata.tsv" \
  --cell-id-column Cell \
  --cell-type-column cell_type \
  --min-cells-per-type 50 \
  --gene-allowlist-path "$REFERENCE_DIR/hd_selected_feature_names_unique.txt" \
  --output-path "$REFERENCE_DIR/reference_counts_scdesign3_nucleus_based_aligned_hd_overlap_unique.npz" \
  --summary-path "$REFERENCE_DIR/reference_counts_scdesign3_nucleus_based_aligned_hd_overlap_unique.summary.json"

conda deactivate
