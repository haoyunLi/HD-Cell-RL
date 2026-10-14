#!/bin/bash
#SBATCH --job-name=Cellpose-SAM
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare 
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=100G
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.err

module load anaconda3
module load cuda/12.4

# Set up environment
WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL"
cd $WORK_DIR

# Activate conda environment
echo "Activating Cellpose conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate $WORK_DIR/cellpose

# Run the Cellpose-SAM segmentation script
mkdir -p workspace_outputs/colorectal_nucleus_based/intermediate
python preprocessing/cellpose_segmentation.py \
    --img_path workspace_outputs/colorectal_nucleus_based/intermediate/cropped_visium_hd_colorectal_nucleus_based.png \
    --output_dir workspace_outputs/colorectal_nucleus_based/intermediate/cellpose_sam_colorectal_nucleus_based_output

# Deactivate conda environment
conda deactivate
