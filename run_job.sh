# LSBATCH: User input
#!/bin/bash
#BSUB -n 8
#BSUB -W 1200
#BSUB -R "rusage[mem=8]"
#BSUB -o output_file.j%J
#BSUB -e error_file.j%J
#BSUB -J my_python_job


source ~/.bashrc
conda activate  /usr/local/usrapps/iselingzhang/sarabi/tensorimage

# Run RFTL-S on real degradation data
mkdir -p results

echo "=== RFTL-S Real Data ==="
python3 code/rftl_s_real.py --n-repeats 50 --n-workers 8 --output-dir results
