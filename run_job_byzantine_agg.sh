# LSBATCH: User input
#!/bin/bash
#BSUB -n 10
#BSUB -W 8640
#BSUB -R "rusage[mem=8]"
#BSUB -o output_file.j%J
#BSUB -e error_file.j%J
#BSUB -J rftl_byzantine_agg


source ~/.bashrc
conda activate  /usr/local/usrapps/iselingzhang/sarabi/tensorimage

# Byzantine-robust aggregation benchmark: 6 rules (sum/median/trimmed_mean/
# geometric_median/krum/multi_krum) vs a physically-motivated hot_block
# faulty-camera attack, n=8 synthetic clients, matched-clean-reference design.
python code/byzantine_agg_real.py --data-path /rs1/researchers/x/xfang8/chapter3/Robust-Federated-Tensor-Learning/data/ --n-repeats 50 --n-workers 10 --output-dir /rs1/researchers/x/xfang8/chapter3/Robust-Federated-Tensor-Learning/output_byzantine_agg/
