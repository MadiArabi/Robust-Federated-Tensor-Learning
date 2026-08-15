# LSBATCH: User input
#!/bin/bash
#BSUB -n 10
#BSUB -W 8640
#BSUB -R "rusage[mem=8]"
#BSUB -o output_file.j%J
#BSUB -e error_file.j%J
#BSUB -J rftl_prediction


source ~/.bashrc
conda activate  /usr/local/usrapps/iselingzhang/sarabi/tensorimage

# Test-side prediction experiment: clean/oracle/baseline/RFTL-S(one-step)/
# RFTL-S(IRLS ablation) under structured train+test contamination of user 0.
# IMPORTANT: fresh output dir — June Gaussian checkpoints are incompatible.
python code/rftl_s_real.py --data-path /rs1/researchers/x/xfang8/chapter3/Robust-Federated-Tensor-Learning/data/ --n-repeats 50 --n-workers 10 --output-dir /rs1/researchers/x/xfang8/chapter3/Robust-Federated-Tensor-Learning/output_testside/
