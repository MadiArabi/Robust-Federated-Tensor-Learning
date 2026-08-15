# LSBATCH: User input
#!/bin/bash
#BSUB -n 10
#BSUB -W 8640
#BSUB -R "rusage[mem=8]"
#BSUB -o output_file.j%J
#BSUB -e error_file.j%J
#BSUB -J rftl_monitoring


source ~/.bashrc
conda activate  /usr/local/usrapps/iselingzhang/sarabi/tensorimage

# SPE-chart monitoring experiment: clean/baseline/RFTL-S(IRLS) under
# structured train-side contamination of user 0 (masking/swamping).
python code/monitoring_real.py --data-path /rs1/researchers/x/xfang8/chapter3/Robust-Federated-Tensor-Learning/data/ --n-repeats 50 --n-workers 10 --output-dir /rs1/researchers/x/xfang8/chapter3/Robust-Federated-Tensor-Learning/output_monitoring/
