# LSBATCH: User input
#!/bin/bash
#BSUB -n 10
#BSUB -W 8640
#BSUB -R "rusage[mem=8]"
#BSUB -o output_file.j%J
#BSUB -e error_file.j%J
#BSUB -J rftl_byzantine_hijack


source ~/.bashrc
conda activate  /usr/local/usrapps/iselingzhang/sarabi/tensorimage

# Byzantine hijack-attack benchmark: same 6 rules and 8-client split as
# byzantine_agg_real.py, worst-case hand-crafted scatter-matrix attack
# targeting the honest data's weakest eigendirection, amplitude sweep
# {2,5,20}x honest top eigenvalue, LOO + full clean references.
python code/byzantine_hijack_real.py --data-path /rs1/researchers/x/xfang8/chapter3/Robust-Federated-Tensor-Learning/data/ --n-repeats 50 --n-workers 10 --output-dir /rs1/researchers/x/xfang8/chapter3/Robust-Federated-Tensor-Learning/output_byzantine_hijack/
