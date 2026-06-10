# LSBATCH: User input
#!/bin/bash
#BSUB -n 8
#BSUB -W 2400
#BSUB -R "rusage[mem=8]"
#BSUB -o output_file.j%J
#BSUB -e error_file.j%J
#BSUB -J my_python_job


source ~/.bashrc
conda activate  /usr/local/usrapps/iselingzhang/sarabi/tensorimage

# Run RFTL-S on real degradation data
python code/rftl_s_real.py --data-path /rs1/researchers/x/xfang8/chapter3/Robust-Federated-Tensor-Learning/data/ --n-repeats 10 --n-workers 10 --output-dir /rs1/researchers/x/xfang8/chapter3/Robust-Federated-Tensor-Learning/output/
