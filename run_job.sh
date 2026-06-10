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

# Run experiments
#DATA="data/Simulated Data/Simulated Data"

#echo "=== RFTL-S ==="
#python3 code/rftl_s.py --data-path "$DATA" --n-repeats 10 --n-workers 8

#echo "=== RFTL-U ==="
#python3 code/rftl_u.py --data-path "$DATA" --n-repeats 10 --n-workers 8

#echo "=== RFTL-21 ==="
#python3 code/rftl_21.py --data-path "$DATA" --n-repeats 10 --n-workers 8
python code/rftl_s_real.py --data-path /rs1/researchers/x/xfang8/chapter3/Robust-Federated-Tensor-Learning/data/ --n-repeats 50 --n-workers 10 --output-dir /rs1/researchers/x/xfang8/chapter3/Robust-Federated-Tensor-Learning/output/
