# LSBATCH: User input
#!/bin/bash
#BSUB -n 8
#BSUB -W 600
#BSUB -R "rusage[mem=8]"
#BSUB -o output_file.j%J
#BSUB -e error_file.j%J
#BSUB -J my_python_job


source ~/.bashrc
conda activate  /usr/local/usrapps/iselingzhang/sarabi/tensorimage

# Run your script
#python3 code/motivation_pilot.py --data-path data/Simulated\ Data/Simulated\ Data --n-repeats 10 --max-files 350
 python3 code/rftl_s.py --data-path data/Simulated\ Data/Simulated\ Data --n-repeats 10 --n-workers 8
