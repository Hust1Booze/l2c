source activate l2sn
python gisp_generator.py -min_n 50 -max_n 60 -exp_dir data/GISP/train -SolveInstance 1 -n_instance 10000 -n_cpu 8
python gisp_generator.py -min_n 50 -max_n 60 -exp_dir data/GISP/valid -SolveInstance 1 -n_instance 1000 -n_cpu 8
python gisp_generator.py -min_n 50 -max_n 60 -exp_dir data/GISP/test -SolveInstance 0  -n_instance 1000  -n_cpu 8
