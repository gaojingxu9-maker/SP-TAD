for K in 1 2 3 4 5
do
python -u main.py\
        --mode train\
        --dataset MSL\
        --input_c 55\
        --output_c 55\
        --zero_probability 0.2\
        --n_memory 5\
        --topk 10\
        --read_K 10\
        --beta_1 0.5\
        --beta_2 0.4\
        --data_path ./data/MSL/MSL/

python -u main.py\
        --mode train\
        --dataset SMAP\
        --input_c 25\
        --output_c 25\
        --zero_probability 0.5\
        --n_memory 3\
        --topk 60\
        --read_K 3\
        --beta_1 0.7\
        --beta_2 0.1\
        --data_path ./data/SMAP/SMAP/

python -u main.py\
        --mode train\
        --dataset PSM\
        --input_c 25\
        --output_c 25\
        --zero_probability 0.3\
        --n_memory 15\
        --topk 90\
        --read_K 4\
        --beta_1 0.4\
        --beta_2 0.8\
        --data_path ./data/PSM/PSM/

python -u main.py\
        --mode train\
        --dataset SWaT\
        --input_c 51\
        --output_c 51\
        --zero_probability 0.3\
        --n_memory 5\
        --topk 5\
        --read_K 7\
        --beta_1 0.8\
        --beta_2 0.1\
        --data_path ./data/SWaT/SWaT/
done