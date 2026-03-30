for win_size in 40 60 70 80 90 100
do
python -u main.py\
        --mode train\
        --dataset SMD\
        --input_c 38\
        --output_c 38\
        --zero_probability 0.3\
        --win_size $win_size\
        --data_path ./data/SMD/SMD/
done