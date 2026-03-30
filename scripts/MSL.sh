
for lamda_1 in 0.2 0.4 0.6 0.8 1 10
do
for lamda_2 in 0.2 0.4 0.6 0.8 1 10
do
python -u main.py\
        --mode train\
        --dataset MSL\
        --input_c 55\
        --output_c 55\
        --lamda_1 $lamda_1\
        --lamda_2 $lamda_2\
        --data_path ./data/MSL/MSL/
done
done