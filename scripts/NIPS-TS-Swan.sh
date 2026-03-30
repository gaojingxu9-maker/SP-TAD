for lamda_1 in 0.2 0.4 0.6 0.8 1 10
do
for lamda_2 in 0.2 0.4 0.6 0.8 1 10
do
python -u main.py\
      --mode train\
      --dataset NIPS_TS_Swan\
      --input_c 38\
      --output_c 38\
      --lamda_1 $lamda_1\
      --lamda_2 $lamda_2\
      --data_path ./data/NIPS_TS_Swan//NIPS_TS_Swan/
done
done
for lamda_1 in 0.2 0.4 0.6 0.8 1 10
do
for lamda_2 in 0.2 0.4 0.6 0.8 1 10
do
python -u main.py\
          --mode train \
          --dataset NIPS_TS_Water\
          --input_c 9\
          --output_c 9\
          --lamda_1 $lamda_1\
          --lamda_2 $lamda_2\
          --data_path ./data/NIPS_TS_Water/NIPS_TS_Water/
done
done