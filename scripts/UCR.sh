for i in {1..250};
do
python -u main.py\
       --mode train \
       --dataset UCR \
       --input_c 1 \
       --output 1 \
       --index $i \
       --beta_1 0.4\
        --beta_2 0.8\
       --win_size 105\
        --data_path ./data/UCR/UCR/
done  

