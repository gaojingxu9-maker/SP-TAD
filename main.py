import argparse
from torch.backends import cudnn
from utils.utils import *
##################
# ours
from solver import Solver
def str2bool(v):
    return v.lower() in ('true')
def set_seed(seed_value):
    if seed_value == -1:
        return
    import random
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
def main(config):
    #set_seed(2024)
    #set_seed(config.seed)
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    if (not os.path.exists('memory_item')):
        mkdir('memory_item')
    solver = Solver(vars(config))
    if config.mode == 'train':
        solver.train()
        # 计算所有行之间的余弦相似度
        solver.test(test=1)
    elif config.mode == 'test':
        solver.test(test=1)
    return solver
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anormly_ratio', type=float, default=1.0)
    parser.add_argument('--index',type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=55)
    parser.add_argument('--output_c', type=int, default=55)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='MSL')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'memory_initial'])
    parser.add_argument('--data_path', type=str, default='./data/MSL/MSL/')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--n_memory', type=int, default=20, help='number of memory items')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.1, help='潜在空间偏差温度参数')

    parser.add_argument('--lamda_1', type=float, default=1)
    parser.add_argument('--lamda_2', type=float, default=1)
    parser.add_argument('--lamda_3', type=float, default=1)
    parser.add_argument('--lamda_4', type=float, default=0.5)





    parser.add_argument('--beta_1', type=float, default=1)
    parser.add_argument('--beta_2', type=float, default=0.8)

    parser.add_argument('--device', type=str, default="cuda:0")
    #parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    config = parser.parse_args()
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
