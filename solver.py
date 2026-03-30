import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from model.Image import Construct_Image
from model.Model import Model
from metrics.metrics import *

from utils.evaluation import evaluation_metrics
from utils.utils import *
from model.loss_functions import *
from data_factory.data_loader import get_loader_segment
import logging
def np_softmax(x):
    # 为了防止溢出，我们通常会减去最大值，这样计算指数时更稳定
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)  # 在最后一个维度上做归一化
def smooth(y,box_size):
    assert box_size%2==1, 'The bosx size should be ood'
    box=np.ones(box_size)   #/box_size
    y=np.array([y[0]]*(box_size//2)+y.tolist()+[y[-1]]*(box_size//2))
    y_smooth=np.convolve(y,box,mode='valid')
    return y_smooth


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class TwoEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
    def __call__(self, val_loss, model, path,ids ):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path,ids)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path,ids)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path,ids):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset)+ str(ids) + f'_checkpoint.pth'))
        self.val_loss_min = val_loss

torch.autograd.set_detect_anomaly(True)
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index,self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.index,self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index,self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index,self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='thre', dataset=self.dataset)
        self.build_model()
        self.criterion = nn.MSELoss()
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        self.construct_image = Construct_Image()

    def build_model(self):
        self.model = Model(win_size =self.win_size,enc_in=self.input_c, c_out=self.output_c, \
                                    d_model=self.d_model,device=self.device,
                                    n_memory=self.n_memory, mode=self.mode,
                                    dataset=self.dataset)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,betas=(0.9, 0.999))
        if torch.cuda.is_available():
            self.model.to(self.device)
    def vali(self,vali_loader):
        with torch.no_grad():
            self.model.eval()
            valid_loss_list = []
            vail_rec_loss_list = []
            vail_contrast_loss_list = []
            vail_entropy_loss_list = []
            vail_gathering_loss_list = []
            vail_regular_loss_list = []

            for i, (input_data, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                output_dict = self.model(input,mode='val')

                output_s = output_dict['out_s']
                output_t = output_dict['out_t']
                gathering_loss = output_dict['gathering_loss']
                entropy_loss = output_dict['entropy_loss']
                contrast_loss = output_dict['contrast_loss']



                regular_loss = output_dict['regular']
                #regular_loss = output_dict.get('regular', torch.tensor(0.0, device=input.device))



                rec_loss = self.criterion(output_t, input) + self.criterion(output_s, input)
                loss =  self.lamda_4 * regular_loss + rec_loss + self.lamda_1 * entropy_loss + self.lamda_2 *contrast_loss + self.lamda_3 * gathering_loss

                vail_rec_loss_list.append(rec_loss.item())
                vail_gathering_loss_list.append(gathering_loss.item())
                vail_contrast_loss_list.append(contrast_loss.item())
                vail_entropy_loss_list.append(entropy_loss.item())
                vail_regular_loss_list.append(regular_loss.item())
                valid_loss_list.append(loss.item())

            return np.average(valid_loss_list),np.average(vail_rec_loss_list), np.average(vail_gathering_loss_list),np.average(vail_contrast_loss_list),np.average(vail_entropy_loss_list),np.average(vail_regular_loss_list)

    def train(self):
        print("======================TRAIN MODE======================")
        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        train_steps = len(self.train_loader)
        early_stopping = TwoEarlyStopping(patience=5, verbose=True, dataset_name=self.dataset)
        from tqdm import tqdm
        for epoch in tqdm(range(self.num_epochs)):
            iter_count = 0
            loss_list = []
            rec_loss_list = []
            gathering_loss_list = []
            entropy_loss_list = []
            contrast_loss_list = []
            regular_loss_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                output_dict = self.model(input,mode='train')
                output_s = output_dict['out_s']
                output_t = output_dict['out_t']

                gathering_loss = output_dict['gathering_loss']
                entropy_loss = output_dict['entropy_loss']
                contrast_loss = output_dict['contrast_loss']



                regular_loss = output_dict['regular']
                #regular_loss = output_dict.get('regular', torch.tensor(0.0, device=input.device))



                rec_loss = self.criterion(output_t, input) + self.criterion(output_s, input)

                loss =  self.lamda_4 * regular_loss + rec_loss + self.lamda_1 * entropy_loss + self.lamda_2 *contrast_loss + self.lamda_3 * gathering_loss

                loss_list.append(loss.item())
                rec_loss_list.append(rec_loss.item())
                regular_loss_list.append(regular_loss.item())
                gathering_loss_list.append(gathering_loss.item())
                entropy_loss_list.append(entropy_loss.item())
                contrast_loss_list.append(contrast_loss.item())

                if (i + 1) % 200 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                self.optimizer.step()

                # for name, parameter in self.model.named_parameters():
                #     if parameter.grad is not None:
                #         print(f"{name} has gradient")
                #     else:
                #         print(f"{name} NO gradient")

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(loss_list)
            train_rec_loss = np.average(rec_loss_list)
            train_gathering_loss = np.average(gathering_loss_list)
            train_contrast_loss = np.average(contrast_loss_list)
            train_entropy_loss = np.average(entropy_loss_list)
            train_regular_loss = np.average(regular_loss_list)

            valid_loss, valid_rec, valid_gathering, valid_contrast, valid_entropy, valid_regular= self.vali(self.test_loader)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, valid_loss))
            print(
                "Epoch: {0}, Steps: {1} | TRAIN reconstruction Loss: {2:.7f} TRAIN gathering Loss: {3:.7f} TRAIN contrast Loss: {4:.7f} TRAIN entropy Loss: {5:.7f} TRAIN regular Loss: {6:.7f} ".format(
                    epoch + 1, train_steps, train_rec_loss,train_gathering_loss,train_contrast_loss,train_entropy_loss,train_regular_loss ))
            print(
                "Epoch: {0}, Steps: {1} | VALID reconstruction Loss: {2:.7f} VALID gathering Loss: {3:.7f} VALID contrast Loss: {4:.7f} VALID entropy Loss: {5:.7f} VALID regular Loss: {4:.7f} ".format(
                    epoch + 1, train_steps, valid_rec, valid_gathering, valid_contrast, valid_entropy, valid_regular))

            # self.test()

            early_stopping(valid_loss, self.model, path,self.lamda_1)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
    def test(self,test=0):
        if test == 1:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(str(self.model_save_path), str(self.dataset) + str(self.lamda_1) + '_checkpoint.pth')))
            # self.model.load_state_dict(
            #     torch.load(
            #         os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
            print("======================TEST MODE======================")
        with torch.no_grad():
            self.model.eval()
            criterion = nn.MSELoss(reduce=False)
            test_labels = []
            true_list= []
            test_s_energy = []
            test_t_energy = []
            test_gather_energy = []
            test_rec_energy = []
            gathering_loss = GatheringLoss(reduce=False)
            temperature = self.temperature
            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)
                output_dict= self.model(input)

                t_loss = output_dict['t_loss']
                s_loss = output_dict['s_loss']
                #t_loss = output_dict.get('t_loss', torch.zeros(input.size(0), input.size(1), device=input.device))
                #s_loss = output_dict.get('s_loss', torch.zeros(input.size(0), input.size(1), device=input.device))
                output_s = output_dict['out_s']
                output_t = output_dict['out_t']

                t_keys = output_dict['t_keys']
                t_values = output_dict['t_values']
                s_keys = output_dict['s_keys']
                s_values = output_dict['s_values']

                rec_loss = torch.mean(criterion(input, output_s), dim=-1) + torch.mean(criterion(input, output_t), dim=-1)
                rec_loss = torch.softmax(rec_loss/temperature, dim=-1)
                rec_loss = rec_loss.detach().cpu().numpy()
                test_rec_energy.append(rec_loss)

                s_gathering = gathering_loss(output_dict['s_query'], output_dict['representation'], s_keys, s_values)
                t_gathering = gathering_loss(output_dict['t_query'], output_dict['representation'], t_keys, t_values)
                gather_loss = torch.softmax((s_gathering['values_gathering'] + t_gathering['values_gathering'] )/temperature, dim=-1)
                gather_loss = gather_loss.detach().cpu().numpy()
                test_gather_energy.append(gather_loss)

                t_loss = torch.softmax(t_loss/temperature, dim=-1)
                t_loss = t_loss.detach().cpu().numpy()
                test_t_energy.append(t_loss)

                #s_loss = torch.softmax(s_loss/temperature, dim=-1)
                s_loss = s_loss.detach().cpu().numpy()
                test_s_energy.append(s_loss)

                test_labels.append(labels)
                input = input.detach().cpu().numpy()
                true_zhi = input[:, :, 0:1]
                true_list.append(true_zhi)

            test_rec_energy = np.concatenate(test_rec_energy, axis=0).reshape(-1)
            test_t_energy = np.concatenate(test_t_energy, axis=0).reshape(-1)
            test_s_energy = np.concatenate(test_s_energy, axis=0).reshape(-1)
            test_gather_energy = np.concatenate(test_gather_energy, axis=0).reshape(-1)

            test_rec_energy = MinMaxScaler().fit_transform(test_rec_energy.reshape(-1, 1))
            test_t_energy = MinMaxScaler().fit_transform(test_t_energy.reshape(-1, 1))
            test_s_energy = MinMaxScaler().fit_transform(test_s_energy.reshape(-1, 1))
            test_gather_energy = MinMaxScaler().fit_transform(test_gather_energy.reshape(-1, 1))
            #test_rec_energy = smooth(test_rec_energy, 5)
            # test_rec_energy = StandardScaler().fit_transform(test_rec_energy)
            # test_rec = MinMaxScaler().fit_transform(test_rec_energy.reshape(-1, 1))

            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_labels = np.array(test_labels)  # 51200
            true_list = np.concatenate(true_list, axis=0).reshape(-1, 1)  # 50688 * 1
            true_list = np.array(true_list).reshape(-1)

            x = test_rec_energy + self.beta_1 * test_gather_energy + self.beta_2 * (test_s_energy + test_t_energy)

            test_energy = np.array(x).reshape(-1)

            # test_energy = smooth(test_energy, 5)

            result = evaluation_metrics(test_labels, test_energy, test, self.dataset, self.lamda_1, self.lamda_2)

            return 0


