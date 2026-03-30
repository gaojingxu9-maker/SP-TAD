import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

from matplotlib import pyplot as plt
from tqdm import tqdm

class Construct_Image(nn.Module):
    def __init__(self, construct_image_data=True,grid_layout=(7,8),image_size=None,
        linestyle="-", linewidth=0.5, marker="*", markersize=1, differ=True,
        override=False, outlier=None,interpolation=True,order=False,missing_ratios=[0.]):
        super(Construct_Image, self).__init__()
        # Encoder
        self.construct_image_data = construct_image_data
        self.grid_layout = grid_layout
        self.image_size = image_size
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.marker = marker
        self.markersize = markersize
        self.differ = differ
        self.override = override
        self.outlier = outlier

        self.interpolation = interpolation
        self.order = order
        self.missing_ratios = missing_ratios

        self.color_detailed_description = {
            "green": "1",
            "black": "2",
            "blue": "3",
            "brown": "4",
            "chartreuse": "5",
            "chocolate": "6",
            "coral": "7",
            "crimson": "8",
            "blueviolet": "9",
            "darkblue": "10",
            "darkgreen": "11",
            "firebrick": "12",
            "gold": "13",
            "teal": "14",
            "grey": "15",
            "indigo": "16",
            "steelblue": "17",
            "indianred": "18",
            "goldenrod": "19",
            "darkred": "20",
            "darkorange": "21",
            "magenta": "22",
            "maroon": "23",
            "navy": "24",
            "olive": "25",
            "orange": "26",
            "orchid": "27",
            "pink": "28",
            "plum": "29",
            "purple": "30",
            "red": "31",
            "cornflowerblue": "32",
            "sienna": "33",
            "darkkhaki": "34",
            "tan": "35",
            "dodgerblue": "36",
            "darkseagreen": "37",
            "cadetblue": "38"
        }

    def draw_image(self,pid, ts_values, ts_scales,
                   override, differ, outlier, interpolation,
                   missing_ratio,
                   image_size,
                   grid_layout,
                   linestyle, linewidth, marker, markersize,
                   ts_color_mapping, ts_idx_mapping):

        max_hours, num_params = ts_values.shape[0], ts_values.shape[1]

        # set matplotlib param
        assert grid_layout[0] * grid_layout[1] >= num_params
        grid_height = grid_layout[0]
        grid_width = grid_layout[1]
        if image_size is None:
            cell_height = 100
            cell_width =  100
            img_height = grid_height * cell_height
            img_width = grid_width * cell_width
        else:
            img_height = image_size[0]
            img_width = image_size[1]

        dpi = 100
        plt.rcParams['savefig.dpi'] = dpi  # default=100
        plt.rcParams['figure.figsize'] = (img_width / dpi, img_height / dpi)
        plt.rcParams['figure.frameon'] = False

        # save path
        base_path = "images"

        if interpolation:
            base_path = "interpolation_" + base_path
        base_path = "./processed_data/" + base_path
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        img_path = os.path.join(base_path, f"{pid}.png")
        if os.path.exists(img_path):
            if not override:
                return []

        drawed_params = []

        max_hours, num_params = ts_values.shape[0], ts_values.shape[1]
        selected_param_idxs = [i for i in range(num_params)]

        for param_idx in selected_param_idxs:  # ts_desc: (215, 36)
            # for param_idx in ts_orders:
            param = str(param_idx)

            ts_time = np.arange(0, max_hours, dtype=float)
            ts_value = ts_values[:, param_idx]

            # the scale of x, y axis
            param_scale_x = [0, max_hours]
            param_scale_y = ts_scales[param_idx]
            # only one value, expand the y axis
            if param_scale_y[0] == param_scale_y[1]:
                param_scale_y = [param_scale_y[0] - 0.5, param_scale_y[0] + 0.5]

            ts_time = np.array(ts_time).reshape(-1, 1)
            ts_value = np.array(ts_value).reshape(-1, 1)
            # handling missing value and extreme values
            kept_index = (ts_value != 0)
            removed_index = (ts_value == 0)
            if interpolation:
                ts_time = ts_time[kept_index]
                ts_value = ts_value[kept_index]
            else:
                ts_time[removed_index] = np.nan
                ts_value[removed_index] = np.nan
            # handling extreme values
            min_index = (ts_value < param_scale_y[0])
            ts_value[min_index] = param_scale_y[0]
            # handling extreme values
            max_index = (ts_value > param_scale_y[1])
            ts_value[max_index] = param_scale_y[1]

            ##### draw the plot for each parameter
            param_color = ts_color_mapping[param]
            param_idx = ts_idx_mapping[param]

            plt.subplot(grid_layout[0], grid_layout[1], param_idx + 1)

            if differ:  # using different colors and markers
                plt.plot(ts_time, ts_value, linestyle=linestyle, linewidth=linewidth, marker=marker,
                         markersize=markersize, color=param_color)
            else:
                plt.plot(ts_time, ts_value, linestyle=linestyle, linewidth=linewidth, marker=marker,
                         markersize=markersize)

            plt.xlim(param_scale_x)
            plt.ylim(param_scale_y)
            plt.xticks([])
            plt.yticks([])

            drawed_params.append(param)

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_path, pad_inches=0)
        plt.clf()
        return drawed_params


    def forward(self,input_data):
        '''
        x (input_data) : B x L x C
        '''
        s = input_data.data.shape
        num_params = s[2]

        ts_params = [str(i) for i in range(num_params)]  # 0,1,2,3,4,5,6,7 (8 classes)

        plt_colors = list(self.color_detailed_description.keys())
        num_colors = len(plt_colors)
        """
        for random param color exp
        """
        if num_colors < num_params:
            plt_colors = []
            rs = list(np.linspace(0.0, 1.0, num_params))
            random.shuffle(rs)  # from 0 to 1
            gs = list(np.linspace(0.0, 1.0, num_params))
            random.shuffle(gs)  # from 0 to 1
            bs = list(np.linspace(0.0, 1.0, num_params))
            random.shuffle(bs)  # from 0 to 1
            for idx in range(num_params):
                color = (rs[idx], gs[idx], bs[idx])
                plt_colors.append(color)

        # construct the mapping from param to marker, color, and idx
        ts_idx_mapping = {}
        ts_color_mapping = {}
        for idx, param in enumerate(ts_params):
            ts_color_mapping[param] = plt_colors[idx]
            ts_idx_mapping[param] = idx
        # first round, find the mean and std for each param across all the data
        all_ts_values = [[] for _ in range(num_params)]
        stat_ts_values = np.ones(shape=(num_params, 12))  # mean, std, min, max
        for idx, p in tqdm(enumerate(input_data)):
            ts_values = p  # (600,17)
            for param_idx in range(num_params):  # ts_desc: (60, 34)
                ts_value = ts_values[:, param_idx]
                ts_value = np.array(ts_value).reshape(-1, 1)
                all_ts_values[param_idx].extend(list(ts_value))
        for param_idx in range(num_params):  # ts_desc: (60, 34)
            param_ts_value = np.array(all_ts_values[param_idx])
            stat_ts_values[param_idx, 0] = param_ts_value.mean()
            stat_ts_values[param_idx, 1] = param_ts_value.std()
            stat_ts_values[param_idx, 2] = param_ts_value.min()
            stat_ts_values[param_idx, 3] = param_ts_value.max()

            """
            option 1. remove outliers with boxplot
            """
            q1 = np.percentile(param_ts_value, 25)
            q3 = np.percentile(param_ts_value, 75)
            med = np.median(param_ts_value)
            iqr = q3 - q1
            upper_bound = q3 + (1.5 * iqr)
            lower_bound = q1 - (1.5 * iqr)
            stat_ts_values[param_idx, 4] = lower_bound
            stat_ts_values[param_idx, 5] = upper_bound
            param_ts_value1 = param_ts_value[(lower_bound < param_ts_value) & (upper_bound > param_ts_value)]
            outlier_ratio = 1 - (len(param_ts_value1) / len(param_ts_value))
            # print(f"{param_idx}, {outlier_ratio}")

            """
            option 2. remove outliers with standard deviation
            """
            med = np.median(param_ts_value)
            std = np.std(param_ts_value)
            upper_bound = med + (3 * std)
            lower_bound = med - (3 * std)
            stat_ts_values[param_idx, 6] = lower_bound
            stat_ts_values[param_idx, 7] = upper_bound
            param_ts_value2 = param_ts_value[(lower_bound < param_ts_value) & (upper_bound > param_ts_value)]
            outlier_ratio = 1 - (len(param_ts_value2) / len(param_ts_value))
            # print(f"{param_idx}, {outlier_ratio}")

            """
            option 3. remove outliers with modified z-score
            """
            med = np.median(param_ts_value)
            deviation_from_med = param_ts_value - med
            mad = np.median(np.abs(deviation_from_med))
            # modified_z_score = (deviation_from_med / mad)*0.6745
            lower_bound = (-3.5 / 0.6745) * mad + med
            upper_bound = (3.5 / 0.6745) * mad + med
            stat_ts_values[param_idx, 8] = lower_bound
            stat_ts_values[param_idx, 9] = upper_bound
            param_ts_value3 = param_ts_value[(lower_bound < param_ts_value) & (upper_bound > param_ts_value)]
            outlier_ratio = 1 - (len(param_ts_value3) / len(param_ts_value))
            # print(f"{param_idx}, {outlier_ratio}")

            """
            option 4. quartile
            """
            sorted_param_ts_value = np.sort(param_ts_value)
            value_len = sorted_param_ts_value.shape[0]
            max_position = round(value_len * 0.9999)
            min_position = round(value_len * 0.0001)
            upper_bound = sorted_param_ts_value[max_position]
            lower_bound = sorted_param_ts_value[min_position]
            stat_ts_values[param_idx, 10] = lower_bound
            stat_ts_values[param_idx, 11] = upper_bound

        # the order of params
        ts_orders = list(range(num_params))

        # second round, draw the image and prompt for each sample
        for idx, p in tqdm(enumerate(input_data)):

            pid = idx
            ts_values = p  # (600, 17) time_length = 600, num_param = 17
            # normalize the values
            if not self.outlier:
                ts_scales = stat_ts_values[:, 2:4]  # no removal
            elif self.outlier == "iqr":
                ts_scales = stat_ts_values[:, 4:6]  # iqr
            elif self.outlier == "sd":
                ts_scales = stat_ts_values[:, 6:8]  # sd
            elif self.outlier == "mzs":
                ts_scales = stat_ts_values[:, 8:10]  # mzs
            elif self.outlier == "quartile":
                ts_scales = stat_ts_values[:, 10:12]  # mzs

            if self.construct_image_data:
                # draw the image for each p
                for missing_ratio in self.missing_ratios:
                    drawed_params = self.draw_image(pid, ts_values, ts_scales, self.override, self.differ,
                                                self.outlier, self.interpolation,
                                                missing_ratio,
                                                self.image_size,
                                                self.grid_layout,
                                                self.linestyle, self.linewidth, self.marker, self.markersize,
                                                ts_color_mapping, ts_idx_mapping)
                    # ImageDict = {
                    #         "id": pid,
                    #         "param_num": len(drawed_params),
                    #         "text": "",
                    #         "label": int(label[0]),
                    #         "label_name": str(int(label[0])),
                    #     }
                    #
                    # if missing_ratio == 0:
                    #     ImageDict_list.append(ImageDict)
                    # elif missing_ratio == 0.1:
                    #     ms10_ImageDict_list.append(ImageDict)
                    # elif missing_ratio == 0.2:
                    #     ms20_ImageDict_list.append(ImageDict)
                    # elif missing_ratio == 0.3:
                    #     ms30_ImageDict_list.append(ImageDict)
                    # elif missing_ratio == 0.4:
                    #     ms40_ImageDict_list.append(ImageDict)
                    # elif missing_ratio == 0.5:
                    #     ms50_ImageDict_list.append(ImageDict)



        # if construct_image_data:
        #     for idx, ID_list in enumerate(
        #             [ImageDict_list, ms10_ImageDict_list, ms20_ImageDict_list, ms30_ImageDict_list,
        #                 ms40_ImageDict_list, ms50_ImageDict_list]):
        #         print(len(ID_list))
        #         if len(ID_list) > 0:
        #             if idx == 0:
        #                 save_path = f'../processed_data/ImageDict_list.npy'
        #             else:
        #                 missing_ratio = [0., 0.1, 0.2, 0.3, 0.4, 0.5][idx]
        #                 save_path = f'../processed_data/{feature_removal_level}_{missing_ratio}_ImageDict_list.npy'
        #             np.save(save_path, ID_list)
        #             print(f"Save data in {save_path}")

        return 0