import sys
import numpy as np
import pandas as pd
import random
import os
import torch
import pickle


class ImageGridSequence:
    def __init__(self,
                 num_grids,
                 balance=['topic', 't_f']):
        self.num_grids = num_grids
        self.balance = balance
        self.grid_dict, self.grid_info_dict = self.create_grid_sequence(num_grids)

    def load_image_df(self):
        path = '../data/DEF_images_data.xlsx'
        image_df = pd.read_excel(path, index_col=0)
        return image_df

    def load_image_embeddings(self):
        path = '../data/resnet50_images_embedding.pt'
        image_embeddings = torch.load(path)
        return image_embeddings

    def create_grid_sequence(self, num_grids):
        grid_dict = {}
        grid_info_dict = {}
        classes = {}
        classes_count = {}

        image_df = self.load_image_df()
        groups = image_df.groupby(self.balance)
        for group in groups:
            temp = group[1].groupby(['authenticity'])
            classes[group[0]] = {}
            classes_count[group[0]] = {}
            for authenticity_class in temp:
                classes[group[0]][authenticity_class[0]] = [(authenticity_class[1]['path_drive_not_annotated'][i] + os.path.sep + i) for i in
                                                            authenticity_class[1].index.tolist()]
                classes_count[group[0]][authenticity_class[0]] = [0, len(classes[group[0]][authenticity_class[0]])]

        image_list = []
        for grid in range(num_grids):
            grid_dict[grid] = []
            grid_info_dict[grid] = []
            for group in groups:
                min_elements = 1000
                min_classes = []
                for class_k, class_v in classes[group[0]].items():
                    len_classes = classes_count[group[0]][class_k][0]
                    if len_classes < min_elements and len_classes < classes_count[group[0]][class_k][1]:
                        min_elements = len_classes
                        min_classes = [class_k]
                        # classes_count = classes_count[group[0]][class_k][1]
                    elif len_classes == min_elements and len_classes < classes_count[group[0]][class_k][1]:
                        min_classes.append(class_k)
                select_class_k = random.choice(min_classes)
                select_class_v = classes[group[0]][select_class_k]
                topic_sample = random.choice(select_class_v)
                select_class_v.remove(topic_sample)
                classes_count[group[0]][select_class_k][0] += 1
                grid_dict[grid].append(topic_sample)
                grid_info_dict[grid].append((group[0][0], select_class_k))
                image_list.append(topic_sample)

            intermediate_list = list(zip(grid_dict[grid], grid_info_dict[grid]))
            random.shuffle(intermediate_list)
            grid_dict[grid], grid_info_dict[grid] = zip(*intermediate_list)

        return grid_dict, grid_info_dict

    def get_image_grid_sequence(self):
        return self.grid_dict

    def get_image_grid_info(self):
        return self.grid_info_dict


def save_grid(grid_sequence, filename: str):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump({'grid_sequence': grid_sequence}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_grid(filename: str):
    with open(filename+'.pickle', 'rb') as handle:
        return pickle.load(handle)['grid_sequence']


if __name__ == '__main__':
    test_grid = ImageGridSequence(5)
    print(test_grid.get_image_grid_sequence())
    print(test_grid.get_image_grid_info())
