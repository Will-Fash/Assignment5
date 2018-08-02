import numpy as np


class DataSet:
    def __init__(self):
        self.tranning_data = None
        self.tranning_label = None
        self.testing_data = None

    def get_test_data_set(self, file_name='test.txt'):
        if self.testing_data is not None:
            return self.testing_data
        with open(file_name, 'r', encoding="utf8")as f:
            self.testing_data = f.readlines()
        return self.testing_data

    def get_train_data_set(self, file_name='train.txt'):
        if self.tranning_data is not None:
            return self.tranning_data, self.tranning_label
        else:
            self.tranning_data = list()
            self.tranning_label = list()
            try:
                with open(file_name, 'r', encoding="utf8") as f:
                    lines = f.readlines()
                for line in lines:
                    arr = line.split("\t")
                    self.tranning_data.append(arr[0])
                    self.tranning_label.append(arr[1].replace("\n", ''))
            except Exception as ex:
                print(ex)
            class_name = ['bg', 'mk', 'bs', 'hr', 'sr', 'cz', 'sk'
                , 'es-AR', 'es-ES', 'pt-BR', 'pt-PT', 'id', 'my', 'xx']
            return np.array(self.tranning_data), np.array(self.tranning_label), class_name