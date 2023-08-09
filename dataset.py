import gzip
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class Moving_MNIST(Dataset):

    def __init__(self, args):

        super(Moving_MNIST, self).__init__()

        with gzip.open(args.train_data_dir, 'rb') as f:
            self.datas = np.frombuffer(f.read(), np.uint8, offset=16)
            self.datas = self.datas.reshape(-1, *args.image_size)

        self.datas = self.datas[0: args.train_samples]

        self.image_size = args.image_size
        self.input_size = args.input_size
        self.step_length = args.step_length
        self.num_objects = args.num_objects

        self.num_frames_input = args.num_frames_input
        self.num_frames_output = args.num_frames_output
        self.num_frames_total = args.num_frames_input + args.num_frames_output

        print('Loaded {} {} samples'.format(self.__len__(), 'train'))

    def _get_random_trajectory(self, seq_length):

        assert self.input_size[0] == self.input_size[1]
        assert self.image_size[0] == self.image_size[1]

        canvas_size = self.input_size[0] - self.image_size[0]

        x = random.random()
        y = random.random()

        theta = random.random() * 2 * np.pi

        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)

        for i in range(seq_length):

            y += v_y * self.step_length
            x += v_x * self.step_length

            if x <= 0.: x = 0.; v_x = -v_x;
            if x >= 1.: x = 1.; v_x = -v_x
            if y <= 0.: y = 0.; v_y = -v_y;
            if y >= 1.: y = 1.; v_y = -v_y

            start_y[i] = y
            start_x[i] = x

        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)

        return start_y, start_x

    def _generate_moving_mnist(self, num_digits=2):

        data = np.zeros((self.num_frames_total, *self.input_size), dtype=np.float32)

        for n in range(num_digits):

            start_y, start_x = self._get_random_trajectory(self.num_frames_total)
            ind = np.random.randint(0, self.__len__())
            digit_image = self.datas[ind]

            for i in range(self.num_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.image_size[0]
                right = left + self.image_size[1]
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]

        return data

    def __getitem__(self, item):

        num_digits = random.choice(self.num_objects)

        images = self._generate_moving_mnist(num_digits)

        inputs = torch.from_numpy(images[:self.num_frames_input]).permute(0, 3, 1, 2).contiguous()
        targets = torch.from_numpy(images[self.num_frames_output:]).permute(0, 3, 1, 2).contiguous()

        inputs /= 255.
        targets /= 255.

        return inputs, targets

    def __len__(self):
        return self.datas.shape[0]


class Moving_MNIST_Test(Dataset):
    def __init__(self, args):
        super(Moving_MNIST_Test, self).__init__()

        self.data_file = args.test_data_dir

        self.num_frames_input = args.num_frames_input
        self.num_frames_output = args.num_frames_output
        self.num_frames_total = args.num_frames_input + args.num_frames_output

        self.data_list = os.listdir(self.data_file)
        self.samples_list = []

        for data in self.data_list:
            data_path = os.path.join(self.data_file, data)
            self.samples_list.append(data_path)

        print('Loaded {} {} samples '.format(self.__len__(), 'test'))

    def _get_data(self, index):
        data_ptah = self.samples_list[index]
        images = np.fromfile(data_ptah, dtype='f').reshape(self.num_frames_total, 1, 64, 64)

        return images

    def __getitem__(self, index):
        images = self._get_data(index)

        inputs = torch.from_numpy(images[:self.num_frames_input]).contiguous()
        targets = torch.from_numpy(images[self.num_frames_output:]).contiguous()

        return inputs, targets

    def __len__(self):
        return len(self.samples_list)
