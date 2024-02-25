# -*- coding: utf-8 -*-
# @File    : train_option.py
# @Software: PyCharm
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--test', type=bool, default=False, help='flag')

        self.isTrain = True
        return parser


if __name__ == '__main__':
    opt = TrainOptions().parse()

    print(opt)