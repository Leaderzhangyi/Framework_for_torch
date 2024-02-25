# -*- coding: utf-8 -*-
# @File    : test_option.py
# @Software: PyCharm

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='beat_params.pth', help='saves results here.')
        parser.add_argument('--test', type=bool, default=True, help='flag')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        # Set the default = 5000 to test the whole test set.
        parser.add_argument('--num_test', type=int, default=5000, help='how many test images to run')
        # To avoid cropping, the load_size should be the same as crop_size
        self.isTrain = False
        return parser
