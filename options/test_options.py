from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # parser.add_argument("--image", type=str, default='./Data_folder/test/images/0.nii')
        # parser.add_argument("--label", type=str, default='./Data_folder/test/labels/0.nii')
        parser.add_argument("--test_result", type=str, default='/home/po/Desktop/PDData/test_result_images', help='path to the .nii result to save')
        parser.add_argument("--result", type=str, default='/home/po/Desktop/PDData/result_images', help='path to the .nii result to save')
        parser.add_argument('--phase', type=str, default='test', help='test')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument("--stride_inplane", type=int, nargs=1, default=16, help="Stride size in 2D plane")
        parser.add_argument("--stride_layer", type=int, nargs=1, default=16, help="Stride size in z direction")
        parser.add_argument("--verification_path", type=str, default='./Data_folder/test/cyclegan', help='path to the .nii result to save')
        parser.set_defaults(model='test')
        self.isTrain = False
        return parser