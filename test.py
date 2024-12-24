import json
import warnings

import torchvision

from metrics.uciqe import batch_uciqe

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

from config import Config
from data import get_data
from models import *

from utils import *

warnings.filterwarnings('ignore')


def test():
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    accelerator = Accelerator()
    device = accelerator.device

    # Data Loader
    val_dir = opt.TESTING.VAL_DIR

    val_dataset = get_data(val_dir, opt.TESTING.INPUT, opt.TESTING.TARGET, 'test', opt.TRAINING.ORI,
                           {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    # Model & Metrics
    model = UIR_PolyKernel()

    load_checkpoint(model, opt.TESTING.WEIGHT)

    model, testloader = accelerator.prepare(model, testloader)

    model.eval()

    size = len(testloader)
    stat_psnr = 0
    stat_ssim = 0
    stat_uciqe = 0

    for _, test_data in enumerate(tqdm(testloader)):
        # get the inputs; data is a list of [targets, inputs, filename]
        inp = test_data[0].contiguous()
        tar = test_data[1]

        with torch.no_grad():
            res = model(inp)

        if not os.path.isdir(opt.TESTING.RESULT_DIR):
            os.makedirs(opt.TESTING.RESULT_DIR)
        torchvision.utils.save_image(res, os.path.join(opt.TESTING.RESULT_DIR, test_data[2][0]))

        stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
        stat_ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
        stat_uciqe += batch_uciqe(res)

    stat_psnr /= size
    stat_ssim /= size
    stat_uciqe /= size

    test_info = ("Test Result on {}, check point {}, testing data {}".
                 format(opt.MODEL.SESSION, opt.TESTING.WEIGHT, opt.TESTING.VAL_DIR))
    log_stats = ("PSNR: {}, SSIM: {}, UCIQUE: {}".
                 format(stat_psnr, stat_ssim, stat_uciqe))
    print(test_info)
    print(log_stats)
    with open(os.path.join(opt.LOG.LOG_DIR, opt.TESTING.LOG_FILE), mode='a', encoding='utf-8') as f:
        f.write(json.dumps(test_info) + '\n')
        f.write(json.dumps(log_stats) + '\n')


if __name__ == '__main__':
    os.makedirs('result', exist_ok=True)
    test()
