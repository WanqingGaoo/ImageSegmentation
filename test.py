import os
import utils.metrics
from models import unet
import matplotlib.pyplot as plt
from data.VOCSegCProcess import *
import argparse
from utils.seed import set_random_seed
def modelTest(args):
    set_random_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    palette = Palette()

    input_transform = MaskToTensor_input(args.resize_to)
    target_transform = MaskToTensor()
    test_dataset = VOC('test', args.data_path, args.resize_to, transform=input_transform, target_transform=target_transform, max_files=args.max_files)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    model = unet.UNet(args.num_class)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()  # Put in eval mode (disables batchnorm/dropout) !
    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing
        pacc = 0
        miou = 0
        num_iter = 0
        for iter, (inputs, labels, names) in enumerate(test_loader):
            print('iter',iter)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            pacc += utils.metrics.pixel_acc(outputs, labels).item()
            miou += utils.metrics.iou(outputs, labels).item()
            num_iter += 1
            if args.save_img:
                for idx in (0, args.batch_size-1):
                    imgct = labels[idx].cpu().numpy()
                    imgcy = np.argmax(outputs[idx].cpu().numpy(), axis=0)
                    imgt = np.zeros((imgct.shape[0], imgct.shape[1], 3))
                    imgy = np.zeros((imgcy.shape[0], imgcy.shape[1], 3))
                    for i in range(imgct.shape[0]):
                        for j in range(imgct.shape[1]):
                            imgt[i, j] = palette[3 * imgct[i, j]:3 * imgct[i, j] + 3]
                            imgy[i, j] = palette[3 * imgcy[i, j]:3 * imgcy[i, j] + 3]
                    imgt = imgt.astype("uint8")
                    imgy = imgy.astype("uint8")
                    plt.imsave(args.save_path + names[idx] + "_target.png", imgt)
                    plt.imsave(args.save_path + names[idx] + "_pred.png", imgy)

        print("test_metrics: ", pacc / num_iter, miou / num_iter)

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--seed', type=int, default=520)
    arg_parser.add_argument('--data_path', type=str, default='F:/ImageSegment')
    arg_parser.add_argument('--model_path', type=str, default='F:\\ImageSegment\\deeplearning-segment\\exp_0106_23_09\\ckpt\\ckpt_bestEpoch1.pth')
    arg_parser.add_argument('--num_class', type=int, default=21)

    arg_parser.add_argument('--max_files', type=int, default=20)
    arg_parser.add_argument('--batch_size', '-bs', type=int, help='Batch size.', default=1)

    arg_parser.add_argument('--save_path', type=str, default='F:/ImageSegment/deeplearning-segment-2/result/exp_0106_23_09/')
    arg_parser.add_argument('--save_img', type=bool, default=True)
    arg_parser.add_argument('--resize_to', nargs='+', type=int,
                            help='Input size to resize to before feeding into neural net; (height, width)',
                            default=[224, 224])
    args = arg_parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.model_path = 'F:\\ImageSegment\\deeplearning-segment\\exp_0106_23_09\\ckpt\\ckpt_bestEpoch1.pth'
    args.save_path = 'F:\\ImageSegment\\deeplearning-segment-2\\save_path\\'
    modelTest(args)