import time
from train import train
from utils.arg_parser import *
from data.VOCSegCProcess import *
from utils.seed import set_random_seed
def main(args):
    set_random_seed(args.seed)
    if not args.eval:
        t0 = time.time()
        input_transform = MaskToTensor_input(args.resize_to)
        target_transform = MaskToTensor()
        train_dataset = VOC('train', args.base_dir, args.resize_to, transform=input_transform, target_transform=target_transform, max_files=args.max_train_file)
        val_dataset = VOC('val', args.base_dir, args.resize_to, transform=input_transform, target_transform=target_transform, max_files=args.max_val_file)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=args.batch_size_val, shuffle=False)
        exit()
        t_ds = time.time() - t0
        print('load dataset time', str(t_ds))

        t0 = time.time()
        train(args, train_loader, val_loader)
        t_train = time.time() - t0

        t_ds_str = "Total time preparing dataset: {:.2f} sec, {:.4f} min".format(t_ds, t_ds / 60.0)
        t_train_str = "Total time training: {:.2f} sec, {:.4f} hrs".format(t_train, t_train / 3600.0)
        t_str = '{}\n{}'.format(t_ds_str, t_train_str)

        # Save training times
        with open(args.time_train_fn, 'a+') as f:
            f.write(t_ds_str)
            f.write('\n')
            f.write(t_train_str)
            f.write('\n')
    print('------------ finish ------------')

if __name__ == '__main__':
    args = prepare_args()
    main(args)
