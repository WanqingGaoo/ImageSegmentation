import argparse
import os
import time

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--eval', action='store_true',
                            help='Whether to load an existing model and evaluate it on validation images. When set to '
                                 'true, exp_dir and exp_id should point to an existing directory.')

    arg_parser.add_argument('--seed', type=int, default=520)

    arg_parser.add_argument('--base_dir', type=str, default='F:\\ImageSegment\\')
    arg_parser.add_argument('--num_class', type=int, default=21)
    arg_parser.add_argument('--exp_id', type=str, help='Experiment ID.', default='')
    arg_parser.add_argument('--exp_dir', type=str, help='Experiment save path.',default='F:\\ImageSegment\\deeplearning-segment-2\\save')
    #arg_parser.add_argument('--max_files', type=int, default=20)

    arg_parser.add_argument('--max_train_file', type=int, help='Maximum number of training images to generate '
                                                                      'predictions for.', default=None)
    arg_parser.add_argument('--max_val_file', type=int, help='Maximum number of testing images to generate '
                                                                     'predictions for.', default=None)
    
    arg_parser.add_argument('--batch_size', '-bs', type=int, help='Batch size.', default=8)
    arg_parser.add_argument('--batch_size_val', type=int, default=8)
    arg_parser.add_argument('--epochs', type=int, help='Number of epochs to train.', default=200)
    arg_parser.add_argument('--learning_rate', type=float, help='Learning rate.', default=1e-3)
    arg_parser.add_argument('--display_iter', type=int, help='Number of epochs to train.', default=10)
    arg_parser.add_argument('--snapshots_folder', type=str, default=None)
    arg_parser.add_argument('--resize_to', nargs='+', type=int,
                            help='Input size to resize to before feeding into neural net; (height, width)',
                            default=[224, 224])
    arg_parser.add_argument('--patience', type=int, default=10)
    arg_parser.add_argument('--ckpt_monitor', type=str, help='Which metric the checkpoint saver monitors.',
                            default='val_psnr')



    args = arg_parser.parse_args()
    return args


def prepare_experiment(args):
    if args.eval:
        ckpt_name = '{}_results'.format(os.path.basename(args.pretrained_model_dir))
        out_dir = os.path.join(args.exp_dir, ckpt_name)
        args.ckpt_dir = args.pretrained_model_dir
    else:
        # Create experiment directory for new experiment
        timestamp = time.strftime('%m%d_%H_%M', time.localtime())
        print('----------current time-----------', timestamp)

        if args.exp_id:
            exp_name = '{}_{}'.format(args.exp_id, timestamp)
        else:
            exp_name = '{}'.format(timestamp)
        out_dir = os.path.join(args.exp_dir, exp_name)
        os.makedirs(out_dir, exist_ok=True)

        args.ckpt_dir = os.path.join(out_dir, 'ckpt', 'ckpt_best')
        os.makedirs(args.ckpt_dir, exist_ok=True)

        args.train_hist_fn = os.path.join(out_dir, 'training_history.pickle')
        args.model_summary_fn = os.path.join(out_dir, 'model_summary.txt')
        args.time_train_fn = os.path.join(out_dir, 'time_train.txt')

    # Directories/Files for prediction outputs
    args.results_dir = os.path.join(out_dir, 'results')
    args.results_dir_val = os.path.join(args.results_dir, 'validation')
    args.results_dir_train = os.path.join(args.results_dir, 'train')
    args.results_dir_test = os.path.join(args.results_dir, 'test')
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.results_dir_val, exist_ok=True)
    os.makedirs(args.results_dir_train, exist_ok=True)
    os.makedirs(args.results_dir_test, exist_ok=True)
    args.time_pred_fn = os.path.join(args.results_dir, 'time_pred.txt')
    args.eval_fn = os.path.join(args.results_dir, 'eval.txt')

    if not args.eval:
        args_file = os.path.join(out_dir, 'args.txt')
        args_str = ''
        for key in vars(args):
            args_str += '{}: {}\n'.format(key, vars(args)[key])
        f = open(args_file, 'w')
        f.write(args_str)
        f.close()
    return args


def prepare_args():
    args = parse_args()
    args = prepare_experiment(args)
    return args

if __name__ == "__main__":
    args = prepare_args()