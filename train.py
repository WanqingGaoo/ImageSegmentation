import utils.metrics
from models import unet
import torch
import time
import torch.optim as optim
from utils.seed import set_random_seed

def train(args, train_ds, val_ds=None):
    set_random_seed(args.seed)
    patience = args.patience          # 早停等待期（连续10个epoch验证loss不下降则停止）
    epochs_wait = 0    # 当前连续等待计数
    best_valid_loss = 10000           # 最佳验证loss初始值（用于早停判断）
    early_stop = False
    best_iou_score = 0.0
    best_acc_score = 0.0
    train_epoch_loss = []
    valid_epoch_loss = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = unet.UNet(args.num_class)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)    # 余弦退火学习率调度器，周期为10个epoch
    criterion = torch.nn.CrossEntropyLoss()  # TODO Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html

    model.train()

    for epoch in range(args.epochs):
        if early_stop:
            break
        ts = time.time()
        train_loss = 0
        num_iter = 0
        for iteration, data in enumerate(train_ds):
            inp, gt = data
            # 前向传播
            pred = model(inp.to(device))
            # 损失计算
            loss = criterion(pred, gt.to(device))
            # 反向传播与优化
            optimizer.zero_grad()   # 清空梯度
            loss.backward()         # 计算梯度
            optimizer.step()        # 更新权重
            scheduler.step()        # 更新学习率  todo have error

            train_loss += loss.item()
            num_iter += 1

            if ((iteration + 1) % args.display_iter) == 0:
                print("Loss at iteration", iteration + 1, ":", loss.item())

        train_epoch_loss.append(train_loss/num_iter)
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        # ------------- val -------------
        if val_ds:
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                pacc = 0
                miou = 0
                num_iter = 0
                for iteration, (inputs, labels) in enumerate(val_ds):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    pacc += utils.metrics.pixel_acc(outputs, labels)
                    miou += utils.metrics.iou(outputs, labels).item()
                    num_iter += 1
                valid_loss_mean = valid_loss / num_iter
                miou_mean = miou / num_iter
                pacc_mean = pacc / num_iter
                valid_epoch_loss.append(valid_loss_mean)    # 记录每个epoch每batch的平均损失
                print(f"Loss at epoch: {epoch} is {valid_loss_mean}")
                print(f"IoU at epoch: {epoch} is {miou_mean}")
                print(f"Pixel acc at epoch: {epoch} is {pacc_mean}")

            # 异常检测点，早停逻辑：如果连续patience个epoch验证loss没有下降，则停止训练
            if valid_loss_mean > best_valid_loss:
                epochs_wait += 1
            else:
                best_valid_loss = valid_loss_mean
                epochs_wait = 0
            if epochs_wait == patience:
                early_stop = True

            model.train()

        # ------------- save -------------
        if miou_mean > best_iou_score:
            best_iou_score = miou_mean
            best_acc_score = pacc_mean
            torch.save(model.state_dict(), args.ckpt_dir + "/epoch" + str(epoch) + '.pth')
        # if epoch in [10, 20, 50, 100, 200, 500, 800, 999]:
        #         torch.save(model.state_dict(), args.snapshots_folder + "Epoch" + str(epoch) + '.pth')
    print('--------- finsih ---------')
    print("best_valid_metrics: ", best_acc_score, best_iou_score)


