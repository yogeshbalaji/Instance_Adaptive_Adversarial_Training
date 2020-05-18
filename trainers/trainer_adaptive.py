import torch
import torchvision.transforms as T
import datasets
import models
import torch.optim as optim
import os.path as osp
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import time
import utils
from attacks import PGDAttacker, PGDAttackerAdaptive


class TrainerAdaptive:
    def __init__(self, args):

        self.args = args

        # Creating data loaders
        transform_train = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

        transform_test = T.Compose([
            T.ToTensor()
        ])

        kwargs = {'num_workers': 4, 'pin_memory': True}

        train_dataset = datasets.CIFAR10(args.data_root, train=True, download=True,
                                         transform=transform_train)
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        # Create model, optimizer and scheduler
        self.model = models.WRN(depth=32, width=10, num_classes=10)
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.optimizer = optim.SGD(self.model.parameters(), args.lr,
                                   momentum=0.9, weight_decay=args.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[70, 90, 100], gamma=0.2)

        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))


        self.save_path = args.save_path
        self.epoch = 0

        num_samples = len(train_dataset)
        self.epsilon_memory = torch.FloatTensor(num_samples).zero_().cuda()

        # resume from checkpoint
        ckpt_path = osp.join(args.save_path, 'checkpoint.pth')
        if osp.exists(ckpt_path):
            self._load_from_checkpoint(ckpt_path)
        elif args.restore:
            self._load_from_checkpoint(args.restore)

        cudnn.benchmark = True
        self.attacker = PGDAttackerAdaptive()
        self.attacker_test = PGDAttacker(args.attack_eps)

    def _log(self, message):
        print(message)
        f = open(osp.join(self.save_path, 'log.txt'), 'a+')
        f.write(message + '\n')
        f.close()

    def _load_from_checkpoint(self, ckpt_path):
        print('Loading model from {} ...'.format(ckpt_path))
        model_data = torch.load(ckpt_path)
        self.model.load_state_dict(model_data['model'])
        self.optimizer.load_state_dict(model_data['optimizer'])
        self.lr_scheduler.load_state_dict(model_data['lr_scheduler'])
        self.epoch = model_data['epoch'] + 1
        print('Model loaded successfully')

        eps_memory = model_data['mem']
        self.epsilon_memory = eps_memory
        self.epsilon_memory = self.epsilon_memory.cuda()

    def _save_checkpoint(self, model_name='checkpoint.pth'):
        self.model.eval()
        model_data = dict()
        model_data['model'] = self.model.state_dict()
        model_data['optimizer'] = self.optimizer.state_dict()
        model_data['lr_scheduler'] = self.lr_scheduler.state_dict()
        model_data['epoch'] = self.epoch
        model_data['mem'] = self.epsilon_memory
        torch.save(model_data, osp.join(self.save_path, model_name))

    def epsilon_select(self, input, target, indices):
        # self.model.eval()
        with torch.no_grad():
            logits = self.model(input)
            _, pred = torch.max(logits, dim=1)
            correct_preds_clean = (pred == target).float()

        if self.epoch < self.args.warmup:
            epsilon = torch.zeros(input.size(0)).fill_(self.args.attack_eps).cuda()
            epsilon = epsilon * correct_preds_clean
        else:
            epsilon_prev = self.epsilon_memory[indices]
            epsilon_low = epsilon_prev - self.args.gamma
            epsilon_cur = epsilon_prev
            epsilon_high = epsilon_prev + self.args.gamma
            attack_lr_cur = torch.clamp(epsilon_cur / (0.5 * self.args.attack_steps), min=self.args.attack_lr)
            attack_lr_high = torch.clamp(epsilon_high / (0.5 * self.args.attack_steps), min=self.args.attack_lr)

            input_cur = self.attacker.attack(input, target, self.model, self.args.attack_steps,
                                             attack_lr_cur, epsilon_cur,
                                             random_init=True, target=None)
            input_high = self.attacker.attack(input, target, self.model, self.args.attack_steps,
                                              attack_lr_high, epsilon_high,
                                              random_init=True, target=None)

            with torch.no_grad():
                logits_cur = self.model(input_cur)
                logits_high = self.model(input_high)
                _, logits_cur = torch.max(logits_cur, dim=1)
                _, logits_high = torch.max(logits_high, dim=1)

                pred_cur = (logits_cur == target).float()
                pred_high = (logits_high == target).float()

                epsilon = pred_high * epsilon_high + (1 - pred_high) * pred_cur * epsilon_cur + \
                          (1 - pred_high) * (1 - pred_cur) * epsilon_low
                epsilon = epsilon * correct_preds_clean
                epsilon = torch.clamp(epsilon, min=0)
                epsilon = epsilon * self.args.beta + epsilon_prev * (1 - self.args.beta)
        # Updating memory
        self.epsilon_memory[indices] = epsilon
        return epsilon

    def train(self):

        losses = utils.AverageMeter()

        while self.epoch < self.args.nepochs:
            self.model.train()
            correct = 0
            total = 0
            start_time = time.time()

            for i, data in enumerate(self.train_loader):
                input, target, indices = data
                target = target.cuda(non_blocking=True)
                input = input.cuda(non_blocking=True)

                if self.args.alg == 'adv_training':
                    epsilon_arr = self.epsilon_select(input, target, indices)
                    attack_lr_arr = torch.clamp(epsilon_arr / (0.5 * self.args.attack_steps), min=self.args.attack_lr)
                    input = self.attacker.attack(input, target, self.model, self.args.attack_steps,
                                                 attack_lr_arr, epsilon_arr,
                                                 random_init=True, target=None)
                    self.model.zero_grad()

                # compute output
                self.optimizer.zero_grad()
                logits = self.model(input)
                loss = F.cross_entropy(logits, target)

                loss.backward()
                self.optimizer.step()

                _, pred = torch.max(logits, dim=1)
                correct += (pred == target).sum()
                total += target.size(0)

                # measure accuracy and record loss
                losses.update(loss.data.item(), input.size(0))

            self.epoch += 1
            self.lr_scheduler.step()
            end_time = time.time()
            batch_time = end_time - start_time

            acc = (float(correct) / total) * 100
            message = 'Epoch {}, Time {}, Loss: {}, Accuracy: {}'.format(self.epoch, batch_time, loss.item(), acc)
            self._log(message)
            self._save_checkpoint()

            if self.epoch == self.args.warmup:
                self._save_checkpoint(model_name='end_of_warmup.pth')

            # Evaluation
            nat_acc = self.eval()
            adv_acc = self.eval_adversarial()
            self._log('Natural accuracy: {}'.format(nat_acc))
            self._log('Adv accuracy: {}'.format(adv_acc))

    def eval(self):
        self.model.eval()

        correct = 0
        total = 0
        for i, data in enumerate(self.val_loader):
            input, target, _ = data
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)

            # compute output
            with torch.no_grad():
                output = self.model(input)

            _, pred = torch.max(output, dim=1)
            correct += (pred == target).sum()
            total += target.size(0)

        accuracy = (float(correct) / total) * 100
        return accuracy

    def eval_adversarial(self):
        self.model.eval()

        correct = 0
        total = 0
        for i, data in enumerate(self.val_loader):
            input, target, _ = data
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)

            input = self.attacker_test.attack(input, target, self.model, self.args.attack_steps,
                                              self.args.attack_lr,
                                              random_init=True)

            # compute output
            with torch.no_grad():
                output = self.model(input)

            _, pred = torch.max(output, dim=1)
            correct += (pred == target).sum()
            total += target.size(0)

        accuracy = (float(correct) / total) * 100
        return accuracy



