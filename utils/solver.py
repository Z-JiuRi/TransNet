import time
import os
import torch
from collections import namedtuple
from torch.utils.tensorboard.writer import SummaryWriter
from utils import logger
from utils.statics import AverageMeter, evaluator_ratio

__all__ = ['Trainer', 'Tester']


field = ('nmse', 'epoch')
Result = namedtuple('Result', field, defaults=(None,) * len(field))

class Trainer:
    """ The training pipeline for encoder-decoder architecture
    """

    def __init__(self, model, device, optimizer, criterion, scheduler, resume=None,
                 save_path='./checkpoints', tensorboard_dir=None, print_freq=20,
                 val_freq=10, test_freq=10):

        # Basic arguments
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        # Verbose arguments
        self.resume_file = resume
        self.save_path = save_path
        self.tensorboard_dir = tensorboard_dir
        self.print_freq = print_freq
        self.val_freq = val_freq
        self.test_freq = test_freq

        # Pipeline arguments
        self.cur_epoch = 1
        self.all_epoch = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.best_nmse = Result()
        self.adapter_initial_state = None

        self.tester = Tester(model, device, criterion, print_freq)
        self.test_loader = None
        if self.tensorboard_dir is None:
            self.tensorboard_dir = os.path.join("exps", "default", "tensorboard")
        self.vision = SummaryWriter(log_dir=self.tensorboard_dir)

    def loop(self, epochs, train_loader, val_loader, test_loader):
        """ The main loop function which runs training and validation iteratively.

        Args:
            epochs (int): The total epoch for training
            train_loader (DataLoader): Data loader for training data.
            val_loader (DataLoader): Data loader for validation data.
            test_loader (DataLoader): Data loader for test data.
        """

        self.all_epoch = epochs
        self._resume()
        adapter_monitor_enabled = self._has_adapter_params()
        if adapter_monitor_enabled:
            self.adapter_initial_state = self._capture_adapter_state()
            self.test_loss, nmse = self.test(test_loader)
            self.vision.add_scalar("adapter_monitor/test_loss", self.test_loss,
                                   global_step=self.cur_epoch - 1)
            self.vision.add_scalar("adapter_monitor/nmse", nmse,
                                   global_step=self.cur_epoch - 1)
            self._log_adapter_monitor("before_train", nmse=nmse,
                                      test_loss=self.test_loss)

        for ep in range(self.cur_epoch, epochs + 1):
            self.cur_epoch = ep

            # conduct training, validation and test
            self.train_loss = self.train(train_loader)
            self.val_loss = None
            if ep % self.val_freq == 0:
                self.val_loss = self.val(val_loader)

            if adapter_monitor_enabled or ep % self.test_freq == 0:
                self.test_loss, nmse = self.test(test_loader)
                self.vision.add_scalar("test/loss", self.test_loss, global_step=ep)
                self.vision.add_scalar("test/nmse", nmse, global_step=ep)
                self.vision.add_scalar("test/train_loss", self.train_loss, global_step=ep)
            else:
                nmse = None

            if adapter_monitor_enabled:
                self._log_adapter_monitor(f"epoch={ep}", nmse=nmse,
                                          train_loss=self.train_loss,
                                          val_loss=self.val_loss,
                                          test_loss=self.test_loss)

            # conduct saving, visualization and log printing
            self._loop_postprocessing(nmse)

    def train(self, train_loader):
        """ train the model on the given data loader for one epoch.

        Args:
            train_loader (DataLoader): the training data loader
        """

        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader)

    def val(self, val_loader):
        """ exam the model with validation set.

        Args:
            val_loader: (DataLoader): the validation data loader
        """

        self.model.eval()
        with torch.no_grad():
            return self._iteration(val_loader)

    def test(self, test_loader):
        """ Truly test the model on the test dataset for one epoch.

        Args:
            test_loader (DataLoader): the test data loader
        """

        self.model.eval()
        with torch.no_grad():
            return self.tester(test_loader, verbose=False)

    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, (sparse_gt, ) in enumerate(data_loader):
            sparse_gt = sparse_gt.to(self.device, dtype=torch.float32)
            sparse_pred = self.model(sparse_gt)
            loss = self.criterion(sparse_pred, sparse_gt)

            # Scheduler update, backward pass and optimization
            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            # Log and visdom update
            iter_loss.update(loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                            f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'lr: {self.scheduler.get_lr()[0]:.2e} | '
                            f'MSE loss: {iter_loss.avg:.4e} | '
                            f'time: {iter_time.avg:.3f}')
                self.vision.add_scalar("every/lr", self.scheduler.get_lr()[0],
                                       global_step=self.cur_epoch)
                self.vision.add_scalar("every/mse_loss", iter_loss.avg, self.cur_epoch)

        mode = 'Train' if self.model.training else 'Val'
        logger.info(f'=> {mode}  Loss: {iter_loss.avg:.4e}\n')

        return iter_loss.avg

    def _save(self, state, name):
        if self.save_path is None:
            logger.warning('No path to save checkpoints.')
            return

        os.makedirs(self.save_path, exist_ok=True)
        torch.save(state, os.path.join(self.save_path, name))

    def _resume(self):
        """ protected function which resume from checkpoint at the beginning of training.
        """

        if self.resume_file is None:
            return None
        assert os.path.isfile(self.resume_file)
        logger.info(f'=> loading checkpoint {self.resume_file}')
        checkpoint = torch.load(self.resume_file)
        self.cur_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_nmse = checkpoint.get('best_nmse', Result())
        self.cur_epoch += 1  # start from the next epoch

        logger.info(f'=> successfully loaded checkpoint {self.resume_file} '
                    f'from epoch {checkpoint["epoch"]}.\n')

    def _loop_postprocessing(self, nmse):
        """ private function which makes loop() function neater.
        """

        # save state generate
        state = {
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_nmse': self.best_nmse
        }

        # save model with best nmse
        if nmse is not None:
            if self.best_nmse.nmse is None or self.best_nmse.nmse > nmse:
                self.best_nmse = Result(nmse=nmse, epoch=self.cur_epoch)
                state['best_nmse'] = self.best_nmse
                self._save(state, name=f"best_nmse.pth")

        # self._save(state, name='last.pth')

        # print current best results
        if self.best_nmse.nmse is not None:
            logger.info(f'\n=! Best NMSE: {self.best_nmse.nmse:.4e} ('
                        f'epoch={self.best_nmse.epoch})\n')
            self.vision.add_scalar("best/mse", self.best_nmse.nmse,
                                   global_step=self.best_nmse.epoch)

    def save_encoder_outputs(self, data_loader, output_path):
        if output_path is None:
            logger.warning('No path to save encoder outputs.')
            return
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        encoder_outputs = []
        with torch.no_grad():
            for batch_idx, (sparse_gt, ) in enumerate(data_loader):
                sparse_gt = sparse_gt.to(self.device, dtype=torch.float32)
                encoder_output = self.model.encode(sparse_gt)
                encoder_outputs.append(encoder_output.cpu())

        encoder_outputs_tensor = torch.cat(encoder_outputs, dim=0)
        torch.save(encoder_outputs_tensor, output_path)
        logger.info(f'=> Saved encoder outputs to {output_path}')

    def _adapter_named_parameters(self):
        return [
            (name, param) for name, param in self.model.named_parameters()
            if "lora_" in name or "adapter_" in name
        ]

    def _has_adapter_params(self):
        return len(self._adapter_named_parameters()) > 0

    def _capture_adapter_state(self):
        return {
            name: param.detach().cpu().clone()
            for name, param in self._adapter_named_parameters()
        }

    def _adapter_param_stats(self):
        adapter_l2_sq = 0.0
        adapter_abs_max = 0.0
        adapter_delta_l2_sq = 0.0
        adapter_initial_l2_sq = 0.0
        trainable_scalars = 0
        non_adapter_trainable_scalars = 0
        total_scalars = 0
        group_l2_sq = {}
        component_l2_sq = {}

        for name, param in self.model.named_parameters():
            if "lora_" not in name and "adapter_" not in name and param.requires_grad:
                non_adapter_trainable_scalars += param.numel()

        for name, param in self._adapter_named_parameters():
            data = param.detach()
            total_scalars += data.numel()
            if param.requires_grad:
                trainable_scalars += data.numel()

            adapter_l2_sq += data.square().sum().item()
            adapter_abs_max = max(adapter_abs_max, data.abs().max().item())

            if "lora_A" in name:
                group = "lora_A"
            elif "lora_B" in name:
                group = "lora_B"
            else:
                group = name.rsplit(".", 1)[-1]
            group_l2_sq[group] = group_l2_sq.get(group, 0.0) + data.square().sum().item()
            component = name.split(".", 1)[0]
            component_l2_sq[component] = (
                component_l2_sq.get(component, 0.0) + data.square().sum().item()
            )

            if self.adapter_initial_state is not None and name in self.adapter_initial_state:
                initial = self.adapter_initial_state[name].to(data.device)
                adapter_delta_l2_sq += (data - initial).square().sum().item()
                adapter_initial_l2_sq += initial.square().sum().item()

        delta_l2 = adapter_delta_l2_sq ** 0.5
        initial_l2 = adapter_initial_l2_sq ** 0.5
        delta_rel = delta_l2 / initial_l2 if initial_l2 > 0 else 0.0
        return {
            "trainable_scalars": trainable_scalars,
            "non_adapter_trainable_scalars": non_adapter_trainable_scalars,
            "total_scalars": total_scalars,
            "adapter_l2": adapter_l2_sq ** 0.5,
            "lora_A_l2": group_l2_sq.get("lora_A", 0.0) ** 0.5,
            "lora_B_l2": group_l2_sq.get("lora_B", 0.0) ** 0.5,
            "group_l2": {
                group: value ** 0.5
                for group, value in sorted(group_l2_sq.items())
            },
            "component_l2": {
                component: value ** 0.5
                for component, value in sorted(component_l2_sq.items())
            },
            "adapter_abs_max": adapter_abs_max,
            "delta_l2": delta_l2,
            "delta_rel": delta_rel,
        }

    def _log_adapter_monitor(self, label, nmse=None, train_loss=None,
                             val_loss=None, test_loss=None):
        stats = self._adapter_param_stats()
        parts = [
            f"=> Adapter monitor [{label}]",
            f"trainable={stats['trainable_scalars']}/{stats['total_scalars']}",
            f"non_adapter_trainable={stats['non_adapter_trainable_scalars']}",
            f"adapter_l2={stats['adapter_l2']:.6e}",
            f"lora_A_l2={stats['lora_A_l2']:.6e}",
            f"lora_B_l2={stats['lora_B_l2']:.6e}",
            "group_l2=" + ",".join(
                f"{name}:{value:.6e}"
                for name, value in stats["group_l2"].items()
            ),
            "component_l2=" + ",".join(
                f"{name}:{value:.6e}"
                for name, value in stats["component_l2"].items()
            ),
            f"adapter_abs_max={stats['adapter_abs_max']:.6e}",
            f"delta_l2={stats['delta_l2']:.6e}",
            f"delta_rel={stats['delta_rel']:.6e}",
        ]
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.6e}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.6e}")
        if test_loss is not None:
            parts.append(f"test_loss={test_loss:.6e}")
        if nmse is not None:
            parts.append(f"nmse={float(nmse):.6e}")
        logger.info(" | ".join(parts))



class Tester:
    """ The testing interface for classification
    """

    def __init__(self, model, device, criterion, print_freq=20):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq

    def __call__(self, test_data, verbose=True):
        """ Runs the testing procedure.

        Args:
            test_data (DataLoader): Data loader for validation data.
        """

        self.model.eval()
        with torch.no_grad():
            loss, nmse = self._iteration(test_data)
        if verbose:
            logger.info(f'\n=> Test result: \nloss: {loss:.4e}'
                        f'    NMSE: {nmse:.4e}\n')
        return loss, nmse

    def _iteration(self, data_loader):
        """ protected function which test the model on given data loader for one epoch.
        """

        iter_nmse = AverageMeter('Iter nmse')
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        nmse_ratio_sum = 0.0
        nmse_ratio_count = 0
        time_tmp = time.time()

        for batch_idx, (sparse_gt, ) in enumerate(data_loader):
            sparse_gt = sparse_gt.to(self.device, dtype=torch.float32)
            sparse_pred = self.model(sparse_gt)
            loss = self.criterion(sparse_pred, sparse_gt)
            nmse_ratio = evaluator_ratio(sparse_pred, sparse_gt)
            nmse_ratio_sum += nmse_ratio.sum().item()
            nmse_ratio_count += nmse_ratio.numel()
            nmse = 10 * torch.log10(
                torch.tensor(nmse_ratio_sum / nmse_ratio_count,
                             device=self.device)
            )

            # Log and visdom update
            iter_loss.update(loss, sparse_gt.size(0))
            iter_nmse.update(nmse)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'loss: {iter_loss.avg:.4e} | '
                            f'NMSE: {iter_nmse.avg:.4e} | time: {iter_time.avg:.3f}')

        final_nmse = 10 * torch.log10(
            torch.tensor(nmse_ratio_sum / nmse_ratio_count,
                         device=self.device)
        )
        logger.info(f'=> Test NMSE: {final_nmse:.4e}\n')


        return iter_loss.avg, final_nmse
