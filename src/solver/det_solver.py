'''
by lyuwenyu
'''
import time
import json
import datetime
import numpy as np

import torch

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import profile_model, train_one_epoch, evaluate


class DetSolver(BaseSolver):

    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        profile_module = self.ema.module if self.ema else self.model
        profile_stats = profile_model(profile_module, self.postprocessor, self.val_dataloader, self.device)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                args.clip_max_norm,
                print_freq=args.log_step,
                ema=self.ema,
                scaler=self.scaler
            )

            self.lr_scheduler.step()

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')

                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                base_ds,
                self.device,
                self.output_dir,
                profile_stats=profile_stats,
            )

            for k in test_stats.keys():
                if k in best_stat and isinstance(test_stats[k], list):
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                elif k not in best_stat and isinstance(test_stats[k], list):
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

            print('best_stat: ', best_stat)

            log_stats = {
                'epoch': epoch + 1,

                'train/giou_loss': train_stats.get('loss_giou', 0.0),
                'train/cls_loss': train_stats.get('loss_vfl', 0.0),
                'train/l1_loss': train_stats.get('loss_bbox', 0.0),

                'val/giou_loss': test_stats.get('loss_giou', 0.0),
                'val/cls_loss': test_stats.get('loss_vfl', 0.0),
                'val/l1_loss': test_stats.get('loss_bbox', 0.0),

                'metrics/precision(B)': test_stats.get('metrics/precision(B)', 0.0),
                'metrics/recall(B)': test_stats.get('metrics/recall(B)', 0.0),
                'metrics/F1(B)': test_stats.get('metrics/F1(B)', 0.0),
                'metrics/mAP50(B)': test_stats.get('metrics/mAP50(B)', 0.0),
                'metrics/mAP50-95(B)': test_stats.get('metrics/mAP50-95(B)', 0.0),
                'metrics/FPS': test_stats.get('metrics/FPS', 0.0),
                'model/FLOPs(G)': test_stats.get('model/FLOPs(G)', 0.0),
                'model/Params(M)': test_stats.get('model/Params(M)', 0.0),

                'lr': train_stats.get('lr', 0.0),
                'n_parameters': n_parameters,
            }

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        bbox_eval = coco_evaluator.coco_eval["bbox"]
                        ultralytics_prf1 = getattr(coco_evaluator, 'custom_prf1', None)

                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')

                        for name in filenames:
                            torch.save(bbox_eval.eval, self.output_dir / "eval" / name)

                        if ultralytics_prf1 is not None:
                            np.savez(
                                self.output_dir / "pr_curve.npz",
                                recall=ultralytics_prf1['curve_x'],
                                precision=ultralytics_prf1['pr_curve']
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        profile_module = self.ema.module if self.ema else self.model
        profile_stats = profile_model(profile_module, self.postprocessor, self.val_dataloader, self.device)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            base_ds,
            self.device,
            self.output_dir,
            profile_stats=profile_stats,
        )

        if self.output_dir and coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
