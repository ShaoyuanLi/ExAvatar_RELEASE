import argparse
from config import cfg
import torch
from base import Trainer
# import faulthandler
# faulthandler.enable()   
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--fit_pose_to_test', dest='fit_pose_to_test', action='store_true')
    parser.add_argument('--continue', dest='continue_train', action='store_true')

    args = parser.parse_args()
    assert args.subject_id, "Please set subject ID"
    return args

def main():
    args = parse_args()
    cfg.set_args(args.subject_id, args.fit_pose_to_test, args.continue_train)

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
    
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr, data in enumerate(trainer.batch_generator):
            # 如果是续训的第一个epoch，则跳过已经训练过的batch
            if epoch == trainer.start_epoch and itr < trainer.start_itr:
                continue
            # 在第一个有效batch之后，重置start_itr，确保后续epoch从0开始
            if itr == trainer.start_itr:
                trainer.start_itr = 0
        
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()
            
            # set stage
            cur_itr = epoch * len(trainer.batch_generator) + itr
            cfg.set_stage(cur_itr)

            # set learning rate
            tot_itr = cfg.end_epoch * len(trainer.batch_generator)
            trainer.set_lr(cur_itr, tot_itr)
           
            # forward
            trainer.optimizer.zero_grad()
            stats, loss = trainer.model(data, cur_itr, 'train')
            loss = {k:loss[k].mean() for k in loss}

            # backward
            sum(loss[k] for k in loss).backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=20.0)
            # densify and prune scene Gaussians
            if (not cfg.fit_pose_to_test) and (cur_itr < cfg.densify_end_itr):
                with torch.no_grad():
                    stats['mean_2d_grad'] = torch.stack([x.grad.detach() for x in stats['mean_2d']])
                    stats.pop('mean_2d', None)
                    stats = {k: v for k,v in stats.items()}
                    trainer.model.module.adjust_gaussians(stats, trainer.model.module.scene_gaussian, cur_itr, trainer.optimizer)

            # update
            trainer.optimizer.step()
            
            # log
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
            trainer.logger.info(' '.join(screen))
            print(cfg.model_dir)

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
            # --- 新增：在每个 batch 后保存模型 ---
            # 您可以根据需要调整保存频率，例如每 N 个 batch 保存一次
            if (itr + 1) % cfg.save_interval == 0: # 假设 cfg.save_interval 是保存间隔
                 trainer.save_model({
                    'epoch': epoch,
                    'itr': itr + 1, # 保存下一个应该开始的 batch 索引
                    'network': trainer.model.module.state_dict(),
                    'optimizer': trainer.optimizer.state_dict(),
                }, epoch, itr + 1)
            cur_itr += 1
            

        # save model
        trainer.save_model({
            'epoch': epoch + 1,
            'itr': 0,
            'network': trainer.model.module.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch + 1, 0)
   
if __name__ == "__main__":
    main()
