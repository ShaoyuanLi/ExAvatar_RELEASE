import argparse
import torch
import numpy as np
import cv2  # [新增] 导入cv2
import os   # [新增] 导入os
from config import cfg
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

# [新增函数] 封装保存逻辑
def save_checkpoint(trainer, stats, save_epoch, save_itr):
    """
    统一处理图片保存和模型保存
    Args:
        trainer: 训练器实例
        stats: 包含 human_img 的字典
        save_epoch: 当前要保存的 epoch 索引
        save_itr: 当前要保存的 itr 索引
    """
    # 1. 保存可视化图片
    if 'human_img' in stats:
        with torch.no_grad():
            img_tensor = stats['human_img']
            # 如果是 Batch 维度 [B, C, H, W]，取第一张
            if img_tensor.dim() == 4:
                img_tensor = img_tensor[0]
            # 处理图像: Detach -> CPU -> Numpy -> Transpose(H,W,C) -> RGB2BGR -> Scale -> Clip -> Uint8
            human_img = (img_tensor.detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)
            human_img = np.ascontiguousarray(human_img)
            # 绘制文字信息 (可选)
            cv2.putText(human_img, f'E{save_epoch} I{save_itr}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            # 保存图片，文件名带上 itr 防止覆盖
            file_name = f'vis_epoch_{save_epoch}_itr_{save_itr}.jpg'
            cv2.imwrite(os.path.join(cfg.model_dir, file_name), human_img)
    # 2. 保存模型 Checkpoint
    trainer.save_model({
        'epoch': save_epoch,
        'itr': save_itr,
        'network': trainer.model.module.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
    }, save_epoch, save_itr)

def main():
    args = parse_args()
    cfg.set_args(args.subject_id, args.fit_pose_to_test, args.continue_train)

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
    
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        # 初始化 stats 变量，防止作用域问题
        stats = {}
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
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=10.0)
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
            # [修复后]
            screen += ['%s: %.4f' % ('loss_' + k, v.item()) for k,v in loss.items()]

            trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

            cur_itr += 1
            
            # --- [修改] 调用封装好的函数进行保存 ---
            if (itr + 1) % cfg.save_interval == 0:
                # 保存当前 epoch 进度，itr 指向下一个 batch
                save_checkpoint(trainer, stats, epoch, itr + 1)
        # --- [修改] Epoch 结束，保存模型 ---
        # 保存为下一个 epoch 的起始 (itr=0)
        save_checkpoint(trainer, stats, epoch + 1, 0)
   
if __name__ == "__main__":
    main()
