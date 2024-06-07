import copy
import functools
import os
from pathlib import Path

import numpy as np
import blobfile as bf
import torch as th
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

import cg_diffusion.unet as unet
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,

        # required args for diffusion sampling
        classifier = None,
        class_cond = None,
        image_size = None,
        num_classes = None,
        diffusion_sample_interval = 0,

        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,

        # optional args for diffusion sampling
        specific_classes_to_sample=list(),
        classifier_scale=1.0,
        num_diffusion_samples=16,
        use_ddim=False,
        clip_denoised=True
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.diffusion_sample_interval = diffusion_sample_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        # args for diffusion sampling
        self.classifier=classifier
        self.image_size=image_size
        assert class_cond is None or not class_cond or num_classes>1
        self.class_cond=class_cond
        self.num_classes=num_classes
        self.diffusion_sample_interval=diffusion_sample_interval
        self.num_diffusion_samples=num_diffusion_samples
        self.use_ddim=use_ddim
        self.clip_denoised=clip_denoised
        self.specific_classes_to_sample=specific_classes_to_sample
        self.classifier_scale=classifier_scale

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )
            logger.log("model loading complete")

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            # if dist.get_rank() == 0:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.diffusion_sample_interval > 0 and self.step % self.diffusion_sample_interval == 0:
                self.diffusion_sample()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.model.train()
        self.ddp_model.train()
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def diffusion_sample(self):
        self.model.eval()
        self.ddp_model.eval()
        self.classifier.eval()
        if isinstance(self.model, unet.OmicsUNetModel):
            bs = 1
            n_genomics_genes = self.model.num_genomics_genes
            genomics_genes = th.from_numpy(
                np.tile(
                    np.arange(n_genomics_genes).reshape(1,n_genomics_genes),
                    (bs,1)
                )
            )
            genomics_mutation = th.randint(0, 2, (bs, n_genomics_genes))

            n_transcriptomics_genes = self.model.num_transcriptomics_genes
            transcriptomics_genes = th.from_numpy(
                np.tile(
                    np.arange(n_transcriptomics_genes).reshape(1,n_transcriptomics_genes),(bs,1)
                )
            )
            transcriptomics_exp = th.rand(bs, n_transcriptomics_genes)
            model_kwargs = {
                "genomics_genes": genomics_genes,
                "genomics_mutation": genomics_mutation,
                "transcriptomics_genes": transcriptomics_genes,
                "transcriptomics_exp": transcriptomics_exp,
            }
        else:
            bs=1
            model_kwargs = dict()

        arr, label_arr = diffusion_sampling(
            model=self.model,
            diffusion=self.diffusion,
            classifier=self.classifier,
            num_diffusion_samples=self.num_diffusion_samples,
            batch_size=bs,
            classifier_scale=self.classifier_scale,
            class_cond=self.class_cond,
            num_classes=self.num_classes,
            image_size=self.image_size,
            specific_classes_to_sample=self.specific_classes_to_sample,
            use_ddim=self.use_ddim,
            clip_denoised=self.clip_denoised,
            enable_logging=True,
            model_kwargs=model_kwargs
        )

        if dist.get_rank() == 0:
            logger.logkv("step",self.step + self.resume_step)
            logger.add_images("diffusion_samples",arr.transpose([0,3,1,2]))
            logger.dumpimages()

        dist.barrier()
        logger.log("sampling complete")

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    if os.environ.get("AUTO_RESUME","False")=="True":
        log_dir=logger.get_dir()
        if not os.path.exists(log_dir):
            return None
        files=sorted(str(fp) for fp in Path(log_dir).glob("model*.pt"))
        if len(files)==0:
            return None
        latest_ckpt=files[-1]
        logger.log("automatically resuming from checkpoint",latest_ckpt)
        return latest_ckpt
    else:
        return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def diffusion_sampling(
    model,
    diffusion,
    classifier,
    num_diffusion_samples,
    classifier_scale,
    class_cond,
    num_classes,
    image_size,
    specific_classes_to_sample=None,
    use_ddim=False,
    clip_denoised=False,
    enable_logging=True,
    batch_size=1,
    model_kwargs={},
):
    def cond_fn(x, t, y=None, **kwargs):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * classifier_scale 

    def model_fn(x, t, y=None, **kwargs):
        assert y is not None
        return model(x, t, y if class_cond else None, **kwargs) 
    
    if enable_logging:
        logger.log("diffusion sampling...")

    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size=1

    if enable_logging:
        logger.log(f"need to sample {num_diffusion_samples} images, using batch_size {batch_size}, world_size is {world_size}")

    all_images = []
    all_labels = []
    batch_index=0
    while len(all_images) * batch_size < num_diffusion_samples:
        model_kwargs_for_sampling = {}
        if 'RANK' in os.environ and "WORLD_SIZE" in os.environ:
            global_rank=int(os.environ["RANK"])
            world_size=int(os.environ["WORLD_SIZE"])
        else:
            global_rank=0
            world_size=1

        if specific_classes_to_sample is None:
            specific_classes_to_sample=list(range(num_classes))

        cl=specific_classes_to_sample[(batch_index*world_size+global_rank)%(len(specific_classes_to_sample))]
            
        classes = th.full((batch_size,),cl,device=dist_util.dev())            

        model_kwargs_for_sampling["y"] = classes
        for k,v in model_kwargs.items():
            if isinstance(v,th.Tensor):
                v=v.to(dist_util.dev())
                model_kwargs_for_sampling[k]=v
            else:
                model_kwargs_for_sampling[k]=v
                
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (batch_size, 3, image_size, image_size),
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs_for_sampling,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        if enable_logging:
            logger.log(f"created {len(all_images) * batch_size} samples, num_diffusion_samples={num_diffusion_samples}")

        batch_index+=1

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: num_diffusion_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: num_diffusion_samples]

    args=np.argsort(label_arr)
    arr=np.ascontiguousarray(arr[args])
    label_arr=np.ascontiguousarray(label_arr[args])
    return arr, label_arr

def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)

def split_microbatches_with_extra(microbatch, *args):
    def subset(arg, i, microbatch):
        if isinstance(arg, dict):
            return {k: subset(v, i, microbatch) for k, v in arg.items()}
        elif isinstance(arg, list):
            return [subset(v, i, microbatch) for v in arg]
        elif isinstance(arg, tuple):
            return tuple(subset(v, i, microbatch) for v in arg)
        elif isinstance(arg, th.Tensor):
            return arg[i : i + microbatch] if arg is not None else None
        else:
            raise ValueError(f"Unknown type: {type(arg)}")
        
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(
                subset(x, i, microbatch) if x is not None else None for x in args
            )

if __name__=="__main__":
    pass
