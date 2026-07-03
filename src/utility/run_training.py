"""
run_training.py
Universal entry point for all training runs.

Usage
-----
    from src.utils.run_training import run_training
    run_training(args)

args is an argparse.Namespace.  Required fields are documented in
get_default_parser() at the bottom of this file.

Directory layout produced per run
----------------------------------
    src/train_results/{run_name}/
        config.json              - hyperparams + timing + paths
        training_graph_data.json - full loss/lr history (appended on resume)
        {run_name}.png           - training plot (overwritten every epoch)
        train_samples_ep{A}-{B}.png   - sample grid for each run segment
        weights.pt               - full checkpoint (model + optimiser + epoch)
"""

import argparse
import os
import sys
from datetime import datetime
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from src.utility.model_builders.util.vae_builder import load_pretrained_vae
from src.utility.evaluation import compute_final_eval

sys.path.append(".")

from src.utility.dataset_builders import build_dataset
from src.utility.general import (
    _get_device,
    _load_checkpoint,
    _load_graph_data,
    _run_dir,
    _save_checkpoint,
    _save_config,
    _save_graph_data,
    load_ldm_config,
    _clear_ddpm_files,
)
from src.utility.model_builders.model_builder import build_model
from src.utility.plotting import (
    plot_final_samples,  # noqa: F401
    plot_forward_trajectory_progression,
    plot_fphi_progression,
    plot_fphi_weight_histograms,
    plot_reconstruction_diffusion_progression,  # noqa: F401
    plot_reconstruction_progression,
    plot_sample_progression,
    plot_training,
    plot_val_elbo_progression,  # noqa: F401
    plot_weight_profile_progression,  # noqa: F401
)
from src.utility.training import train, train_ddpm, train_vae


# =============================================================================
# Public API
# =============================================================================


def run_training(
    args: argparse.Namespace,
) -> nn.Module:
    """
    Full training pipeline: data → model → train → save.

    Parameters
    ----------
    args       : argparse.Namespace (see get_default_parser()).
    sample_fn  : Optional callable(model, step, device) -> None.
                 Called at 5 checkpoints during training and once at the end.

    Returns
    -------
    Trained model on CPU.
    """
    start_time = datetime.now()
    run_dir = _run_dir(args.run_name)
    device = _get_device()
    resume_path = getattr(args, "resume", None)
    use_modulation = args.use_modulation

    print("\n" + "=" * 60)
    print(f"  Run     : {args.run_name}")
    print(f"  Model   : {args.model}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Dir     : {run_dir}")
    print(f"  use_modulation: {use_modulation}")
    print("=" * 60)

    if resume_path is None:
        import glob

        # Clear stale plots and metadata from a previous run with the same name
        for fname in [
            "tqdm.log",
            "denoising_trajectory_progression.png",
            "Forward_noising_progression_*.png",
            "Reverse_denoising_progression_ep*.png",
            "fphi_weight_histogram_ep*.png",
            "metadata/fphi_weight_histogram_*.json",
            "metadata/fphi_weight_histogram_*.npy",
            "training_graph.png",
            "final_samples_ep*.png",
            "sample_progression_ep*.png",
            "fphi_progression_ep*.png",
            "ztrans_histogram_ep*.png",
            "reconstruction_progression_ep*.png",
            "reconstruction_norm_progression_ep*.png",
            "reconstruction_diffusion_progression_ep*.png",
            "weight_profile_progression_ep*.png",
            "weight_distribution_progression_ep*.png",
            "metadata/training_graph_data.json",
            "metadata/sample_progression_*.json",
            "metadata/sample_progression_*.npy",
            "metadata/reconstruction_progression_*.json",
            "metadata/reconstruction_progression_*.npy",
            "metadata/reconstruction_norm_progression_*.json",
            "metadata/reconstruction_norm_progression_*.npy",
            "metadata/reconstruction_diffusion_progression_*.json",
            "metadata/reconstruction_diffusion_progression_*.npy",
            "metadata/weight_profile_progression_*.json",
            "metadata/weight_profile_progression_*.npy",
            "metadata/weight_distribution_progression_*.json",
            "metadata/weight_distribution_progression_*.npy",
            "metadata/denoising_trajectory_progression_*.json",
            "metadata/denoising_trajectory_progression_*.npy",
            "metadata/Forward_noising_progression_*.json",
            "metadata/Forward_noising_progression_*.npy",
            "metadata/Reverse_denoising_progression_*.json",
            "metadata/Reverse_denoising_progression_*.npy",
            "metadata/fphi_progression_*.json",
            "metadata/fphi_progression_*.npy",
            "metadata/ztrans_histogram_*.json",
            "metadata/ztrans_histogram_*.npy",
            "weights/weights.pt",
            "metadata/config.json",
            "val_elbo_progression.png",
            "metadata/val_elbo_progression_*.json",
        ]:
            for fpath in glob.glob(os.path.join(run_dir, fname)):
                os.remove(fpath)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    print("\n[ 1 / 4 ]  Building dataset …")
    train_dataset, val_dataset, data_config = build_dataset(
        dataset_name=args.dataset,
        data_root=getattr(args, "data_root", "data/"),
        subset_frac=getattr(args, "subset_frac", 1.0),
        single_class=getattr(args, "single_class", False),
        single_class_label=getattr(args, "single_class_label", 1),
    )
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=getattr(args, "num_workers", 0),
    )
    data_loader_val = torch.utils.data.DataLoader(  # noqa: F841
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=getattr(args, "num_workers", 0),
    )
    print(f"  Batches per epoch : {len(data_loader_train)}")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("\n[ 2 / 4 ]  Building model …")
    model = build_model(args, data_config).to(device)

    # ── 3. Optimiser & optional resume ───────────────────────────────────────
    print("\n[ 3 / 4 ]  Setting up optimiser …")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.0)
    )

    start_epoch = 0
    if resume_path is not None:
        print(f"  Resuming from checkpoint: {resume_path}")
        start_epoch = _load_checkpoint(resume_path, model, optimizer)
    else:
        print("  Training from scratch.")

    # Determine sample filename for this run segment
    end_epoch = start_epoch + args.epochs

    progression_filename = f"sample_progression_ep{start_epoch + 1}-{end_epoch}"

    def _sample_fn(model, step, device, batch=None):
        epoch = step // len(data_loader_train)
        
        plot_sample_progression(
            model,
            args.model,
            epoch,
            run_dir,
            device,
            data_config,
            filename=progression_filename,
            collect_snapshots=True,
        )
        
        if batch is not None:
            if args.model in ("ndm"):
                plot_fphi_progression(
                    model,
                    batch,
                    epoch,
                    run_dir,
                    device,
                    data_config,
                    filename=f"fphi_progression_ep{start_epoch + 1}-{end_epoch}",
                    model_name=args.model,
                )
            elif args.model in ("latent_inr_diffusion"):
                
                plot_reconstruction_progression(
                    model,
                    batch,
                    epoch,
                    run_dir,
                    device,
                    data_config,
                    filename=f"reconstruction_progression_ep{start_epoch + 1}-{end_epoch}",
                    model_name=args.model,
                )
                plot_forward_trajectory_progression(
                    model=model,
                    batch=batch,
                    epoch=epoch,
                    run_dir=run_dir,
                    device=device,
                    data_config=data_config,
                    filename=f"Forward_noising_progression_ep{start_epoch + 1}-{end_epoch}",
                    model_name=args.model,
                    normalize=args.normalize,
                )
                """
                plot_val_elbo_progression(
                    model=model,
                    data_loader_val=data_loader_val,
                    epoch=epoch,
                    run_dir=run_dir,
                    filename="val_elbo_progression",
                )
                """
            elif args.model == "weight_inr_ndm_diffusion":
                plot_reconstruction_progression(
                    model,
                    batch,
                    epoch,
                    run_dir,
                    device,
                    data_config,
                    filename=f"reconstruction_progression_ep{start_epoch + 1}-{end_epoch}",
                    model_name=args.model,
                )
                plot_forward_trajectory_progression(
                    model=model,
                    batch=batch,
                    epoch=epoch,
                    run_dir=run_dir,
                    device=device,
                    data_config=data_config,
                    filename=f"Forward_noising_progression_ep{start_epoch + 1}-{end_epoch}",
                    model_name=args.model,
                    normalize=args.normalize,
                )
                plot_fphi_weight_histograms(
                    model=model,
                    batch=batch,
                    epoch=epoch,
                    run_dir=run_dir,
                    device=device,
                    data_config=data_config,
                    filename=f"fphi_weight_histogram_ep{start_epoch + 1}-{end_epoch}",
                    model_name=args.model,
                )
            elif args.model in (
                "ndm_inr",
                "ndm_transinr",
                "ndm_static_mlpinr",
                "ndm_temporal_transinr",
                "weight_inr_diffusion",
            ):
                plot_reconstruction_progression(
                    model,
                    batch,
                    epoch,
                    run_dir,
                    device,
                    data_config,
                    filename=f"reconstruction_progression_ep{start_epoch + 1}-{end_epoch}",
                    model_name=args.model,
                )
                """
                plot_val_elbo_progression(
                    model=model,
                    data_loader_val=data_loader_val,
                    epoch=epoch,
                    run_dir=run_dir,
                    filename="val_elbo_progression",
                )
                """
                plot_forward_trajectory_progression(
                    model=model,
                    batch=batch,
                    epoch=epoch,
                    run_dir=run_dir,
                    device=device,
                    data_config=data_config,
                    filename=f"Forward_noising_progression_ep{start_epoch + 1}-{end_epoch}",
                    model_name=args.model,
                    normalize=args.normalize,
                )

    # Load existing history for resumed runs; fresh dict otherwise
    history = _load_graph_data(run_dir)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("\n[ 4 / 4 ]  Training …\n")
    # Patch train() to save graph + graph_data after every epoch by wrapping
    # the epoch_callback hook via the existing history dict reference.
    model = train(
        model=model,
        model_type=args.model,
        data_loader=data_loader_train,
        epochs=args.epochs,
        device=device,
        name=args.run_name,
        lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 0.0),
        grad_clip=getattr(args, "grad_clip", 1.0),
        use_scheduler=getattr(args, "use_scheduler", True),
        warmup_steps=getattr(args, "warmup_steps", 5_000),
        peak_lr=getattr(args, "peak_lr", args.lr),
        log_every_n_steps=getattr(args, "log_every_n_steps", 20),
        save_dir=run_dir,
        sample_fn=_sample_fn,
        start_epoch=start_epoch,
        # Pass existing history so train() appends to it
        history=history,
        # Epoch callback: save graph data + redraw plot after every epoch
        epoch_callback=lambda h: (
            _save_graph_data(h, run_dir),
            plot_training(
                h,
                name=args.run_name,
                graph_dir=run_dir,
                use_scheduler=getattr(args, "use_scheduler", True),
            ),
        ),
        data_config=data_config,
    )

    print("\n  Training complete...")
    print("  Generating final sample grid …")
    #plot_final_samples(
        #model, args.model, end_epoch, run_dir, device, data_config, n_fid_samples=args.n_fid_samples, val_loader=data_loader_val
    #)
    print("Final sample grid saved to training directory.")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    end_time = datetime.now()
    metadata_path = os.path.join(run_dir, "weights")
    weights_path = _save_checkpoint(model, optimizer, end_epoch, metadata_path)
    print(f"\n  Weights saved → {weights_path}")

    _save_config(
        args=args,
        data_config=data_config,
        run_dir=run_dir,
        weights_path=weights_path,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        start_time=start_time,
        end_time=end_time,
    )

    duration = end_time - start_time
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n  Done.  Total time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
    print("=" * 60 + "\n")

    return model.cpu()


def run_two_stage_training(args: argparse.Namespace) -> None:
    """
    Orchestrate two-stage training: VAE → DDPM, then final eval.

    Args:
        args (argparse.Namespace): parsed CLI arguments
    Returns:
        None
    """
    if args.skip_vae and not args.vae_weights:
        raise ValueError("--vae_weights must be provided when --skip_vae is set.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Two-Stage LDM Training: {args.run_name} | mode={args.mode} ---")

    hparams = load_ldm_config(args.ldm_config)
    print(f"Dataset: {hparams['dataset']}")

    dataset, val_dataset, data_config = build_dataset(
        dataset_name=hparams["dataset"],
        data_root="data/",
        subset_frac=args.subset_frac,
        single_class=False,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    channels = data_config["channels"]
    img_size = data_config["img_size"]
    is_3d = data_config.get("is_3d", False)

    results_dir = os.path.join(args.results_dir, args.run_name)

    if args.skip_vae:
        if not os.path.exists(results_dir):
            raise FileNotFoundError(
                f"Results directory '{results_dir}' not found. "
                "Run full training first before using --skip_vae."
            )
        print(f"  --skip_vae: clearing stale DDPM outputs from {results_dir}")
        _clear_ddpm_files(results_dir, args.run_name)
    else:
        if os.path.exists(results_dir):
            import shutil

            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

    # ── Stage 1: VAE ──────────────────────────────────────────────────────────
    if args.skip_vae:
        print("\n" + "=" * 60)
        print("  STAGE 1 — VAE TRAINING SKIPPED (loading pre-trained weights)")
        print("=" * 60)
        vae = load_pretrained_vae(
            args.vae_weights, hparams, channels, img_size, device, is_3d=is_3d
        )
        vae_epochs_done = 0
    else:
        vae = train_vae(
            args,
            hparams,
            dataloader,
            val_loader,
            channels,
            img_size,
            is_3d,
            results_dir,
            device,
        )
        vae_ckpt_path = os.path.join(results_dir, f"{args.run_name}_vae_checkpoint.pt")
        vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
        vae_epochs_done = vae_ckpt["epoch"]

    # ── Stage 2: DDPM ─────────────────────────────────────────────────────────
    ldm = train_ddpm(
        args,
        hparams,
        vae,
        dataloader,
        val_loader,
        channels,
        img_size,
        is_3d,
        data_config,
        results_dir,
        device,
        vae_epochs_done=vae_epochs_done,
    )

    ldm_ckpt_path = os.path.join(results_dir, f"{args.run_name}_ldm_checkpoint.pt")
    ldm_ckpt = torch.load(ldm_ckpt_path, map_location=device)
    ddpm_epochs_done = ldm_ckpt["epoch"]

    # ── Final eval ────────────────────────────────────────────────────────────
    compute_final_eval(
        vae=vae if not args.skip_vae else None,
        ldm=ldm,
        hparams=hparams,
        val_loader=val_loader,
        data_config=data_config,
        args=args,
        results_dir=results_dir,
        device=device,
        vae_epochs=vae_epochs_done,
        ddpm_epochs=ddpm_epochs_done,
        skip_vae_eval=args.skip_vae,
    )
