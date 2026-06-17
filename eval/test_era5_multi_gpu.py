import argparse
import csv
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

from test_era5 import (
    build_date_filter,
    normalize_predictor_value,
    parse_requested_prediction_targets,
    resolve_prediction_target_indices,
    zero_low_solar_gpp_predictions,
)
from utils import resolve_checkpoint_path, resolve_config_path

INTERNAL_ORDER_COLUMN = "__sample_order"


class IndexedDataset:
    def __init__(self, dataset, sample_start: int):
        self.dataset = dataset
        self.sample_start = sample_start

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.sample_start + idx, self.dataset[idx]


class IndexedERA5Collate:
    def __init__(self, base_collate_fn):
        self.base_collate_fn = base_collate_fn

    def __call__(self, batch):
        sample_indices, samples = zip(*batch)
        era5_batch = self.base_collate_fn(samples)
        era5_batch.sample_indices = tuple(int(index) for index in sample_indices)
        return era5_batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run distributed ERA5 inference with torchrun and save predictions to CSV."
    )
    parser.add_argument(
        "--run-path",
        type=Path,
        required=True,
        help="Run directory.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Path to config YAML (default: <run_path>/../config.yml).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help=(
            "Path to checkpoint file. Relative paths resolve from <run_path> "
            "(default: run_path/last.pth, else latest checkpoint-*.pth)."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default is <run_path>/eval/era5_predictions_multi_gpu.csv, "
            "or era5_predictions_<start>_to_<end>_multi_gpu.csv when date bounds are provided."
        ),
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-rank temporary CSV shards "
            "(default: <output_dir>/.<output_name>_shards)."
        ),
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep per-rank shard CSVs after rank 0 merges them.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help=(
            "Only write per-rank shard CSVs. Use eval/merge_era5_shards.py "
            "or scripts/post_era5.sh to merge them later."
        ),
    )
    parser.add_argument(
        "--start-date",
        "--initial-date",
        dest="start_date",
        default=None,
        help=(
            "Inclusive ERA5 start date. Accepts YYYY-MM-DD, YYYYMMDD, or full "
            "YYYYMMDDHHMMSS timestamp."
        ),
    )
    parser.add_argument(
        "--end-date",
        "--final-date",
        dest="end_date",
        default=None,
        help=(
            "Inclusive ERA5 end date. Date-only values include the whole day. "
            "Accepts YYYY-MM-DD, YYYYMMDD, or full YYYYMMDDHHMMSS timestamp."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: use entire dataset).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-GPU batch size override (default: value from config).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloader workers per GPU for inference.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="SQLite database to evaluate against.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=1,
        help="Number of batches prefetched by each dataloader worker.",
    )
    parser.add_argument(
        "--dataloader-out-of-order",
        action="store_true",
        help=(
            "Allow DataLoader workers to return ready batches out of order. This avoids "
            "head-of-line blocking when one worker is repeatedly slower than the others."
        ),
    )
    parser.add_argument(
        "--exclude-igbp",
        nargs="+",
        default=(),
        metavar="CODE",
        help="IGBP code(s) to exclude from ERA5 inference.",
    )
    parser.add_argument(
        "--prediction-targets",
        nargs="+",
        default=None,
        metavar="TARGET",
        help=(
            "Prediction target columns to include in the output CSV. Accepts "
            "comma-separated or space-separated values, with or without the "
            "pred_ prefix. Default: all model outputs."
        ),
    )
    parser.add_argument(
        "--gpp-solar-threshold",
        type=float,
        default=None,
        help=(
            "If provided, force GPP prediction columns to 0.0 when the final "
            "raw SW_IN predictor is below this threshold in W m-2. The value "
            "is converted to the normalized SW_IN units stored in the ERA5 database."
        ),
    )
    parser.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=120,
        help=(
            "Timeout for distributed collectives. Long ERA5 shard imbalance can make "
            "early ranks wait at final synchronization points."
        ),
    )
    return parser.parse_args()


def init_distributed(timeout_minutes: int):
    import torch
    import torch.distributed as dist

    if timeout_minutes <= 0:
        raise ValueError("--distributed-timeout-minutes must be a positive integer.")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) are visible."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )

    return distributed, rank, local_rank, world_size, device


def barrier(distributed: bool, device=None):
    if distributed:
        import torch.distributed as dist

        if device is not None and device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()


def default_output_csv_path(run_path: Path, date_tag: str | None) -> Path:
    filename = (
        "era5_predictions_multi_gpu.csv"
        if date_tag is None
        else f"era5_predictions_{date_tag}_multi_gpu.csv"
    )
    return run_path / "eval" / filename


def rank_sample_bounds(total: int, world_size: int, rank: int) -> tuple[int, int]:
    base = total // world_size
    remainder = total % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def distributed_sum_int(value: int, *, distributed: bool, device) -> int:
    import torch
    import torch.distributed as dist

    tensor_device = device if device.type == "cuda" else torch.device("cpu")
    total = torch.tensor([value], dtype=torch.long, device=tensor_device)
    if distributed:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return int(total.item())


def build_rank_limited_era5_dataset_class(base_dataset_class, np_module):
    class RankLimitedERA5Dataset(base_dataset_class):
        def __init__(self, *args, sample_start: int, sample_end: int, **kwargs):
            self._sample_start = sample_start
            self._sample_end = sample_end
            super().__init__(*args, **kwargs)

        def _build_sample_index(self, conn):
            if self._sample_start >= self._sample_end:
                return (
                    np_module.asarray([], dtype=np_module.int64),
                    np_module.asarray([], dtype=np_module.int64),
                    np_module.asarray([], dtype=np_module.int64),
                )

            filters = []
            params = []
            if self.end_timestamp is not None:
                filters.append("ec.timestamp <= ?")
                params.append(int(self.end_timestamp))
            if self.exclude_igbp:
                placeholders = ",".join("?" for _ in self.exclude_igbp)
                filters.append(
                    "NOT EXISTS ("
                    "SELECT 1 FROM coord_data coord "
                    "WHERE coord.coord_id = ec.coord_id "
                    f"AND coord.igbp IN ({placeholders})"
                    ")"
                )
                params.extend(self.exclude_igbp)
            where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""

            cursor = conn.execute(
                f"""
                SELECT ec.id, ec.coord_id, ec.timestamp
                FROM ec_data ec
                {where_sql}
                ORDER BY ec.coord_id, ec.timestamp, ec.id;
                """,
                params,
            )
            indexes = []
            coord_ids = []
            timestamps = []
            current_coord_id = None
            coord_count = 0
            valid_sample_count = 0

            while True:
                rows = cursor.fetchmany(self._INDEX_FETCH_SIZE)
                if not rows:
                    break
                for row_id, coord_id, timestamp in rows:
                    if coord_id != current_coord_id:
                        current_coord_id = coord_id
                        coord_count = 1
                    else:
                        coord_count += 1
                    if self.start_timestamp is not None and timestamp < self.start_timestamp:
                        continue
                    if coord_count >= self.config.context_length:
                        if valid_sample_count >= self._sample_start:
                            indexes.append(int(row_id))
                            coord_ids.append(int(coord_id))
                            timestamps.append(int(timestamp))
                        valid_sample_count += 1
                        if valid_sample_count >= self._sample_end:
                            return (
                                np_module.asarray(indexes, dtype=np_module.int64),
                                np_module.asarray(coord_ids, dtype=np_module.int64),
                                np_module.asarray(timestamps, dtype=np_module.int64),
                            )

            return (
                np_module.asarray(indexes, dtype=np_module.int64),
                np_module.asarray(coord_ids, dtype=np_module.int64),
                np_module.asarray(timestamps, dtype=np_module.int64),
            )

    return RankLimitedERA5Dataset


def prepare_shard_dir(shard_dir: Path, *, rank: int, distributed: bool, device):
    if rank == 0:
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)
    barrier(distributed, device)


def merge_csv_shards(shard_paths: list[Path], output_csv_path: Path):
    from merge_era5_shards import merge_csv_shards as merge_post_csv_shards

    merge_post_csv_shards(
        shard_paths,
        output_csv_path,
        prediction_targets=None,
        sort_output=True,
        sort_tmp_dir=None,
    )


def move_batch_to_device(batch, device):
    batch.predictor_values = batch.predictor_values.to(device, non_blocking=True)
    batch.aux_values = batch.aux_values.to(device, non_blocking=True)
    if batch.modis_values is not None:
        batch.modis_values = batch.modis_values.to(device, non_blocking=True)
    if batch.modis_present is not None:
        batch.modis_present = batch.modis_present.to(device, non_blocking=True)
    if batch.target_values is not None:
        batch.target_values = batch.target_values.to(device, non_blocking=True)
    return batch


def iter_prediction_rows(batch, preds_cpu, aux_cpu, output_target_indices):
    lat_idx = batch.aux_columns.index("lat") if "lat" in batch.aux_columns else None
    lon_idx = batch.aux_columns.index("lon") if "lon" in batch.aux_columns else None
    sample_indices = getattr(batch, "sample_indices", None)
    if sample_indices is None:
        raise RuntimeError("ERA5 batch is missing sample_indices required for stable shard ordering.")

    for i in range(preds_cpu.shape[0]):
        ts = batch.timestamps[i][-1] if len(batch.timestamps[i]) > 0 else ""
        lat_val = float(aux_cpu[i, lat_idx].item() * 180.0) if lat_idx is not None else float("nan")
        lon_val = float(aux_cpu[i, lon_idx].item() * 180.0) if lon_idx is not None else float("nan")
        row = [
            str(sample_indices[i]),
            f"{lat_val:.2f}",
            f"{lon_val:.2f}",
            batch.igbp[i],
            ts,
        ]
        for j in output_target_indices:
            row.append(f"{preds_cpu[i, j].item():.4f}")
        yield row


def main():
    args = parse_args()
    distributed, rank, local_rank, world_size, device = init_distributed(
        args.distributed_timeout_minutes
    )
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer when provided.")

    try:
        start_timestamp, end_timestamp, date_tag = build_date_filter(args)
        requested_prediction_targets = parse_requested_prediction_targets(args.prediction_targets)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    exclude_igbp = tuple(
        dict.fromkeys(code.strip().upper() for code in args.exclude_igbp if code.strip())
    )
    force_zero_gpp_low_solar = args.gpp_solar_threshold is not None
    if force_zero_gpp_low_solar and not math.isfinite(args.gpp_solar_threshold):
        raise ValueError("--gpp-solar-threshold must be finite.")
    normalized_gpp_solar_threshold = (
        normalize_predictor_value("SW_IN", args.gpp_solar_threshold)
        if force_zero_gpp_low_solar
        else None
    )

    import torch
    import torch.distributed as dist
    import numpy as np
    import yaml
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    from ecoperceiver.components import EcoPerceiverConfig
    from ecoperceiver.dataset import EcoPerceiverLoaderConfig
    from ecoperceiver.era5_dataset import ERA5Dataset
    from ecoperceiver.era5_model import ERA5EcoPerceiver

    repo_root = Path(__file__).resolve().parent.parent
    run_path = args.run_path.resolve()
    config_path = resolve_config_path(run_path, args.config_path)
    explicit_checkpoint_path = args.checkpoint_path.expanduser() if args.checkpoint_path is not None else None
    if explicit_checkpoint_path is not None and not explicit_checkpoint_path.is_absolute():
        explicit_checkpoint_path = run_path / explicit_checkpoint_path
    checkpoint_path = resolve_checkpoint_path(run_path, explicit_checkpoint_path)
    data_path = (repo_root / "experiments/data").resolve()
    db_path = args.db_path.expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    output_csv_path = (
        default_output_csv_path(run_path, date_tag)
        if args.output_csv is None
        else args.output_csv.resolve()
    )
    shard_dir = (
        args.shard_dir.expanduser().resolve()
        if args.shard_dir is not None
        else output_csv_path.parent / f".{output_csv_path.stem}_shards"
    )

    if rank == 0:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_shard_dir(shard_dir, rank=rank, distributed=distributed, device=device)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    try:
        output_target_indices, output_flux_labels = resolve_prediction_target_indices(
            requested_prediction_targets,
            config["model"]["targets"],
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_prediction_columns = [f"pred_{flux}" for flux in output_flux_labels]

    if rank == 0:
        print("Configuration loaded:")
        print(f"Model targets: {config['model']['targets']}")
        print(f"Output prediction columns: {', '.join(output_prediction_columns)}")
        print(f"Context length: {config['model']['context_length']}")
        print(f"Latent space dim: {config['model']['latent_space_dim']}")
        print(f"Run path: {run_path}")
        print(f"Config path: {config_path}")
        print(f"Checkpoint path: {checkpoint_path}")
        print(f"Data path: {data_path}")
        print(f"DB path: {db_path}")
        print(f"Torchrun world size: {world_size}")
        print(f"Output CSV: {output_csv_path}")
        print(f"Shard dir: {shard_dir}")
        if start_timestamp is not None or end_timestamp is not None:
            print(f"ERA5 date filter: {start_timestamp or 'start'} to {end_timestamp or 'end'}")
        if exclude_igbp:
            print(f"Excluded IGBP classes: {', '.join(exclude_igbp)}")
        if force_zero_gpp_low_solar:
            print(
                "Low-solar GPP override: force GPP* predictions to 0 when "
                f"raw final SW_IN < {args.gpp_solar_threshold:g} W m-2 "
                f"(normalized SW_IN < {normalized_gpp_solar_threshold:.6g})"
            )

    print(
        f"[rank {rank}/{world_size}] local_rank={local_rank} device={device} "
        f"visible_cuda={torch.cuda.device_count()}",
        flush=True,
    )

    model_config = EcoPerceiverConfig(**config["model"])
    relative_pretrained_path = repo_root / "ecoperceiver" / "resnet18_weights.pth"
    model = ERA5EcoPerceiver(model_config, relative_pretrained_path)
    if rank == 0:
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    print(f"[rank {rank}] Model loaded from epoch {checkpoint['epoch']} on {device}", flush=True)

    dataset_config = EcoPerceiverLoaderConfig(**config["dataset"])
    if args.max_samples is None:
        dataset = ERA5Dataset(
            data_path,
            config=dataset_config,
            sql_file=db_path,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            exclude_igbp=exclude_igbp,
        )
        max_samples = len(dataset)
        rank_start, rank_end = rank_sample_bounds(max_samples, world_size, rank)
        rank_dataset = Subset(dataset, range(rank_start, rank_end))
        total_indexed_samples = max_samples
    else:
        max_samples = args.max_samples
        rank_start, rank_end = rank_sample_bounds(max_samples, world_size, rank)
        rank_limited_dataset_class = build_rank_limited_era5_dataset_class(ERA5Dataset, np)
        dataset = rank_limited_dataset_class(
            data_path,
            config=dataset_config,
            sql_file=db_path,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            exclude_igbp=exclude_igbp,
            sample_start=rank_start,
            sample_end=rank_end,
        )
        rank_dataset = dataset
        total_indexed_samples = distributed_sum_int(
            len(rank_dataset),
            distributed=distributed,
            device=device,
        )

    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer when provided.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be >= 1.")
    dataloader_batch_size = args.batch_size if args.batch_size is not None else config["dataloader"]["batch_size"]
    dataloader_in_order = not args.dataloader_out_of_order
    indexed_rank_dataset = IndexedDataset(rank_dataset, sample_start=rank_start)
    dataloader_kwargs = dict(
        dataset=indexed_rank_dataset,
        batch_size=dataloader_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=config["dataloader"]["pin_memory"],
        collate_fn=IndexedERA5Collate(dataset.collate_fn),
    )
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
        dataloader_kwargs["in_order"] = dataloader_in_order
    dataloader = DataLoader(**dataloader_kwargs)

    print(
        f"[rank {rank}] Dataset samples: indexed_total={total_indexed_samples} requested={max_samples} "
        f"rank_range=[{rank_start}, {rank_end}) rank_samples={len(rank_dataset)} "
        f"batch_size={dataloader_batch_size} num_workers={args.num_workers} "
        f"prefetch_factor={args.prefetch_factor if args.num_workers > 0 else 'n/a'} "
        f"in_order={dataloader_in_order if args.num_workers > 0 else 'n/a'} "
        f"shard_order_key={INTERNAL_ORDER_COLUMN}",
        flush=True,
    )

    shard_path = shard_dir / f"rank_{rank:05d}.csv"
    fieldnames = [INTERNAL_ORDER_COLUMN, "lat", "lon", "igbp", "timestamp"] + output_prediction_columns
    rows_written = 0
    batches_processed = 0
    low_solar_gpp_rows = 0

    with shard_path.open("w", newline="", encoding="utf-8", buffering=1024 * 1024) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(fieldnames)
        with torch.inference_mode():
            for batch in tqdm(
                dataloader,
                desc=f"Rank {rank} inference",
                unit="batch",
                disable=rank != 0,
            ):
                batches_processed += 1
                batch = move_batch_to_device(batch, device)
                res = model(batch)
                yhat = res.predictions
                if force_zero_gpp_low_solar:
                    yhat, low_solar_count = zero_low_solar_gpp_predictions(
                        yhat,
                        batch,
                        res.flux_labels,
                        normalized_gpp_solar_threshold,
                    )
                    low_solar_gpp_rows += low_solar_count

                if batches_processed == 1:
                    print(f"[rank {rank}] Pred shape per batch: {yhat.shape}", flush=True)

                preds_cpu = yhat.detach().cpu()
                aux_cpu = batch.aux_values.detach().cpu()
                writer.writerows(
                    iter_prediction_rows(batch, preds_cpu, aux_cpu, output_target_indices)
                )
                rows_written += preds_cpu.shape[0]

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print(
        f"[rank {rank}] Wrote {rows_written} rows across {batches_processed} batches to {shard_path}",
        flush=True,
    )

    if args.skip_merge:
        if rank == 0:
            print(f"Skipped final merge. Rank shards are written under: {shard_dir}")
            print(f"Merge later to: {output_csv_path}")
        if distributed:
            dist.destroy_process_group()
        return

    stats_device = device if device.type == "cuda" else torch.device("cpu")
    stats = torch.tensor(
        [rows_written, batches_processed, low_solar_gpp_rows],
        dtype=torch.long,
        device=stats_device,
    )
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_rows_written, total_batches_processed, total_low_solar_gpp_rows = (
        int(stats[0].item()),
        int(stats[1].item()),
        int(stats[2].item()),
    )

    barrier(distributed, device)
    if rank == 0:
        shard_paths = [shard_dir / f"rank_{shard_rank:05d}.csv" for shard_rank in range(world_size)]
        merge_csv_shards(shard_paths, output_csv_path)
        print(
            f"Saved predictions for {total_rows_written} samples across "
            f"{total_batches_processed} batches to: {output_csv_path}"
        )
        if force_zero_gpp_low_solar:
            print(
                f"Forced GPP=0 for {total_low_solar_gpp_rows} samples with "
                f"raw final SW_IN < {args.gpp_solar_threshold:g} W m-2 "
                f"(normalized SW_IN < {normalized_gpp_solar_threshold:.6g})"
            )
        if not args.skip_merge and not args.keep_shards:
            shutil.rmtree(shard_dir)

    barrier(distributed, device)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
