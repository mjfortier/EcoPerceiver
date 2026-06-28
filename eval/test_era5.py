import argparse
import csv
import math
from datetime import datetime, time
from pathlib import Path
from utils import resolve_checkpoint_path, resolve_config_path, resolve_device

DATE_ONLY_FORMATS = ("%Y-%m-%d", "%Y%m%d")
DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y%m%d%H%M%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%Y%m%d%H%M",
)


def parse_era5_date_bound(value: str, *, is_end: bool) -> tuple[int, str]:
    raw_value = value.strip()
    for date_format in DATE_ONLY_FORMATS:
        try:
            parsed_date = datetime.strptime(raw_value, date_format).date()
        except ValueError:
            continue

        bound_time = time(23, 59, 59) if is_end else time(0, 0, 0)
        timestamp = datetime.combine(parsed_date, bound_time)
        return int(timestamp.strftime("%Y%m%d%H%M%S")), parsed_date.strftime("%Y%m%d")

    for date_format in DATETIME_FORMATS:
        try:
            timestamp = datetime.strptime(raw_value, date_format)
        except ValueError:
            continue

        return int(timestamp.strftime("%Y%m%d%H%M%S")), timestamp.strftime("%Y%m%d%H%M%S")

    raise ValueError(
        f"Invalid date '{value}'. Use YYYY-MM-DD, YYYYMMDD, or a full timestamp like YYYYMMDDHHMMSS."
    )


def build_date_filter(args: argparse.Namespace) -> tuple[int | None, int | None, str | None]:
    start_timestamp = None
    end_timestamp = None
    start_label = None
    end_label = None

    if args.start_date is not None:
        start_timestamp, start_label = parse_era5_date_bound(args.start_date, is_end=False)
    if args.end_date is not None:
        end_timestamp, end_label = parse_era5_date_bound(args.end_date, is_end=True)

    if start_timestamp is not None and end_timestamp is not None and end_timestamp < start_timestamp:
        raise ValueError("--end-date/--final-date must be on or after --start-date/--initial-date.")

    if start_label is None and end_label is None:
        date_tag = None
    else:
        date_tag = f"{start_label or 'start'}_to_{end_label or 'end'}"

    return start_timestamp, end_timestamp, date_tag


def default_output_csv_path(run_path: Path, date_tag: str | None) -> Path:
    filename = "era5_predictions.csv" if date_tag is None else f"era5_predictions_{date_tag}.csv"
    return run_path / "eval" / filename


def parse_requested_prediction_targets(values: list[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None

    requested = []
    for value in values:
        requested.extend(value.replace(",", " ").split())

    requested = list(dict.fromkeys(requested))
    if not requested:
        raise ValueError("--prediction-targets requires at least one target when provided.")
    return tuple(requested)


def resolve_prediction_target_indices(
    prediction_targets: tuple[str, ...] | None,
    flux_labels,
) -> tuple[list[int], list[str]]:
    flux_labels = list(flux_labels)
    if prediction_targets is None:
        return list(range(len(flux_labels))), flux_labels

    flux_to_index = {flux: idx for idx, flux in enumerate(flux_labels)}
    selected_indices = []
    selected_flux_labels = []
    invalid = []
    for prediction_target in prediction_targets:
        flux = prediction_target.removeprefix("pred_")
        if flux not in flux_to_index:
            invalid.append(prediction_target)
            continue
        selected_indices.append(flux_to_index[flux])
        selected_flux_labels.append(flux)

    if invalid:
        available = ", ".join(f"pred_{flux}" for flux in flux_labels)
        raise ValueError(
            f"Unknown --prediction-targets value(s): {', '.join(invalid)}. "
            f"Available prediction columns: {available}"
        )

    return selected_indices, selected_flux_labels


def normalize_predictor_value(predictor: str, raw_value: float) -> float:
    from ecoperceiver.constants import DEFAULT_NORM

    norm_config = DEFAULT_NORM[predictor]
    value_max = norm_config["norm_max"]
    value_min = norm_config["norm_min"]
    value_mid = (value_max + value_min) / 2
    value_range = value_max - value_min
    if norm_config["cyclic"]:
        value_range /= 2

    return (raw_value - value_mid) / value_range


def gpp_flux_indices(flux_labels) -> list[int]:
    return [
        idx
        for idx, flux in enumerate(flux_labels)
        if flux == "GPP" or flux.startswith("GPP_")
    ]


def zero_low_solar_gpp_predictions(yhat, batch, flux_labels, normalized_threshold: float):
    if "SW_IN" not in batch.predictor_columns:
        raise ValueError("--gpp-solar-threshold requires SW_IN in predictor columns.")

    gpp_indices = gpp_flux_indices(flux_labels)
    if not gpp_indices:
        raise ValueError("--gpp-solar-threshold was set, but the model has no GPP output.")

    sw_in_idx = batch.predictor_columns.index("SW_IN")
    final_sw_in = batch.predictor_values[:, -1, sw_in_idx].to(device=yhat.device)
    low_solar_mask = final_sw_in < normalized_threshold
    low_solar_count = int(low_solar_mask.sum().item())
    if low_solar_count == 0:
        return yhat, 0

    yhat = yhat.clone()
    for gpp_idx in gpp_indices:
        yhat[low_solar_mask, gpp_idx] = 0.0
    return yhat, low_solar_count


def parse_args():
    parser = argparse.ArgumentParser(description="Run ERA5 inference and save predictions to CSV.")
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
            "Output CSV path. Default is <run_path>/eval/era5_predictions.csv, "
            "or era5_predictions_<start>_to_<end>.csv when date bounds are provided."
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
        help="Optional batch size override (default: value from config).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloader workers for inference.",
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
    return parser.parse_args()


def main():
    args = parse_args()
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
    import yaml
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    from ecoperceiver.dataset import EcoPerceiverLoaderConfig
    from ecoperceiver.era5_dataset import ERA5Dataset
    from ecoperceiver.components import EcoPerceiverConfig
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
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

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

    model_config = EcoPerceiverConfig(**config["model"])
    relative_pretrained_path = repo_root / "ecoperceiver" / "resnet18_weights.pth"
    model = ERA5EcoPerceiver(model_config, relative_pretrained_path)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    device = resolve_device("auto")
    model = model.to(device)
    model.eval()
    print(f"Model moved to {device} and set to evaluation mode")

    dataset_config = EcoPerceiverLoaderConfig(**config["dataset"])
    dataset = ERA5Dataset(
        data_path,
        config=dataset_config,
        sql_file=db_path,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        exclude_igbp=exclude_igbp,
    )
    if args.max_samples is None:
        max_samples = len(dataset)
    else:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be a positive integer when provided.")
        max_samples = min(args.max_samples, len(dataset))
    dataset_for_loader = Subset(dataset, range(max_samples)) if max_samples < len(dataset) else dataset

    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer when provided.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be >= 1.")
    dataloader_batch_size = args.batch_size if args.batch_size is not None else config["dataloader"]["batch_size"]
    dataloader_kwargs = dict(
        dataset=dataset_for_loader,
        batch_size=dataloader_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=config["dataloader"]["pin_memory"],
        collate_fn=dataset.collate_fn,
    )
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
    dataloader = DataLoader(**dataloader_kwargs)

    print(f"Dataset created with {len(dataset)} samples")
    print(f"Using {max_samples} samples for inference")
    print(f"Batch size: {dataloader_batch_size}")
    print(f"Num workers: {args.num_workers}")
    if args.num_workers > 0:
        print("Persistent workers: True")
        print(f"Prefetch factor: {args.prefetch_factor}")
    print(f"Output CSV: {output_csv_path}")

    print("Testing model inference...")
    rows_written = 0
    batches_processed = 0
    low_solar_gpp_rows = 0
    with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = None
        include_ground_truth = False
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference", unit="batch"):
                batches_processed += 1
                if hasattr(batch, "to"):
                    batch = batch.to(device)

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

                if writer is None:
                    fieldnames = ["lat", "lon", "igbp", "timestamp"] + output_prediction_columns
                    include_ground_truth = res.ground_truth is not None
                    if include_ground_truth:
                        fieldnames += [f"gt_{flux}" for flux in output_flux_labels]
                    writer = csv.writer(csv_file)
                    writer.writerow(fieldnames)
                    print(f"Pred shape per batch: {yhat.shape}")

                preds_cpu = yhat.detach().cpu()
                gt_cpu = res.ground_truth.detach().cpu() if include_ground_truth else None
                lat_idx = batch.aux_columns.index("lat") if "lat" in batch.aux_columns else None
                lon_idx = batch.aux_columns.index("lon") if "lon" in batch.aux_columns else None
                aux_cpu = batch.aux_values.detach().cpu()

                rows = []
                for i in range(preds_cpu.shape[0]):
                    ts = batch.timestamps[i][-1] if len(batch.timestamps[i]) > 0 else ""
                    lat_val = float(aux_cpu[i, lat_idx].item() * 180.0) if lat_idx is not None else float("nan")
                    lon_val = float(aux_cpu[i, lon_idx].item() * 180.0) if lon_idx is not None else float("nan")
                    row = [
                        f"{lat_val:.2f}",
                        f"{lon_val:.2f}",
                        batch.igbp[i],
                        ts,
                    ]
                    for j in output_target_indices:
                        row.append(f"{preds_cpu[i, j].item():.4f}")
                    if include_ground_truth:
                        for j in output_target_indices:
                            row.append(f"{gt_cpu[i, j].item():.4f}")
                    rows.append(row)
                writer.writerows(rows)
                rows_written += len(rows)

    print(f"Saved predictions for {rows_written} samples across {batches_processed} batches to: {output_csv_path}")
    if force_zero_gpp_low_solar:
        print(
            f"Forced GPP=0 for {low_solar_gpp_rows} samples with "
            f"raw final SW_IN < {args.gpp_solar_threshold:g} W m-2 "
            f"(normalized SW_IN < {normalized_gpp_solar_threshold:.6g})"
        )


if __name__ == "__main__":
    main()
