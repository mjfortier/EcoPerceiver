from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def resolve_checkpoint_path(run_path: Path, explicit_checkpoint_path: Path | None) -> Path:
    if explicit_checkpoint_path is not None:
        checkpoint_path = explicit_checkpoint_path.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        return checkpoint_path

    last_checkpoint = run_path / "last.pth"
    if last_checkpoint.exists():
        return last_checkpoint.resolve()

    candidates = list(run_path.glob("checkpoint-*.pth"))
    numeric_candidates: list[tuple[int, Path]] = []
    for candidate in candidates:
        suffix = candidate.stem.removeprefix("checkpoint-")
        if suffix.isdigit():
            numeric_candidates.append((int(suffix), candidate))

    if not numeric_candidates:
        raise FileNotFoundError(
            f"No checkpoint found in {run_path}. Expected last.pth or checkpoint-*.pth."
        )

    return max(numeric_candidates, key=lambda item: item[0])[1].resolve()


def resolve_config_path(run_folder: Path, explicit_config_path: Path | None) -> Path:
    if explicit_config_path is not None:
        if not explicit_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {explicit_config_path}")
        return explicit_config_path

    candidates = [
        run_folder / "config.yml",
        run_folder.parent / "config.yml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve config.yml from run folder {run_folder}. "
        "Looked at run_folder/config.yml and run_folder/../config.yml."
    )


def resolve_device(device_arg: str) -> "torch.device":
    import torch

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_data_dir(data_dir: Path) -> Path:
    resolved_data_dir = data_dir.expanduser().resolve()
    sql_file = resolved_data_dir / "carbonsense_v2.sql"
    if not sql_file.exists():
        raise FileNotFoundError(
            f"Dataset sqlite file not found: {sql_file}. "
            "Provide --data_dir pointing to extracted carbonsense_v2 directory."
        )
    return resolved_data_dir
