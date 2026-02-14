"""Generate LaTeX tables from evaluation metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

BASELINE_MODEL_COMPARISON = {
    # IGBP: {"xgb": {"nse": ..., "rmse": ...}, "eco": {"nse": ..., "rmse": ...}}
    "CRO": {"xgb": {"nse": 0.8066, "rmse": 3.2381}, "eco": {"nse": 0.8482, "rmse": 2.8677}},
    "CSH": {"xgb": {"nse": 0.7510, "rmse": 1.5224}, "eco": {"nse": 0.7670, "rmse": 1.4709}},
    "CVM": {"xgb": {"nse": 0.5277, "rmse": 5.5157}, "eco": {"nse": 0.5763, "rmse": 5.2236}},
    "DBF": {"xgb": {"nse": 0.7250, "rmse": 4.0959}, "eco": {"nse": 0.7547, "rmse": 3.8678}},
    "DNF": {"xgb": {"nse": 0.2803, "rmse": 4.0974}, "eco": {"nse": 0.4336, "rmse": 3.6322}},
    "EBF": {"xgb": {"nse": 0.7966, "rmse": 4.6050}, "eco": {"nse": 0.8220, "rmse": 4.3070}},
    "ENF": {"xgb": {"nse": 0.7765, "rmse": 2.8141}, "eco": {"nse": 0.7694, "rmse": 2.8579}},
    "GRA": {"xgb": {"nse": 0.7461, "rmse": 3.2487}, "eco": {"nse": 0.7967, "rmse": 2.9059}},
    "MF": {"xgb": {"nse": 0.7559, "rmse": 3.8633}, "eco": {"nse": 0.7717, "rmse": 3.7361}},
    "OSH": {"xgb": {"nse": 0.5451, "rmse": 1.8796}, "eco": {"nse": 0.6060, "rmse": 1.7475}},
    "SAV": {"xgb": {"nse": 0.5802, "rmse": 1.6514}, "eco": {"nse": 0.7368, "rmse": 1.3070}},
    "SNO": {"xgb": {"nse": -0.0370, "rmse": 1.4291}, "eco": {"nse": 0.2898, "rmse": 1.1816}},
    "WAT": {"xgb": {"nse": -11.0524, "rmse": 3.1838}, "eco": {"nse": -14.4010, "rmse": 3.5802}},
    "WET": {"xgb": {"nse": 0.4530, "rmse": 2.2073}, "eco": {"nse": 0.4137, "rmse": 2.2830}},
    "WSA": {"xgb": {"nse": 0.6132, "rmse": 2.5153}, "eco": {"nse": 0.6267, "rmse": 2.4706}},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert eval_test_sites_metrics CSV to a LaTeX table."
    )
    parser.add_argument(
        "--run_folder",
        type=Path,
        default=None,
        help="Run folder. When provided, defaults outputs to <run_folder>/eval/latex/.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to the metrics CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the LaTeX output. Uses stdout when set to '-'.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Number of decimal places to format floating values.",
    )
    parser.add_argument(
        "--nan-token",
        default="--",
        help="Token to use for missing values (NaN).",
    )
    parser.add_argument(
        "--caption",
        default="RMSE and NSE",
        help="LaTeX caption for the table.",
    )
    parser.add_argument(
        "--label",
        default="tab:metrics",
        help="LaTeX label for the table.",
    )
    parser.add_argument(
        "--igbp-output",
        type=Path,
        default=None,
        help="Where to write the IGBP RMSE LaTeX output. Uses stdout when set to '-'.",
    )
    parser.add_argument(
        "--igbp-caption",
        default="RMSE by IGBP",
        help="LaTeX caption for the IGBP RMSE table.",
    )
    parser.add_argument(
        "--igbp-label",
        default="tab:rmse_metrics_igbp",
        help="LaTeX label for the IGBP RMSE table.",
    )
    parser.add_argument(
        "--nse-igbp-output",
        type=Path,
        default=None,
        help="Where to write the IGBP NSE LaTeX output. Uses stdout when set to '-'.",
    )
    parser.add_argument(
        "--nse-igbp-caption",
        default="NSE by IGBP",
        help="LaTeX caption for the IGBP NSE table.",
    )
    parser.add_argument(
        "--nse-igbp-label",
        default="tab:nse_metrics_igbp",
        help="LaTeX label for the IGBP NSE table.",
    )
    parser.add_argument(
        "--compare-output",
        type=Path,
        default=None,
        help="Where to write the comparison LaTeX output. Uses stdout when set to '-'.",
    )
    parser.add_argument(
        "--compare-caption",
        default="EcoPerceiver v2: RMSE/NSE comparison",
        help="LaTeX caption for the comparison table.",
    )
    parser.add_argument(
        "--compare-label",
        default="tab:eco_v2_metrics",
        help="LaTeX label for the comparison table.",
    )
    return parser.parse_args()


def pick_rmse_columns(columns: Iterable[str]) -> List[str]:
    return [c for c in columns if c.startswith("rmse_")]


def pick_nse_columns(columns: Iterable[str]) -> List[str]:
    return [c for c in columns if c.startswith("nse_")]


def format_metric_name(name: str) -> str:
    """Strip metric prefix, drop trailing _dt, remove underscores, uppercase."""
    if name.startswith("rmse_"):
        stripped = name.removeprefix("rmse_")
    elif name.startswith("nse_"):
        stripped = name.removeprefix("nse_")
    else:
        stripped = name
    lower = stripped.lower()
    if lower.endswith("_dt"):
        stripped = stripped[: -3]  # drop _dt (case-insensitive)
    elif lower.endswith("dt"):
        stripped = stripped[: -2]  # fallback: drop trailing dt
    stripped = stripped.replace("_", "")
    return stripped.upper()


def build_float_formatter(decimals: int, nan_token: str):
    def fmt(x: float) -> str:
        if pd.isna(x):
            return nan_token
        return f"{x:.{decimals}f}"

    return fmt


def escape_metric_name(name: str) -> str:
    return name.replace("_", r"\_")


def choose_source_row(df: pd.DataFrame) -> pd.Series:
    """Prefer the TOTAL row (if a site column exists), else first row."""
    if "site" in df.columns:
        mask = df["site"].astype(str).str.lower() == "total"
        if mask.any():
            return df.loc[mask].iloc[0]
    return df.iloc[0]


def build_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    rmse_cols = pick_rmse_columns(df.columns)
    if not rmse_cols:
        raise ValueError("No RMSE columns (rmse_*) found in CSV.")

    row = choose_source_row(df)
    rows = []
    for rmse_col in rmse_cols:
        nse_col = rmse_col.replace("rmse_", "nse_", 1)
        rows.append(
            {
                "Target": format_metric_name(rmse_col),
                "RMSE": row[rmse_col],
                "NSE": row[nse_col] if nse_col in row.index else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def build_igbp_metric_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregate average metric per IGBP for metric in {'rmse', 'nse'}."""
    if "igbp" not in df.columns:
        raise ValueError("Column 'igbp' not found in CSV.")

    metric = metric.lower()
    if metric == "rmse":
        metric_cols = pick_rmse_columns(df.columns)
        if not metric_cols:
            raise ValueError("No RMSE columns (rmse_*) found in CSV.")
    elif metric == "nse":
        metric_cols = pick_nse_columns(df.columns)
        if not metric_cols:
            raise ValueError("No NSE columns (nse_*) found in CSV.")
    else:
        raise ValueError("metric must be one of {'rmse', 'nse'}.")

    grouped = (
        df.loc[df["igbp"].notna()]
        .groupby("igbp")[metric_cols]
        .mean()
        .reset_index()
    )

    rename_map = {c: format_metric_name(c) for c in metric_cols}
    return grouped.rename(columns={"igbp": "IGBP", **rename_map})


def bold_if(text: str, condition: bool) -> str:
    return f"\\textbf{{{text}}}" if condition else text


def build_eco_v2_table(
    df: pd.DataFrame, decimals: int, nan_token: str
) -> pd.DataFrame:
    """Build RMSE/NSE comparison table with grouped headers."""
    if "igbp" not in df.columns or "rmse_NEE" not in df.columns or "nse_NEE" not in df.columns:
        raise ValueError("Columns 'igbp', 'rmse_NEE', and 'nse_NEE' must exist in CSV.")

    baseline_rmse = (
        df.loc[df["igbp"].notna()]
        .groupby("igbp")["rmse_NEE"]
        .mean()
        .round(decimals)
    )
    baseline_nse = (
        df.loc[df["igbp"].notna()]
        .groupby("igbp")["nse_NEE"]
        .mean()
        .round(decimals)
    )

    def fmt(val: float) -> str:
        if pd.isna(val):
            return nan_token
        return f"{val:.{decimals}f}"

    rows = []
    igbps = sorted(set(baseline_rmse.index).union(BASELINE_MODEL_COMPARISON.keys()))

    for igbp in igbps:
        comp = BASELINE_MODEL_COMPARISON.get(igbp, {})
        xgb_rmse = comp.get("xgb", {}).get("rmse", pd.NA)
        xgb_nse = comp.get("xgb", {}).get("nse", pd.NA)
        eco_rmse = comp.get("eco", {}).get("rmse", pd.NA)
        eco_nse = comp.get("eco", {}).get("nse", pd.NA)
        eco_v2_rmse = baseline_rmse.get(igbp, pd.NA)
        eco_v2_nse = baseline_nse.get(igbp, pd.NA)

        # Winner is the lowest RMSE among available values
        rmse_values = {
            "xgb": xgb_rmse,
            "eco": eco_rmse,
            "eco_v2": eco_v2_rmse,
        }
        rmse_best = min(
            v for v in rmse_values.values() if not pd.isna(v)
        ) if any(not pd.isna(v) for v in rmse_values.values()) else pd.NA

        # Winner is the highest NSE among available values.
        nse_values = {
            "xgb": xgb_nse,
            "eco": eco_nse,
            "eco_v2": eco_v2_nse,
        }
        nse_best = max(
            v for v in nse_values.values() if not pd.isna(v)
        ) if any(not pd.isna(v) for v in nse_values.values()) else pd.NA

        def bold_if_rmse_best(key: str, text: str) -> str:
            if pd.isna(rmse_best):
                return text
            val = rmse_values[key]
            if pd.isna(val):
                return text
            return bold_if(text, val == rmse_best)

        def bold_if_nse_best(key: str, text: str) -> str:
            if pd.isna(nse_best):
                return text
            val = nse_values[key]
            if pd.isna(val):
                return text
            return bold_if(text, val == nse_best)

        xgb_rmse_txt = bold_if_rmse_best("xgb", fmt(xgb_rmse))
        eco_rmse_txt = bold_if_rmse_best("eco", fmt(eco_rmse))
        eco_v2_rmse_txt = bold_if_rmse_best("eco_v2", fmt(eco_v2_rmse))
        xgb_nse_txt = bold_if_nse_best("xgb", fmt(xgb_nse))
        eco_nse_txt = bold_if_nse_best("eco", fmt(eco_nse))
        eco_v2_nse_txt = bold_if_nse_best("eco_v2", fmt(eco_v2_nse))

        rows.append(
            {
                ("", "IGBP"): igbp,
                ("XGBoost", "RMSE"): xgb_rmse_txt,
                ("XGBoost", "NSE"): xgb_nse_txt,
                ("EcoPerceiver", "RMSE"): eco_rmse_txt,
                ("EcoPerceiver", "NSE"): eco_nse_txt,
                ("EcoPerceiver v2", "RMSE"): eco_v2_rmse_txt,
                ("EcoPerceiver v2", "NSE"): eco_v2_nse_txt,
            }
        )

    columns = pd.MultiIndex.from_tuples(
        [
            ("", "IGBP"),
            ("XGBoost", "RMSE"),
            ("XGBoost", "NSE"),
            ("EcoPerceiver", "RMSE"),
            ("EcoPerceiver", "NSE"),
            ("EcoPerceiver v2", "RMSE"),
            ("EcoPerceiver v2", "NSE"),
        ]
    )

    return pd.DataFrame(rows, columns=columns)


def to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    float_format,
    column_format: str | None = None,
    enable_multicolumn: bool = False,
    cmidrule_line: str | None = None,
) -> str:
    if column_format is None:
        column_format = "l" + "c" * (len(df.columns) - 1)

    tabular = df.to_latex(
        index=False,
        float_format=float_format,
        column_format=column_format,
        escape=False,
        multicolumn=enable_multicolumn,
        multicolumn_format="c",
        multirow=False,
        na_rep="--",
        sparsify=True,
        index_names=True,
    )

    if enable_multicolumn:
        lines = tabular.splitlines()
        for i, line in enumerate(lines):
            if "XGBoost" in line:
                cmid = cmidrule_line or " \\cmidrule(lr){2-2}\\cmidrule(lr){3-3}\\cmidrule(lr){4-4}"
                lines.insert(i + 1, cmid)
                break
        tabular = "\n".join(lines)

    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{tabular}\n"
        "\\end{table}\n"
    )


def write_table(content: str, output_path: Path, label: str) -> None:
    if str(output_path) == "-":
        print(content)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"Wrote {label} LaTeX table to {output_path.resolve()}")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent

    if args.run_folder is not None:
        run_folder = args.run_folder.expanduser().resolve()
        eval_dir = run_folder / "eval"
        latex_dir = eval_dir / "latex"
        default_csv = eval_dir / "test_sites_metrics.csv"
        default_output = latex_dir / "metrics.tex"
        default_igbp_output = latex_dir / "rmse_by_igbp.tex"
        default_nse_igbp_output = latex_dir / "nse_by_igbp.tex"
        default_compare_output = latex_dir / "eco_v2.tex"
    else:
        default_csv = script_dir / "eval_test_sites_metrics.csv"
        default_output = script_dir / "metrics.tex"
        default_igbp_output = script_dir / "rmse_by_igbp.tex"
        default_nse_igbp_output = script_dir / "nse_by_igbp.tex"
        default_compare_output = script_dir / "eco_v2.tex"

    csv_path = args.csv if args.csv is not None else default_csv
    output_path = args.output if args.output is not None else default_output
    igbp_output_path = (
        args.igbp_output if args.igbp_output is not None else default_igbp_output
    )
    nse_igbp_output_path = (
        args.nse_igbp_output
        if args.nse_igbp_output is not None
        else default_nse_igbp_output
    )
    compare_output_path = (
        args.compare_output if args.compare_output is not None else default_compare_output
    )
    return csv_path, output_path, igbp_output_path, nse_igbp_output_path, compare_output_path


def main() -> None:
    args = parse_args()
    (
        csv_path,
        output_path,
        igbp_output_path,
        nse_igbp_output_path,
        compare_output_path,
    ) = resolve_paths(args)
    df_raw = pd.read_csv(csv_path)

    metrics_table = build_metrics_table(df_raw)
    igbp_rmse_table = build_igbp_metric_table(df_raw, metric="rmse")
    igbp_nse_table = build_igbp_metric_table(df_raw, metric="nse")
    eco_v2_table = build_eco_v2_table(df_raw, args.decimals, args.nan_token)
    float_fmt = build_float_formatter(args.decimals, args.nan_token)
    latex = to_latex_table(
        metrics_table,
        caption=args.caption,
        label=args.label,
        float_format=float_fmt,
    )
    latex_igbp = to_latex_table(
        igbp_rmse_table,
        caption=args.igbp_caption,
        label=args.igbp_label,
        float_format=float_fmt,
    )
    latex_igbp_nse = to_latex_table(
        igbp_nse_table,
        caption=args.nse_igbp_caption,
        label=args.nse_igbp_label,
        float_format=float_fmt,
    )
    latex_eco_v2 = to_latex_table(
        eco_v2_table,
        caption=args.compare_caption,
        label=args.compare_label,
        float_format=float_fmt,
        column_format="lcccccc",
        enable_multicolumn=True,
        cmidrule_line=" \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
    )

    write_table(latex, output_path, "overall metrics")
    write_table(latex_igbp, igbp_output_path, "IGBP RMSE metrics")
    write_table(latex_igbp_nse, nse_igbp_output_path, "IGBP NSE metrics")
    write_table(latex_eco_v2, compare_output_path, "comparison (EcoPerceiver v2)")


if __name__ == "__main__":
    main()
