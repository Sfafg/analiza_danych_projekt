#!/usr/bin/env python3
"""
Eksploracyjna analiza danych ze zbioru steam.csv.

Wymagania:
    pip install pandas matplotlib

Przykłady uruchomienia:
    python3 analiza_steam.py
    python3 analiza_steam.py --csv steam.csv --out wykresy --top 15

Skrypt wypisuje statystyki wszystkich kolumn, a wykresy zapisuje w katalogu
podanym w argumencie --out (domyślnie "wykresy").
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MULTI_VALUE_COLUMNS = {"platforms", "categories", "genres", "steamspy_tags"}
ID_COLUMNS = {"appid", "name"}
DISCRETE_COLUMNS = {"english", "required_age"}

COLOR_MAIN = "#4C72B0"
COLOR_MEAN = "#C44E52"
COLOR_MEDIAN = "#2E8B57"
COLOR_ACCENT = "#DD8452"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analiza statystyczna i wizualizacja danych Steam."
    )
    parser.add_argument(
        "--csv",
        default="steam.csv",
        help="Ścieżka do pliku CSV (domyślnie: steam.csv).",
    )
    parser.add_argument(
        "--out",
        default="wykresy",
        help="Katalog na wykresy i tabele statystyk (domyślnie: wykresy).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Liczba kategorii prezentowanych na wykresach słupkowych.",
    )
    return parser.parse_args()


def parse_owner_range(value: object) -> tuple[float, float]:
    if pd.isna(value):
        return np.nan, np.nan
    parts = str(value).strip().split("-")
    try:
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
        number = float(parts[0])
        return number, number
    except ValueError:
        return np.nan, np.nan


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    original_columns = list(df.columns)

    parsed_dates = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = parsed_dates.dt.year

    owners_bounds = df["owners"].map(parse_owner_range)
    df["owners_mid"] = owners_bounds.map(lambda bounds: sum(bounds) / 2)

    return df, original_columns


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        mode = series.mode()
        rows.append(
            {
                "liczba": int(series.count()),
                "braki": int(series.isna().sum()),
                "średnia": series.mean(),
                "odchylenie_std": series.std(),
                "min": series.min(),
                "Q1": q1,
                "mediana": series.median(),
                "Q3": q3,
                "max": series.max(),
                "rozstęp": series.max() - series.min(),
                "IQR": q3 - q1,
                "asymetria": series.skew(),
                "kurtoza": series.kurt(),
                "dominanta": mode.iloc[0] if not mode.empty else np.nan,
            }
        )
    return pd.DataFrame(rows, index=columns).rename_axis("kolumna")


def categorical_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        series = df[column].dropna().astype(str)
        counts = series.value_counts()
        top_value = counts.index[0] if not counts.empty else ""
        top_count = int(counts.iloc[0]) if not counts.empty else 0
        rows.append(
            {
                "liczba": int(series.size),
                "braki": int(df[column].isna().sum()),
                "unikalne": int(series.nunique()),
                "najczęstsza_wartość": shorten(top_value, 80),
                "częstość": top_count,
                "udział_%": 100 * top_count / len(series) if len(series) else np.nan,
            }
        )
    return pd.DataFrame(rows, index=columns).rename_axis("kolumna")


def category_counts(df: pd.DataFrame, column: str) -> pd.Series:
    series = df[column].dropna().astype(str)
    if column in MULTI_VALUE_COLUMNS:
        series = series.str.split(";").explode().str.strip()
        series = series[series != ""]
    return series.value_counts()


def owner_lower_bound(value: object) -> float:
    return parse_owner_range(value)[0]


def shorten(value: object, width: int = 60) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def safe_filename(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()


def should_use_log_scale(series: pd.Series) -> bool:
    if len(series) < 3 or series.min() < 0:
        return False
    return bool(series.skew() > 2 and series.max() > 10)


def draw_histogram(
    ax: plt.Axes,
    series: pd.Series,
    column: str,
    bins: int = 40,
    show_legend: bool = True,
) -> None:
    data = series.dropna()
    use_log = should_use_log_scale(data)
    plot_data = np.log10(data + 1) if use_log else data

    ax.hist(
        plot_data,
        bins=bins,
        color=COLOR_MAIN,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )

    if use_log:
        mean_position = np.log10(data.mean() + 1)
        median_position = np.log10(data.median() + 1)
        ax.set_xlabel(f"log10(1 + {column})")
        title_suffix = " – skala logarytmiczna"
    else:
        mean_position = data.mean()
        median_position = data.median()
        ax.set_xlabel(column)
        title_suffix = ""

    ax.axvline(
        mean_position,
        color=COLOR_MEAN,
        linestyle="--",
        linewidth=1.8,
        label=f"Średnia: {data.mean():,.2f}",
    )
    ax.axvline(
        median_position,
        color=COLOR_MEDIAN,
        linestyle=":",
        linewidth=2.0,
        label=f"Mediana: {data.median():,.2f}",
    )

    ax.set_ylabel("Liczba gier")
    ax.set_title(f"Histogram – {column}{title_suffix}")
    ax.grid(axis="y", alpha=0.3)
    if show_legend:
        ax.legend(frameon=False)


def draw_boxplot(
    ax: plt.Axes,
    series: pd.Series,
    column: str,
    show_title: bool = True,
) -> None:
    data = series.dropna()
    ax.boxplot(
        data,
        widths=0.45,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        boxprops={"facecolor": "#CFE2F3", "edgecolor": COLOR_MAIN, "linewidth": 1.2},
        medianprops={"color": COLOR_MEDIAN, "linewidth": 2.0},
        meanprops={"color": COLOR_MEAN, "linestyle": "--", "linewidth": 1.5},
        whiskerprops={"color": COLOR_MAIN, "linewidth": 1.1},
        capprops={"color": COLOR_MAIN, "linewidth": 1.1},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": COLOR_ACCENT,
            "markeredgecolor": COLOR_ACCENT,
            "alpha": 0.45,
        },
    )

    title_suffix = ""
    if should_use_log_scale(data):
        ax.set_yscale("symlog", linthresh=1)
        title_suffix = " – skala symlog"

    ax.set_ylabel(column)
    ax.set_xticks([1])
    ax.set_xticklabels([column])
    if show_title:
        ax.set_title(f"Wykres pudełkowy – {column}{title_suffix}")
    ax.grid(axis="y", alpha=0.3)


def save_histogram(df: pd.DataFrame, column: str, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    draw_histogram(ax, df[column], column)
    fig.tight_layout()
    path = output_dir / f"hist_{safe_filename(column)}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_boxplot(df: pd.DataFrame, column: str, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    draw_boxplot(ax, df[column], column)
    fig.tight_layout()
    path = output_dir / f"box_{safe_filename(column)}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_overview(
    df: pd.DataFrame,
    columns: list[str],
    output_dir: Path,
    kind: str,
) -> Path:
    columns_per_row = 3
    rows = math.ceil(len(columns) / columns_per_row)
    fig, axes = plt.subplots(
        rows,
        columns_per_row,
        figsize=(5.2 * columns_per_row, 4.2 * rows),
    )
    flat_axes = np.atleast_1d(axes).ravel()

    for ax, column in zip(flat_axes, columns):
        if kind == "hist":
            draw_histogram(ax, df[column], column, bins=30, show_legend=False)
        else:
            draw_boxplot(ax, df[column], column)

    for ax in flat_axes[len(columns) :]:
        ax.set_visible(False)

    title = (
        "Rozkłady zmiennych numerycznych – histogramy"
        if kind == "hist"
        else "Rozkłady zmiennych numerycznych – wykresy pudełkowe"
    )
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    filename = "przeglad_histogramy.png" if kind == "hist" else "przeglad_boxploty.png"
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_category_plot(
    df: pd.DataFrame,
    column: str,
    output_dir: Path,
    top_n: int,
) -> Path:
    counts = category_counts(df, column)

    if column == "owners":
        ordered = sorted(counts.index, key=owner_lower_bound)
        counts = counts.reindex(ordered)
    else:
        counts = counts.head(top_n).sort_values()

    fig_height = max(4.5, 0.42 * len(counts) + 2.0)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    positions = np.arange(len(counts))
    labels = [shorten(value, 45) for value in counts.index]

    ax.barh(
        positions,
        counts.values,
        color=COLOR_MAIN,
        alpha=0.9,
        edgecolor="white",
    )
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Liczba wystąpień")
    ax.set_title(f"Najczęstsze wartości – {column}")
    ax.grid(axis="x", alpha=0.3)

    max_value = counts.max() if not counts.empty else 0
    offset = max_value * 0.015
    for position, value in zip(positions, counts.values):
        ax.text(
            value + offset,
            position,
            f"{value:,}",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, max_value * 1.15 if max_value else 1)

    fig.tight_layout()
    path = output_dir / f"kategorie_{safe_filename(column)}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_release_year_plot(df: pd.DataFrame, output_dir: Path) -> Path:
    counts = df["release_year"].dropna().astype(int).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(counts.index, counts.values, color=COLOR_MAIN, edgecolor="white")
    ax.set_xlabel("Rok wydania")
    ax.set_ylabel("Liczba gier")
    ax.set_title("Liczba gier według roku wydania")
    ax.set_xticks(counts.index[::2])
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "wydania_wg_roku.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def print_report(
    df: pd.DataFrame,
    dataset_name: str,
    original_columns: list[str],
    categorical_columns: list[str],
    numeric_table: pd.DataFrame,
    categorical_table: pd.DataFrame,
    top_n: int,
) -> None:
    separator = "=" * 88
    print(separator)
    print(f"RAPORT ZBIORU DANYCH: {dataset_name}")
    print(separator)
    print(f"Liczba rekordów: {len(df):,}".replace(",", " "))
    print(f"Liczba kolumn: {len(original_columns)}")
    print(f"Duplikaty pełnych wierszy: {int(df.duplicated().sum())}")
    print(
        f"Łączna liczba brakujących wartości: {int(df[original_columns].isna().sum().sum())}"
    )

    missing = df[original_columns].isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("Brakujące wartości: brak")
    else:
        print("\nBrakujące wartości według kolumn:")
        print(missing.to_string())

    print("\n" + separator)
    print("STATYSTYKI NUMERYCZNE")
    print(separator)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(numeric_table.round(3).to_string())

    print("\n" + separator)
    print("STATYSTYKI KATEGORIALNE")
    print(separator)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(categorical_table.round(2).to_string())

    print("\n" + separator)
    print("ZMIENNA DATOWA")
    print(separator)
    dates = pd.to_datetime(df["release_date"], errors="coerce")
    valid_dates = dates.dropna()
    if valid_dates.empty:
        print("Brak poprawnych dat.")
    else:
        span_years = (valid_dates.max() - valid_dates.min()).days / 365.25
        print(
            f"Zakres dat: {valid_dates.min():%Y-%m-%d} – {valid_dates.max():%Y-%m-%d}"
        )
        print(f"Rozpiętość: {span_years:.1f} lat")
        print(f"Liczba unikalnych lat wydania: {valid_dates.dt.year.nunique()}")
        print(f"Niepoprawne daty: {int(dates.isna().sum())}")

    print("\n" + separator)
    print("ZMIENNE POCHODNE")
    print(separator)
    print("release_year – rok wydania gry na podstawie release_date")
    print("owners_mid  – środek przedziału z kolumny owners")
    derived = numeric_summary(df, ["release_year", "owners_mid"])
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(
            derived[["średnia", "min", "Q1", "mediana", "Q3", "max"]]
            .round(2)
            .to_string()
        )

    print("\n" + separator)
    print(f"NAJCZĘSTSZE WARTOŚCI (TOP {top_n})")
    print(separator)
    for column in categorical_columns:
        counts = category_counts(df, column)
        if column == "owners":
            counts = counts.reindex(sorted(counts.index, key=owner_lower_bound))
            presented = counts
        else:
            presented = counts.head(top_n)
        print(f"\n{column}:")
        print(f"  {'wartość':50s} {'liczba':>9s}")
        for value, count in presented.items():
            print(f"  {shorten(value, 50):50s} {count:>9,d}")


def save_tables(
    numeric_table: pd.DataFrame,
    categorical_table: pd.DataFrame,
    derived_table: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths = []
    tables = {
        "statystyki_numeryczne.csv": numeric_table,
        "statystyki_kategoryczne.csv": categorical_table,
        "statystyki_pochodne.csv": derived_table,
    }
    for filename, table in tables.items():
        path = output_dir / filename
        table.to_csv(
            path,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.out)

    if not csv_path.exists():
        raise SystemExit(f"Nie znaleziono pliku: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.family": "DejaVu Sans",
        }
    )

    df = pd.read_csv(csv_path)
    df, original_columns = prepare_data(df)

    numeric_columns = [
        column
        for column in original_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_columns = [
        column
        for column in original_columns
        if column not in numeric_columns and column != "release_date"
    ]

    numeric_table = numeric_summary(df, numeric_columns)
    categorical_table = categorical_summary(df, categorical_columns)
    derived_table = numeric_summary(df, ["release_year", "owners_mid"])

    print_report(
        df=df,
        dataset_name=csv_path.name,
        original_columns=original_columns,
        categorical_columns=categorical_columns,
        numeric_table=numeric_table,
        categorical_table=categorical_table,
        top_n=args.top,
    )

    continuous_columns = [
        column
        for column in numeric_columns
        if column not in ID_COLUMNS and column not in DISCRETE_COLUMNS
    ]
    plotted_categorical_columns = [
        "english",
        "required_age",
        "owners",
        "platforms",
        "categories",
        "genres",
        "steamspy_tags",
        "developer",
        "publisher",
    ]
    plotted_categorical_columns = [
        column for column in plotted_categorical_columns if column in original_columns
    ]

    generated_plots: list[Path] = []

    for column in continuous_columns:
        generated_plots.append(save_histogram(df, column, output_dir))
        generated_plots.append(save_boxplot(df, column, output_dir))

    generated_plots.append(save_overview(df, continuous_columns, output_dir, "hist"))
    generated_plots.append(save_overview(df, continuous_columns, output_dir, "box"))

    for column in plotted_categorical_columns:
        generated_plots.append(save_category_plot(df, column, output_dir, args.top))

    generated_plots.append(save_release_year_plot(df, output_dir))

    saved_tables = save_tables(
        numeric_table,
        categorical_table,
        derived_table,
        output_dir,
    )

    print("\n" + "=" * 88)
    print("ZAPISANE PLIKI")
    print("=" * 88)
    print(f"Wykresy ({len(generated_plots)} plików): {output_dir}")
    for path in generated_plots:
        print(f"  - {path.name}")
    print(f"\nTabele statystyk: {output_dir}")
    for path in saved_tables:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
