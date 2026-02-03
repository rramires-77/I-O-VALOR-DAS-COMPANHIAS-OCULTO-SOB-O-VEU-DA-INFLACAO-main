#!/usr/bin/env python3
"""Modelo de Ohlson (Portugal) com valores totais winsorizados."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from value_relevance_analysis import build_dataset_ohlson, winsorize

EXCEL_PATH = Path('Dados/PT/Planilha Clonada para Portugal_VF_Out.25.xlsx')
COLUMNS = ['market_value', 'pl_hist_total', 'pl_corr_total', 'net_income']


def prepare_dataset(path: Path) -> pd.DataFrame:
    df = build_dataset_ohlson(path)
    if df.empty:
        return df

    df = df[['empresa', 'ano', 'market_value', 'pl_hist_total', 'pl_corr_total', 'net_income']].copy()
    df.sort_values(['empresa', 'ano'], inplace=True)
    df = df.dropna(subset=['market_value', 'net_income'])
    df[COLUMNS] = df[COLUMNS].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=COLUMNS)
    return df


def run_model(df: pd.DataFrame, pct: float) -> None:
    clipped = winsorize(df, COLUMNS, pct)

    def regress(predictors: list[str], label: str) -> None:
        subset = clipped.dropna(subset=predictors + ['market_value', 'net_income']).copy()
        if subset.empty:
            print(f'Sem dados suficientes para {label}.')
            return
        X = sm.add_constant(subset[predictors + ['net_income']])
        y = subset['market_value']
        result = sm.OLS(y, X).fit()
        print(f'--- {label} ---')
        print(result.summary())
        print()

    print('=' * 80)
    print(f'Winsorização nos percentis {pct:.0%} e {(1 - pct):.0%}')
    regress(['pl_hist_total'], 'MV ~ PL histórico + Lucro líquido')
    regress(['pl_corr_total'], 'MV ~ PL corrigido + Lucro líquido')
    regress(['pl_hist_total', 'pl_corr_total'], 'MV ~ PL histórico + PL corrigido + Lucro líquido')


def main() -> None:
    df = prepare_dataset(EXCEL_PATH)
    if df.empty:
        print('Nenhum dado disponível.')
        return

    for pct in (0.01, 0.02, 0.05):
        run_model(df, pct)


if __name__ == '__main__':
    main()
