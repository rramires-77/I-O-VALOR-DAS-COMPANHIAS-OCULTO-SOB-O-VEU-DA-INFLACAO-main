#!/usr/bin/env python3
"""Compara PL histórico vs. corrigido no modelo de Ohlson para Portugal."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from value_relevance_analysis import build_dataset_ohlson

EXCEL_PATH = Path('Dados/PT/Planilha Clonada para Portugal_VF_Out.25.xlsx')


def prepare_dataset(path: Path) -> pd.DataFrame:
    df = build_dataset_ohlson(path)
    if df.empty:
        return df

    df = df.copy()
    df.sort_values(['empresa', 'ano'], inplace=True)

    if 'price' not in df.columns:
        df['price'] = df['market_value']
    df['bvps_hist_pl'] = df.get('bvps_hist', df['pl_hist_total'])
    df['bvps_corr_pl'] = df.get('bvps_corr', df['pl_corr_total'])
    df['eps'] = df.get('eps', df['net_income'])

    numeric_cols = ['price', 'eps', 'bvps_hist_pl', 'bvps_corr_pl']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=numeric_cols)

    return df


def run_model(df: pd.DataFrame, predictors: list[str], label: str) -> None:
    subset = df.dropna(subset=['price', 'eps', *predictors]).copy()
    if subset.empty:
        print(f'Sem dados suficientes para {label}.')
        return

    X = sm.add_constant(subset[predictors + ['eps']])
    y = subset['price']
    result = sm.OLS(y, X).fit()

    print(f'--- {label} ---')
    print(result.summary())
    print()


def main() -> None:
    df = prepare_dataset(EXCEL_PATH)
    if df.empty:
        print('Nenhum dado disponível para o modelo.')
        return

    print(f'Observações disponíveis: {len(df)}')
    run_model(df, ['bvps_hist_pl'], 'Preço ~ PL histórico por ação + EPS')
    run_model(df, ['bvps_corr_pl'], 'Preço ~ PL corrigido por ação + EPS')
    run_model(df, ['bvps_hist_pl', 'bvps_corr_pl'], 'Preço ~ PL histórico + PL corrigido por ação + EPS')


if __name__ == '__main__':
    main()
