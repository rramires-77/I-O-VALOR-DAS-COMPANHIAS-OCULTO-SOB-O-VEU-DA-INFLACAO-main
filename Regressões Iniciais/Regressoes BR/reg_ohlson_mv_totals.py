#!/usr/bin/env python3
"""Modelo de Ohlson usando valores totais (MV, PL, lucro líquido)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

from value_relevance_analysis import build_dataset_ohlson

EXCEL_PATH = Path('Dados/BR/B3_1996_2024_Depreciado.xlsx')


def prepare_dataset(path: Path) -> pd.DataFrame:
    df = build_dataset_ohlson(path)
    if df.empty:
        return df

    df = df[['empresa', 'ano', 'market_value', 'pl_hist_total', 'pl_corr_total', 'net_income']].copy()
    df.sort_values(['empresa', 'ano'], inplace=True)
    df = df.dropna(subset=['market_value', 'net_income'])
    return df


def run_model(df: pd.DataFrame, predictors: list[str], label: str) -> None:
    subset = df.dropna(subset=predictors + ['market_value', 'net_income']).copy()
    if subset.empty:
        print(f'Sem dados suficientes para {label}.')
        return

    X = sm.add_constant(subset[predictors + ['net_income']])
    y = subset['market_value']
    result = sm.OLS(y, X).fit()

    print(f'--- {label} ---')
    print(result.summary())
    print()


def main() -> None:
    df = prepare_dataset(EXCEL_PATH)
    if df.empty:
        print('Nenhum dado disponível.')
        return

    print(f'Observações totais: {len(df)}')
    run_model(df, ['pl_hist_total'], 'MV ~ PL histórico + Lucro líquido')
    run_model(df, ['pl_corr_total'], 'MV ~ PL corrigido + Lucro líquido')
    run_model(df, ['pl_hist_total', 'pl_corr_total'], 'MV ~ PL histórico + PL corrigido + Lucro líquido')


if __name__ == '__main__':
    main()
