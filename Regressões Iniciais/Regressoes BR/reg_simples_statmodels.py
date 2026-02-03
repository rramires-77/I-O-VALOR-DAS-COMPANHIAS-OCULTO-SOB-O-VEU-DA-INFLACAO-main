#!/usr/bin/env python3
"""Regressões simples relacionando valor de mercado com PL histórico e corrigido."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

from value_relevance_analysis import build_dataset

EXCEL_PATH = Path('Dados/BR/B3_1996_2024_Depreciado.xlsx')


def run_model(df: pd.DataFrame, predictors: list[str], label: str) -> None:
    subset = df.dropna(subset=['valor_mercado', *predictors]).copy()
    if subset.empty:
        print(f'Sem observações suficientes para {label}.')
        return

    y = subset['valor_mercado'].astype(float)
    X = sm.add_constant(subset[predictors].astype(float))
    result = sm.OLS(y, X).fit()

    print(f'--- Modelo: MV ~ {label} ---')
    print(result.summary())


def main() -> None:
    df = build_dataset(EXCEL_PATH)
    if df.empty:
        print('Nenhum dado foi encontrado no arquivo de entrada.')
        return

    df = df.rename(columns={'pl': 'pl_hist'}).copy()
    numeric_cols = ['valor_mercado', 'pl_hist', 'pl_corrigido']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    run_model(df, ['pl_hist'], 'PL histórico')
    print()
    run_model(df, ['pl_corrigido'], 'PL corrigido')
    print()
    run_model(df, ['pl_hist', 'pl_corrigido'], 'PL histórico + PL corrigido')


if __name__ == '__main__':
    main()
