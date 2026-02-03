#!/usr/bin/env python3
"""Regressões simples após winsorização de variáveis em diferentes níveis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import statsmodels.api as sm

from value_relevance_analysis import build_dataset, winsorize

EXCEL_PATH = Path('Dados/BR/B3_1996_2024_Depreciado.xlsx')
COLUMNS = ['valor_mercado', 'pl_hist', 'pl_corrigido']


def run_model(df: pd.DataFrame, predictors: Iterable[str], label: str) -> None:
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
    df[COLUMNS] = df[COLUMNS].apply(pd.to_numeric, errors='coerce')

    for pct in (0.01, 0.02, 0.05):
        print('=' * 80)
        print(f'Winsorização nos percentis {pct:.0%} e {(1 - pct):.0%}')
        clipped = winsorize(df, COLUMNS, pct)
        run_model(clipped, ['pl_hist'], 'PL histórico')
        print()
        run_model(clipped, ['pl_corrigido'], 'PL corrigido')
        print()
        run_model(clipped, ['pl_hist', 'pl_corrigido'], 'PL histórico + PL corrigido')
        print()


if __name__ == '__main__':
    main()
