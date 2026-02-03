#!/usr/bin/env python3
"""Modelo de Ohlson winsorizado com erros robustos (HC3 e HAC)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from value_relevance_analysis import build_dataset_ohlson, winsorize

EXCEL_PATH = Path('Dados/BR/B3_1996_2024_Depreciado.xlsx')
COLUMNS = ['market_value', 'pl_hist_total', 'pl_corr_total', 'net_income']


def prepare_dataset(path: Path) -> pd.DataFrame:
    df = build_dataset_ohlson(path)
    if df.empty:
        return df

    df = df[['empresa', 'ano', 'market_value', 'pl_hist_total', 'pl_corr_total', 'net_income']].copy()
    df.sort_values(['empresa', 'ano'], inplace=True)
    df[COLUMNS] = df[COLUMNS].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=COLUMNS)
    return df


def fit_with_covariance(base_model: RegressionResultsWrapper) -> dict[str, RegressionResultsWrapper]:
    return {
        'HC3': base_model.get_robustcov_results(cov_type='HC3'),
        'HAC(1)': base_model.get_robustcov_results(cov_type='HAC', maxlags=1),
    }


def regress(df: pd.DataFrame, predictors: list[str], label: str) -> None:
    subset = df.dropna(subset=predictors + ['market_value', 'net_income']).copy()
    if subset.empty:
        print(f'Sem dados suficientes para {label}.')
        return

    X = sm.add_constant(subset[predictors + ['net_income']])
    y = subset['market_value']
    base = sm.OLS(y, X).fit()
    robust_results = fit_with_covariance(base)

    print(f'--- {label} ---')
    print('> HC3')
    print(robust_results['HC3'].summary())
    print('> HAC (Newey-West, maxlag=1)')
    print(robust_results['HAC(1)'].summary())
    print()


def run_level(df: pd.DataFrame, pct: float) -> None:
    clipped = winsorize(df, COLUMNS, pct)
    print('=' * 80)
    print(f'Winsorização nos percentis {pct:.0%} e {(1 - pct):.0%}')
    regress(clipped, ['pl_hist_total'], 'MV ~ PL histórico + Lucro líquido')
    regress(clipped, ['pl_corr_total'], 'MV ~ PL corrigido + Lucro líquido')
    regress(clipped, ['pl_hist_total', 'pl_corr_total'], 'MV ~ PL histórico + PL corrigido + Lucro líquido')


def main() -> None:
    df = prepare_dataset(EXCEL_PATH)
    if df.empty:
        print('Nenhum dado disponível.')
        return

    for pct in (0.01, 0.02, 0.05):
        run_level(df, pct)


if __name__ == '__main__':
    main()
