#!/usr/bin/env python3
"""Regressões simples com winsorização, erros robustos e componente ortogonal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from value_relevance_analysis import build_dataset, winsorize

EXCEL_PATH = Path('Dados/BR/B3_1996_2024_Depreciado.xlsx')
COLUMNS = ['valor_mercado', 'pl_hist', 'pl_corrigido']


def prepare_dataset(path: Path, winsor_pct: float = 0.05) -> pd.DataFrame:
    df = build_dataset(path)
    if df.empty:
        return df

    df = df.rename(columns={'pl': 'pl_hist'}).dropna(subset=COLUMNS).copy()
    df[COLUMNS] = df[COLUMNS].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=COLUMNS)
    df = winsorize(df, COLUMNS, winsor_pct)

    X_hist = sm.add_constant(df['pl_hist'])
    beta = np.linalg.lstsq(X_hist.to_numpy(), df['pl_corrigido'].to_numpy(), rcond=None)[0]
    df['pl_corr_ortho'] = df['pl_corrigido'] - (beta[0] + beta[1] * df['pl_hist'])

    df.reset_index(drop=True, inplace=True)
    return df


def run_model(df: pd.DataFrame, predictors: list[str], label: str) -> None:
    subset = df.dropna(subset=['valor_mercado', *predictors]).copy()
    if subset.empty:
        print(f'Sem dados suficientes para {label}.')
        return

    X = sm.add_constant(subset[predictors].astype(float))
    y = subset['valor_mercado'].astype(float)

    model = sm.OLS(y, X).fit(cov_type='HC3')
    print(f'--- {label} ---')
    print(model.summary())
    print()


def main() -> None:
    df = prepare_dataset(EXCEL_PATH, winsor_pct=0.05)
    if df.empty:
        print('Nenhum dado disponível.')
        return

    print(f'Observações: {len(df)} | Empresas: {df["empresa"].nunique()} | Anos: {df["ano"].nunique()}')

    run_model(df, ['pl_hist'], 'MV ~ PL histórico (winsor, HC3)')
    run_model(df, ['pl_corrigido'], 'MV ~ PL corrigido (winsor, HC3)')
    run_model(df, ['pl_hist', 'pl_corr_ortho'], 'MV ~ PL histórico + PL corrigido ortogonal (winsor, HC3)')


if __name__ == '__main__':
    main()
