#!/usr/bin/env python3
"""Modelo Ohlson (Portugal) com FE e correção de colinearidade via componente ortogonal."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from value_relevance_analysis import build_dataset_ohlson, winsorize

EXCEL_PATH = Path('Dados/PT/Planilha Clonada para Portugal_VF_Out.25.xlsx')
COLUMNS = ['market_value', 'pl_hist_total', 'pl_corr_total', 'net_income', 'book_value']


def prepare_dataset(path: Path, winsor_pct: float = 0.05) -> pd.DataFrame:
    df = build_dataset_ohlson(path)
    if df.empty:
        return df

    df = df[['empresa', 'ano', *COLUMNS]].copy()
    df.sort_values(['empresa', 'ano'], inplace=True)
    df = df.dropna(subset=COLUMNS)
    df[COLUMNS] = df[COLUMNS].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=COLUMNS)
    df = winsorize(df, COLUMNS, winsor_pct)

    df['log_book'] = np.log(df['book_value'].clip(lower=1.0))

    X_hist = sm.add_constant(df['pl_hist_total'])
    beta = np.linalg.lstsq(X_hist.to_numpy(), df['pl_corr_total'].to_numpy(), rcond=None)[0]
    df['pl_corr_ortho'] = df['pl_corr_total'] - (beta[0] + beta[1] * df['pl_hist_total'])

    df.reset_index(drop=True, inplace=True)
    return df


def add_fixed_effects(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    X = df[list(features)].copy()
    firm_dummies = pd.get_dummies(df['empresa'], prefix='firm', drop_first=True)
    year_dummies = pd.get_dummies(df['ano'], prefix='year', drop_first=True)
    X = pd.concat([X, firm_dummies, year_dummies], axis=1)
    X = X.astype(float)
    return sm.add_constant(X, has_constant='add')


def compute_vif(X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    data = X[cols].astype(float)
    vif_values = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return pd.DataFrame({'variavel': cols, 'VIF': vif_values})


def run_model(df: pd.DataFrame, predictors: List[str], label: str, vif_cols: List[str]) -> None:
    y = df['market_value'].astype(float)
    X = add_fixed_effects(df, predictors + ['net_income', 'log_book'])
    model = sm.OLS(y, X).fit(cov_type='HC3')

    vifs = compute_vif(X, vif_cols)

    print(f'--- {label} ---')
    print(model.summary())
    print('VIF (variáveis selecionadas):')
    print(vifs.to_string(index=False))
    print()


def main() -> None:
    df = prepare_dataset(EXCEL_PATH, winsor_pct=0.05)
    if df.empty:
        print('Nenhum dado carregado.')
        return

    print(f'Observações: {len(df)} | Empresas: {df["empresa"].nunique()} | Anos: {df["ano"].nunique()}')

    vif_base = ['pl_hist_total', 'net_income', 'log_book']
    run_model(df, ['pl_hist_total'], 'MV ~ PL histórico + controles + FE', vif_base)

    vif_base_corr = ['pl_corr_total', 'net_income', 'log_book']
    run_model(df, ['pl_corr_total'], 'MV ~ PL corrigido + controles + FE', vif_base_corr)

    predictors = ['pl_hist_total', 'pl_corr_ortho']
    vif_cols_combo = ['pl_hist_total', 'pl_corr_ortho', 'net_income', 'log_book']
    run_model(df, predictors, 'MV ~ PL histórico + PL corrigido (ortogonal) + controles + FE', vif_cols_combo)


if __name__ == '__main__':
    main()
