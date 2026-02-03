#!/usr/bin/env python3
"""Modelo de Ohlson (Portugal) com controles adicionais e efeitos fixos."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm

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
    df.reset_index(drop=True, inplace=True)
    return df


def run_model(df: pd.DataFrame, predictors: List[str], label: str) -> None:
    if df.empty:
        print('Sem dados disponíveis.')
        return

    cols = predictors + ['net_income', 'log_book']
    X = df[cols].copy()
    firm_dummies = pd.get_dummies(df['empresa'], prefix='firm', drop_first=True)
    year_dummies = pd.get_dummies(df['ano'], prefix='year', drop_first=True)
    X = pd.concat([X, firm_dummies, year_dummies], axis=1)
    X = X.astype(float)
    X = sm.add_constant(X, has_constant='add')

    y = df['market_value'].astype(float)
    model = sm.OLS(y, X).fit(cov_type='HC3')

    print(f'--- {label} ---')
    print(model.summary())
    print()


def main() -> None:
    df = prepare_dataset(EXCEL_PATH, winsor_pct=0.05)
    if df.empty:
        print('Nenhum dado foi carregado.')
        return

    print(f'Observações utilizadas: {len(df)} | Empresas: {df["empresa"].nunique()} | Anos: {df["ano"].nunique()}')

    run_model(df, ['pl_hist_total'], 'MV ~ PL histórico + controles + FE')
    run_model(df, ['pl_corr_total'], 'MV ~ PL corrigido + controles + FE')
    run_model(df, ['pl_hist_total', 'pl_corr_total'], 'MV ~ PL histórico + PL corrigido + controles + FE')


if __name__ == '__main__':
    main()
