#!/usr/bin/env python3
"""
ATR 배수 그리드서치
==================
TP: ATR × {2.0, 2.5, 3.0, 3.5, 4.0}
SL: ATR × {1.5, 2.0, 2.5, 3.0}

+ 기준선(baseline): ATR 미사용 (기존 로직)
"""

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kakao/Desktop/project/연구')

import numpy as np
from modules.backtester import run_backtest
from modules.data_parser import parse_stock_xml
from modules.indicators import calc_all_indicators

XML_DIR = '/Users/kakao/Desktop/project/연구/xml/'

# ATR 배수 그리드
GRID_TP_MULT = [0, 2.0, 2.5, 3.0, 3.5, 4.0]  # 0 = ATR 미사용(기존)
GRID_SL_MULT = [0, 1.5, 2.0, 2.5, 3.0]        # 0 = ATR 미사용(기존)


def load_all_stocks():
    xml_files = sorted([f for f in os.listdir(XML_DIR) if f.endswith('.xml')])
    stocks = []
    for fname in xml_files:
        filepath = os.path.join(XML_DIR, fname)
        try:
            df, sym, name = parse_stock_xml(filepath)
            df = calc_all_indicators(df)
            if len(df) >= 100:
                stocks.append((df, sym, name))
        except:
            pass
    return stocks


def run_grid(stocks, atr_tp, atr_sl):
    """모든 종목에 대해 백테스트 실행, 집계 반환"""
    all_trades = []
    for df, sym, name in stocks:
        trades = run_backtest(df, buy_threshold=4.0, take_profit=0.17,
                              stop_loss=0.07, cooldown=5, benford_window=30,
                              profile_name='default', benford_influence=0.15,
                              benford_min_hits=5, rsi_min=70,
                              atr_tp_mult=atr_tp if atr_tp > 0 else 999,
                              atr_sl_mult=atr_sl if atr_sl > 0 else 999)
        for t in trades:
            if t['result'] in ('WIN', 'LOSS') and np.isfinite(t['return_pct']):
                all_trades.append(t)
    return all_trades


def summarize(trades):
    if not trades:
        return 0, 0.0, 0.0, 0.0
    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']
    n = len(trades)
    wr = len(wins) / n * 100
    avg_ret = np.mean([t['return_pct'] for t in trades])
    avg_win = np.mean([t['return_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['return_pct'] for t in losses]) if losses else 0
    ev = (wr / 100) * avg_win + (1 - wr / 100) * avg_loss
    return n, wr, avg_ret, ev


if __name__ == '__main__':
    print("=" * 75)
    print("  ATR 배수 그리드서치")
    print("  TP 배수: ", GRID_TP_MULT)
    print("  SL 배수: ", GRID_SL_MULT)
    print("=" * 75)

    print("\n종목 로딩 중...")
    stocks = load_all_stocks()
    print(f"  {len(stocks)}개 종목 로드 완료\n")

    results = []

    total = len(GRID_TP_MULT) * len(GRID_SL_MULT)
    done = 0

    for tp_mult in GRID_TP_MULT:
        for sl_mult in GRID_SL_MULT:
            done += 1
            tp_label = f'ATR×{tp_mult}' if tp_mult > 0 else '미사용'
            sl_label = f'ATR×{sl_mult}' if sl_mult > 0 else '미사용'
            print(f"  [{done}/{total}] TP={tp_label}, SL={sl_label} ...", end='', flush=True)

            trades = run_grid(stocks, tp_mult, sl_mult)
            n, wr, avg_ret, ev = summarize(trades)
            results.append({
                'tp_mult': tp_mult, 'sl_mult': sl_mult,
                'n': n, 'wr': wr, 'avg_ret': avg_ret, 'ev': ev,
                'tp_label': tp_label, 'sl_label': sl_label,
            })
            print(f" {n}건, 승률 {wr:.1f}%, 수익 {avg_ret:+.2f}%, EV {ev:+.2f}%")

    # ── 결과 정리 ──
    print(f"\n\n{'=' * 75}")
    print("  그리드서치 결과 (EV 내림차순)")
    print("=" * 75)
    results.sort(key=lambda x: x['ev'], reverse=True)

    print(f"\n  {'순위':>3} {'TP':>10} {'SL':>10} {'거래수':>6} {'승률':>7} {'평균수익':>9} {'EV':>9}")
    print(f"  {'─' * 58}")

    baseline = next((r for r in results if r['tp_mult'] == 0 and r['sl_mult'] == 0), None)

    for i, r in enumerate(results):
        marker = ' ★' if r == baseline else ''
        tp_str = r['tp_label']
        sl_str = r['sl_label']
        print(f"  {i+1:>3}  {tp_str:>10} {sl_str:>10} {r['n']:>6} {r['wr']:>6.1f}% {r['avg_ret']:>+8.2f}% {r['ev']:>+8.2f}%{marker}")

    if baseline:
        print(f"\n  ★ = 기존 로직 (ATR 미사용): 승률 {baseline['wr']:.1f}%, EV {baseline['ev']:+.2f}%")

    # 최적 조합
    best = results[0]
    print(f"\n  🏆 최적 조합: TP={best['tp_label']}, SL={best['sl_label']}")
    print(f"     승률 {best['wr']:.1f}%, 평균수익 {best['avg_ret']:+.2f}%, EV {best['ev']:+.2f}%")

    if baseline:
        ev_diff = best['ev'] - baseline['ev']
        wr_diff = best['wr'] - baseline['wr']
        print(f"     기존 대비: 승률 {wr_diff:+.1f}%p, EV {ev_diff:+.2f}%")
