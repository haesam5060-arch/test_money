#!/usr/bin/env python3
"""배점 수정 전/후 백테스트 비교 검증"""

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kakao/Desktop/project/연구')

import numpy as np
from modules.backtester import run_backtest
from modules.data_parser import parse_stock_xml
from modules.indicators import calc_all_indicators

XML_DIR = '/Users/kakao/Desktop/project/연구/xml/'

def load_stock(filepath):
    try:
        df, sym, name = parse_stock_xml(filepath)
        df = calc_all_indicators(df)
        return df, sym, name
    except:
        return None, None, None

def run_all_backtests():
    xml_files = sorted([f for f in os.listdir(XML_DIR) if f.endswith('.xml')])
    all_trades = []
    for fname in xml_files:
        df, sym, name = load_stock(os.path.join(XML_DIR, fname))
        if df is None or len(df) < 100:
            continue
        trades = run_backtest(df, buy_threshold=4.0, take_profit=0.17,
                              stop_loss=0.07, cooldown=5, benford_window=30,
                              profile_name='default', benford_influence=0.15,
                              benford_min_hits=5, rsi_min=70)
        for t in trades:
            if t['result'] in ('WIN', 'LOSS') and np.isfinite(t['return_pct']):
                all_trades.append(t)
    return all_trades

print("=" * 60)
print("  배점 수정 후 백테스트 검증")
print("=" * 60)
print("\n수정된 배점으로 전체 종목 백테스트 실행 중...")

trades = run_all_backtests()
wins = [t for t in trades if t['result'] == 'WIN']
losses = [t for t in trades if t['result'] == 'LOSS']

wr = len(wins) / len(trades) * 100
avg_ret = np.mean([t['return_pct'] for t in trades])
avg_win = np.mean([t['return_pct'] for t in wins]) if wins else 0
avg_loss = np.mean([t['return_pct'] for t in losses]) if losses else 0
ev = wr/100 * avg_win + (1 - wr/100) * avg_loss

print(f"\n총 거래: {len(trades)}건")
print(f"WIN: {len(wins)}건 | LOSS: {len(losses)}건")
print(f"승률: {wr:.1f}%")
print(f"평균 수익률: {avg_ret:+.2f}%")
print(f"평균 수익(WIN): {avg_win:+.2f}%")
print(f"평균 손실(LOSS): {avg_loss:+.2f}%")
print(f"기대값(EV): {ev:+.2f}%")

# 점수 분포 확인
scores = [t['score'] for t in trades]
print(f"\n점수 분포:")
print(f"  평균: {np.mean(scores):.1f}")
print(f"  최소: {np.min(scores):.1f}")
print(f"  최대: {np.max(scores):.1f}")
print(f"  중앙값: {np.median(scores):.1f}")

# 점수 구간별 성과
print(f"\n점수 구간별 성과:")
for lo, hi in [(4, 6), (6, 8), (8, 10), (10, 12), (12, 20)]:
    sub = [t for t in trades if lo <= t['score'] < hi]
    if len(sub) >= 5:
        sw = [t for t in sub if t['result'] == 'WIN']
        w = len(sw) / len(sub) * 100
        r = np.mean([t['return_pct'] for t in sub])
        print(f"  score {lo:>2}-{hi:<2}: 승률 {w:5.1f}% | 평균수익 {r:>+6.2f}% | {len(sub):>4}건")

print(f"\n{'─'*60}")
print(f"  비교 (수정 전 → 수정 후)")
print(f"{'─'*60}")
print(f"  수정 전: 1881건, 승률 38.2%, 평균수익 +0.75%")
print(f"  수정 후: {len(trades)}건, 승률 {wr:.1f}%, 평균수익 {avg_ret:+.2f}%")
