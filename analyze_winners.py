#!/usr/bin/env python3
"""
역발상 분석: 실제로 +21%/-7% 기준으로 WIN이 되는 모든 날을 찾고,
그 날들의 공통 특성을 분석하여 최적의 매수 규칙을 도출한다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from modules.data_parser import parse_stock_xml
from modules.indicators import calc_all_indicators


def find_all_outcomes(df, take_profit=0.21, stop_loss=0.07):
    """모든 거래일에 대해 매수 시 WIN/LOSS 결과를 계산"""
    results = []

    for idx in range(60, len(df) - 1):
        entry_price = df.iloc[idx]['close']
        target = entry_price * (1 + take_profit)
        stop = entry_price * (1 - stop_loss)

        outcome = None
        exit_idx = None

        for j in range(idx + 1, len(df)):
            day = df.iloc[j]
            hit_target = day['high'] >= target
            hit_stop = day['low'] <= stop

            if hit_target and hit_stop:
                if day['open'] >= target:
                    outcome = 'WIN'
                else:
                    outcome = 'LOSS'
                exit_idx = j
                break
            elif hit_target:
                outcome = 'WIN'
                exit_idx = j
                break
            elif hit_stop:
                outcome = 'LOSS'
                exit_idx = j
                break

        if outcome:
            results.append({
                'idx': idx,
                'date': df.iloc[idx]['date'],
                'close': entry_price,
                'outcome': outcome,
                'holding_days': (df.iloc[exit_idx]['date'] - df.iloc[idx]['date']).days,
            })

    return results


def analyze_characteristics(df, results):
    """WIN vs LOSS 그룹의 기술지표 특성 비교"""
    win_stats = {k: [] for k in [
        'rsi', 'ma5_vs_ma20', 'ma20_vs_ma60', 'vol_ratio',
        'close_vs_ma20_pct', 'close_vs_ma60_pct',
        'recovery_from_20d_low', 'distance_from_20d_high',
        'ma20_slope_10d', 'macd_hist', 'bb_position',
        '5d_return', '10d_return', '20d_return',
        'is_bullish', 'consec_up_days',
    ]}
    loss_stats = {k: [] for k in win_stats}

    for r in results:
        idx = r['idx']
        row = df.iloc[idx]
        target = win_stats if r['outcome'] == 'WIN' else loss_stats

        # RSI
        if pd.notna(row.get('rsi')):
            target['rsi'].append(row['rsi'])

        # MA5 vs MA20
        ma5, ma20, ma60 = row.get('ma_5'), row.get('ma_20'), row.get('ma_60')
        if pd.notna(ma5) and pd.notna(ma20) and ma20 > 0:
            target['ma5_vs_ma20'].append((ma5 - ma20) / ma20 * 100)
        if pd.notna(ma20) and pd.notna(ma60) and ma60 > 0:
            target['ma20_vs_ma60'].append((ma20 - ma60) / ma60 * 100)

        # Volume ratio
        if pd.notna(row.get('vol_ratio')):
            target['vol_ratio'].append(row['vol_ratio'])

        # Close vs MA
        if pd.notna(ma20) and ma20 > 0:
            target['close_vs_ma20_pct'].append((row['close'] - ma20) / ma20 * 100)
        if pd.notna(ma60) and ma60 > 0:
            target['close_vs_ma60_pct'].append((row['close'] - ma60) / ma60 * 100)

        # Recovery from 20d low
        low_20d = df['low'].iloc[max(0, idx-20):idx+1].min()
        if low_20d > 0:
            target['recovery_from_20d_low'].append(
                (row['close'] - low_20d) / low_20d * 100)

        # Distance from 20d high
        high_20d = df['high'].iloc[max(0, idx-20):idx+1].max()
        if high_20d > 0:
            target['distance_from_20d_high'].append(
                (row['close'] - high_20d) / high_20d * 100)

        # MA20 slope (10-day)
        if idx >= 10 and pd.notna(ma20):
            ma20_10ago = df.iloc[idx-10].get('ma_20')
            if pd.notna(ma20_10ago) and ma20_10ago > 0:
                target['ma20_slope_10d'].append(
                    (ma20 - ma20_10ago) / ma20_10ago * 100)

        # MACD histogram
        if pd.notna(row.get('macd_hist')):
            target['macd_hist'].append(row['macd_hist'])

        # Bollinger position (0=lower, 0.5=mid, 1=upper)
        bb_lower = row.get('bb_lower')
        bb_upper = row.get('bb_upper')
        if pd.notna(bb_lower) and pd.notna(bb_upper) and bb_upper > bb_lower:
            target['bb_position'].append(
                (row['close'] - bb_lower) / (bb_upper - bb_lower))

        # N-day returns
        if idx >= 5:
            target['5d_return'].append(
                (row['close'] - df.iloc[idx-5]['close']) / df.iloc[idx-5]['close'] * 100)
        if idx >= 10:
            target['10d_return'].append(
                (row['close'] - df.iloc[idx-10]['close']) / df.iloc[idx-10]['close'] * 100)
        if idx >= 20:
            target['20d_return'].append(
                (row['close'] - df.iloc[idx-20]['close']) / df.iloc[idx-20]['close'] * 100)

        # Bullish candle
        target['is_bullish'].append(1 if row['close'] > row['open'] else 0)

        # Consecutive up days
        consec = 0
        for lb in range(1, min(10, idx+1)):
            if df.iloc[idx-lb+1]['close'] > df.iloc[idx-lb]['close']:
                consec += 1
            else:
                break
        target['consec_up_days'].append(consec)

    return win_stats, loss_stats


def print_comparison(win_stats, loss_stats):
    """WIN vs LOSS 특성 비교 출력"""
    print(f"\n{'지표':<25} | {'WIN 평균':>10} | {'LOSS 평균':>10} | {'차이':>10} | 해석")
    print("─" * 85)

    for key in win_stats:
        w = win_stats[key]
        l = loss_stats[key]
        if w and l:
            w_mean = np.mean(w)
            l_mean = np.mean(l)
            diff = w_mean - l_mean

            # 해석
            if key == 'rsi':
                interp = "WIN이 RSI 높음" if diff > 0 else "WIN이 RSI 낮음"
            elif key == 'ma20_slope_10d':
                interp = "WIN시 MA20 더 상승" if diff > 0 else "WIN시 MA20 더 하락"
            elif 'return' in key:
                interp = "WIN시 직전수익↑" if diff > 0 else "WIN시 직전수익↓"
            elif key == 'close_vs_ma20_pct':
                interp = "WIN시 MA20 위" if diff > 0 else "WIN시 MA20 아래"
            elif key == 'vol_ratio':
                interp = "WIN시 거래량 많음" if diff > 0 else "WIN시 거래량 적음"
            elif key == 'recovery_from_20d_low':
                interp = "WIN시 저점대비 반등↑" if diff > 0 else "WIN시 저점대비 반등↓"
            elif key == 'distance_from_20d_high':
                interp = "WIN시 고점 가까움" if diff > 0 else "WIN시 고점 먼"
            else:
                interp = ""

            print(f"  {key:<23} | {w_mean:>+10.2f} | {l_mean:>+10.2f} | "
                  f"{diff:>+10.2f} | {interp}")


if __name__ == '__main__':
    xml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'kakaopay_stock.xml')
    df, symbol, name = parse_stock_xml(xml_file)
    df = calc_all_indicators(df)

    print(f"=== {name} ({symbol}) WIN/LOSS 특성 분석 ===")
    print(f"기준: 매수 후 +21% 도달=WIN, -7% 도달=LOSS\n")

    results = find_all_outcomes(df)
    wins = [r for r in results if r['outcome'] == 'WIN']
    losses = [r for r in results if r['outcome'] == 'LOSS']

    print(f"전체 거래일: {len(results)}일")
    print(f"WIN: {len(wins)}일 ({len(wins)/len(results)*100:.1f}%)")
    print(f"LOSS: {len(losses)}일 ({len(losses)/len(results)*100:.1f}%)")
    print(f"기본 승률: {len(wins)/len(results)*100:.1f}%")

    win_stats, loss_stats = analyze_characteristics(df, results)
    print_comparison(win_stats, loss_stats)

    # 핵심 필터 조합 테스트
    print("\n" + "=" * 85)
    print("  필터 조합 테스트 (WIN이 되는 조건 탐색)")
    print("=" * 85)

    filters = [
        ("RSI > 50", lambda df, i: pd.notna(df.iloc[i].get('rsi')) and df.iloc[i]['rsi'] > 50),
        ("RSI 40~60", lambda df, i: pd.notna(df.iloc[i].get('rsi')) and 40 <= df.iloc[i]['rsi'] <= 60),
        ("종가 > MA20", lambda df, i: pd.notna(df.iloc[i].get('ma_20')) and df.iloc[i]['close'] > df.iloc[i]['ma_20']),
        ("종가 > MA60", lambda df, i: pd.notna(df.iloc[i].get('ma_60')) and df.iloc[i]['close'] > df.iloc[i]['ma_60']),
        ("MA5 > MA20", lambda df, i: pd.notna(df.iloc[i].get('ma_5')) and pd.notna(df.iloc[i].get('ma_20')) and df.iloc[i]['ma_5'] > df.iloc[i]['ma_20']),
        ("MA20 > MA60", lambda df, i: pd.notna(df.iloc[i].get('ma_20')) and pd.notna(df.iloc[i].get('ma_60')) and df.iloc[i]['ma_20'] > df.iloc[i]['ma_60']),
        ("정배열(5>20>60)", lambda df, i: pd.notna(df.iloc[i].get('ma_5')) and pd.notna(df.iloc[i].get('ma_20')) and pd.notna(df.iloc[i].get('ma_60')) and df.iloc[i]['ma_5'] > df.iloc[i]['ma_20'] > df.iloc[i]['ma_60']),
        ("5일수익>0", lambda df, i: i >= 5 and df.iloc[i]['close'] > df.iloc[i-5]['close']),
        ("10일수익>0", lambda df, i: i >= 10 and df.iloc[i]['close'] > df.iloc[i-10]['close']),
        ("20일수익>0", lambda df, i: i >= 20 and df.iloc[i]['close'] > df.iloc[i-20]['close']),
        ("20일수익>5%", lambda df, i: i >= 20 and (df.iloc[i]['close'] - df.iloc[i-20]['close']) / df.iloc[i-20]['close'] > 0.05),
        ("20일수익>10%", lambda df, i: i >= 20 and (df.iloc[i]['close'] - df.iloc[i-20]['close']) / df.iloc[i-20]['close'] > 0.10),
        ("거래량>1.5x", lambda df, i: pd.notna(df.iloc[i].get('vol_ratio')) and df.iloc[i]['vol_ratio'] > 1.5),
        ("양봉", lambda df, i: df.iloc[i]['close'] > df.iloc[i]['open']),
        ("MACD>Signal", lambda df, i: pd.notna(df.iloc[i].get('macd')) and pd.notna(df.iloc[i].get('macd_signal')) and df.iloc[i]['macd'] > df.iloc[i]['macd_signal']),
        ("MACD히스토>0", lambda df, i: pd.notna(df.iloc[i].get('macd_hist')) and df.iloc[i]['macd_hist'] > 0),
        ("BB상반(>중간)", lambda df, i: pd.notna(df.iloc[i].get('bb_mid')) and df.iloc[i]['close'] > df.iloc[i]['bb_mid']),
        ("20일고점근접(5%이내)", lambda df, i: (df['high'].iloc[max(0,i-20):i+1].max() - df.iloc[i]['close']) / df['high'].iloc[max(0,i-20):i+1].max() < 0.05),
        ("MA20기울기>0(10일)", lambda df, i: i >= 10 and pd.notna(df.iloc[i].get('ma_20')) and pd.notna(df.iloc[i-10].get('ma_20')) and df.iloc[i-10]['ma_20'] > 0 and df.iloc[i]['ma_20'] > df.iloc[i-10]['ma_20']),
        ("MA20기울기>2%(10일)", lambda df, i: i >= 10 and pd.notna(df.iloc[i].get('ma_20')) and pd.notna(df.iloc[i-10].get('ma_20')) and df.iloc[i-10]['ma_20'] > 0 and (df.iloc[i]['ma_20'] - df.iloc[i-10]['ma_20']) / df.iloc[i-10]['ma_20'] > 0.02),
    ]

    print(f"\n  {'필터':<25} | {'해당일':>5} | {'WIN':>4} | {'LOSS':>5} | {'승률':>6}")
    print("  " + "─" * 60)

    for name_f, func in filters:
        w_count = 0
        l_count = 0
        for r in results:
            try:
                if func(df, r['idx']):
                    if r['outcome'] == 'WIN':
                        w_count += 1
                    else:
                        l_count += 1
            except:
                pass

        total = w_count + l_count
        rate = w_count / total * 100 if total > 0 else 0
        marker = " ◀◀" if rate >= 40 else (" ◀" if rate >= 35 else "")
        print(f"  {name_f:<25} | {total:>5} | {w_count:>4} | {l_count:>5} | {rate:>5.1f}%{marker}")

    # 복합 필터 테스트
    print(f"\n  === 복합 필터 조합 ===")
    print(f"  {'필터조합':<45} | {'해당일':>5} | {'WIN':>4} | {'LOSS':>5} | {'승률':>6}")
    print("  " + "─" * 75)

    combos = [
        ("정배열 + 20일수익>5%",
         lambda df, i: (pd.notna(df.iloc[i].get('ma_5')) and pd.notna(df.iloc[i].get('ma_20')) and pd.notna(df.iloc[i].get('ma_60'))
                        and df.iloc[i]['ma_5'] > df.iloc[i]['ma_20'] > df.iloc[i]['ma_60']
                        and i >= 20 and (df.iloc[i]['close'] - df.iloc[i-20]['close']) / df.iloc[i-20]['close'] > 0.05)),
        ("정배열 + 20일수익>10%",
         lambda df, i: (pd.notna(df.iloc[i].get('ma_5')) and pd.notna(df.iloc[i].get('ma_20')) and pd.notna(df.iloc[i].get('ma_60'))
                        and df.iloc[i]['ma_5'] > df.iloc[i]['ma_20'] > df.iloc[i]['ma_60']
                        and i >= 20 and (df.iloc[i]['close'] - df.iloc[i-20]['close']) / df.iloc[i-20]['close'] > 0.10)),
        ("정배열 + MACD양전 + 양봉",
         lambda df, i: (pd.notna(df.iloc[i].get('ma_5')) and pd.notna(df.iloc[i].get('ma_20')) and pd.notna(df.iloc[i].get('ma_60'))
                        and df.iloc[i]['ma_5'] > df.iloc[i]['ma_20'] > df.iloc[i]['ma_60']
                        and pd.notna(df.iloc[i].get('macd_hist')) and df.iloc[i]['macd_hist'] > 0
                        and df.iloc[i]['close'] > df.iloc[i]['open'])),
        ("정배열 + MACD양전 + 20일수익>10%",
         lambda df, i: (pd.notna(df.iloc[i].get('ma_5')) and pd.notna(df.iloc[i].get('ma_20')) and pd.notna(df.iloc[i].get('ma_60'))
                        and df.iloc[i]['ma_5'] > df.iloc[i]['ma_20'] > df.iloc[i]['ma_60']
                        and pd.notna(df.iloc[i].get('macd_hist')) and df.iloc[i]['macd_hist'] > 0
                        and i >= 20 and (df.iloc[i]['close'] - df.iloc[i-20]['close']) / df.iloc[i-20]['close'] > 0.10)),
        ("MA20상승>2% + 20일수익>10% + RSI50~70",
         lambda df, i: (i >= 10 and pd.notna(df.iloc[i].get('ma_20')) and pd.notna(df.iloc[i-10].get('ma_20'))
                        and df.iloc[i-10]['ma_20'] > 0
                        and (df.iloc[i]['ma_20'] - df.iloc[i-10]['ma_20']) / df.iloc[i-10]['ma_20'] > 0.02
                        and i >= 20 and (df.iloc[i]['close'] - df.iloc[i-20]['close']) / df.iloc[i-20]['close'] > 0.10
                        and pd.notna(df.iloc[i].get('rsi')) and 50 <= df.iloc[i]['rsi'] <= 70)),
        ("MA20상승>2% + 정배열 + MACD양전",
         lambda df, i: (i >= 10 and pd.notna(df.iloc[i].get('ma_20')) and pd.notna(df.iloc[i-10].get('ma_20'))
                        and df.iloc[i-10]['ma_20'] > 0
                        and (df.iloc[i]['ma_20'] - df.iloc[i-10]['ma_20']) / df.iloc[i-10]['ma_20'] > 0.02
                        and pd.notna(df.iloc[i].get('ma_5')) and pd.notna(df.iloc[i].get('ma_60'))
                        and df.iloc[i]['ma_5'] > df.iloc[i]['ma_20'] > df.iloc[i]['ma_60']
                        and pd.notna(df.iloc[i].get('macd_hist')) and df.iloc[i]['macd_hist'] > 0)),
        ("20일고점근접5% + 정배열 + MACD양전",
         lambda df, i: (pd.notna(df.iloc[i].get('ma_5')) and pd.notna(df.iloc[i].get('ma_20')) and pd.notna(df.iloc[i].get('ma_60'))
                        and df.iloc[i]['ma_5'] > df.iloc[i]['ma_20'] > df.iloc[i]['ma_60']
                        and pd.notna(df.iloc[i].get('macd_hist')) and df.iloc[i]['macd_hist'] > 0
                        and (df['high'].iloc[max(0,i-20):i+1].max() - df.iloc[i]['close']) / df['high'].iloc[max(0,i-20):i+1].max() < 0.05)),
        ("20일고점근접5% + MA20상승>2% + 양봉",
         lambda df, i: (i >= 10 and pd.notna(df.iloc[i].get('ma_20')) and pd.notna(df.iloc[i-10].get('ma_20'))
                        and df.iloc[i-10]['ma_20'] > 0
                        and (df.iloc[i]['ma_20'] - df.iloc[i-10]['ma_20']) / df.iloc[i-10]['ma_20'] > 0.02
                        and (df['high'].iloc[max(0,i-20):i+1].max() - df.iloc[i]['close']) / df['high'].iloc[max(0,i-20):i+1].max() < 0.05
                        and df.iloc[i]['close'] > df.iloc[i]['open'])),
        ("정배열 + 20일고점5% + 20일수익>10% + 양봉",
         lambda df, i: (pd.notna(df.iloc[i].get('ma_5')) and pd.notna(df.iloc[i].get('ma_20')) and pd.notna(df.iloc[i].get('ma_60'))
                        and df.iloc[i]['ma_5'] > df.iloc[i]['ma_20'] > df.iloc[i]['ma_60']
                        and (df['high'].iloc[max(0,i-20):i+1].max() - df.iloc[i]['close']) / df['high'].iloc[max(0,i-20):i+1].max() < 0.05
                        and i >= 20 and (df.iloc[i]['close'] - df.iloc[i-20]['close']) / df.iloc[i-20]['close'] > 0.10
                        and df.iloc[i]['close'] > df.iloc[i]['open'])),
        ("MA20상승>2%+정배열+MACD양전+RSI50~70+양봉",
         lambda df, i: (i >= 10 and pd.notna(df.iloc[i].get('ma_20')) and pd.notna(df.iloc[i-10].get('ma_20'))
                        and df.iloc[i-10]['ma_20'] > 0
                        and (df.iloc[i]['ma_20'] - df.iloc[i-10]['ma_20']) / df.iloc[i-10]['ma_20'] > 0.02
                        and pd.notna(df.iloc[i].get('ma_5')) and pd.notna(df.iloc[i].get('ma_60'))
                        and df.iloc[i]['ma_5'] > df.iloc[i]['ma_20'] > df.iloc[i]['ma_60']
                        and pd.notna(df.iloc[i].get('macd_hist')) and df.iloc[i]['macd_hist'] > 0
                        and pd.notna(df.iloc[i].get('rsi')) and 50 <= df.iloc[i]['rsi'] <= 70
                        and df.iloc[i]['close'] > df.iloc[i]['open'])),
    ]

    for name_c, func in combos:
        w_count = 0
        l_count = 0
        for r in results:
            try:
                if func(df, r['idx']):
                    if r['outcome'] == 'WIN':
                        w_count += 1
                    else:
                        l_count += 1
            except:
                pass

        total = w_count + l_count
        rate = w_count / total * 100 if total > 0 else 0
        marker = " ★★" if rate >= 50 else (" ★" if rate >= 40 else (" ◀" if rate >= 35 else ""))
        print(f"  {name_c:<45} | {total:>5} | {w_count:>4} | {l_count:>5} | {rate:>5.1f}%{marker}")
