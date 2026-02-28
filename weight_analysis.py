#!/usr/bin/env python3
"""
배점 검증 및 최적화 분석
========================
Phase 1: 지표별 수익 기여도 (상관계수 + 구간별 승률)
Phase 2: 로지스틱 회귀 최적 배점 도출 + 이중 가산 분석
Phase 3: Out-of-Sample 검증 (시간 분할)
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/kakao/Desktop/project/연구')

import numpy as np
import pandas as pd
from modules.backtester import run_backtest
from modules.data_parser import parse_stock_xml
from modules.indicators import calc_all_indicators
from modules.signal_engine import calculate_buy_score, get_profile
from modules.benford import analyze_volume_benford, analyze_price_change_benford

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
XML_DIR = '/Users/kakao/Desktop/project/연구/xml/'
RSI_MIN = 70
BUY_THRESHOLD = 4.0
TAKE_PROFIT = 0.17
STOP_LOSS = 0.07
COOLDOWN = 5
BENFORD_WINDOW = 30
BENFORD_INFLUENCE = 0.15
BENFORD_MIN_HITS = 5
PROFILE_NAME = 'default'
ROUND_TRIP_COST = 0.0051

# OOS 분할 기준일
OOS_CUTOFF = pd.Timestamp('2024-01-01')


# ─────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────
def load_stock(filepath):
    try:
        df, sym, name = parse_stock_xml(filepath)
        df = calc_all_indicators(df)
        return df, sym, name
    except Exception:
        return None, None, None


def extract_features_at_signal(df, idx):
    """신호 시점의 원본 지표값을 숫자로 추출"""
    row = df.iloc[idx]
    close = row['close']
    features = {}

    # 1. RSI
    rsi = row.get('rsi')
    features['rsi'] = float(rsi) if pd.notna(rsi) else np.nan

    # 2. MACD 히스토그램
    macd_hist = row.get('macd_hist')
    features['macd_hist'] = float(macd_hist) if pd.notna(macd_hist) else np.nan

    # MACD 상태: 양전 여부 (0/1)
    features['macd_positive'] = 1.0 if pd.notna(macd_hist) and macd_hist > 0 else 0.0

    # MACD 골든크로스 여부
    if idx >= 1:
        prev = df.iloc[idx - 1]
        macd = row.get('macd')
        macd_sig = row.get('macd_signal')
        prev_macd = prev.get('macd')
        prev_macd_sig = prev.get('macd_signal')
        if all(pd.notna(v) for v in [macd, macd_sig, prev_macd, prev_macd_sig]):
            features['macd_golden'] = 1.0 if (macd > macd_sig and prev_macd <= prev_macd_sig) else 0.0
        else:
            features['macd_golden'] = 0.0
    else:
        features['macd_golden'] = 0.0

    # 3. 20일 수익률
    if idx >= 20:
        price_20ago = df.iloc[idx - 20]['close']
        features['ret_20d'] = (close - price_20ago) / price_20ago if price_20ago > 0 else np.nan
    else:
        features['ret_20d'] = np.nan

    # 4. 거래량 배율
    vol_ratio = row.get('vol_ratio')
    features['vol_ratio'] = float(vol_ratio) if pd.notna(vol_ratio) else np.nan

    # 5. MA200 대비 위치 (%)
    ma200 = row.get('ma_200')
    if pd.notna(ma200) and ma200 > 0:
        features['ma200_dist'] = (close / ma200 - 1)
    else:
        features['ma200_dist'] = np.nan

    # 6. MA200 위/아래 (이진)
    features['above_ma200'] = 1.0 if pd.notna(ma200) and ma200 > 0 and close > ma200 else 0.0

    # 7. 일목균형표 상태
    tenkan = row.get('ichi_tenkan')
    kijun = row.get('ichi_kijun')
    cloud_a = row.get('ichi_cloud_a')
    cloud_b = row.get('ichi_cloud_b')
    if all(pd.notna(v) for v in [tenkan, kijun, cloud_a, cloud_b]):
        cloud_top = max(cloud_a, cloud_b)
        cloud_bot = min(cloud_a, cloud_b)
        features['above_cloud'] = 1.0 if close > cloud_top else (0.0 if close >= cloud_bot else -1.0)
        features['tenkan_gt_kijun'] = 1.0 if tenkan > kijun else 0.0
        features['cloud_bullish'] = 1.0 if cloud_a > cloud_b else 0.0
    else:
        features['above_cloud'] = np.nan
        features['tenkan_gt_kijun'] = np.nan
        features['cloud_bullish'] = np.nan

    # 8. 볼린저밴드 위치
    bb_mid = row.get('bb_mid')
    features['above_bb_mid'] = 1.0 if pd.notna(bb_mid) and close > bb_mid else 0.0

    # 9. MA60 기울기 (20일간)
    ma60 = row.get('ma_60')
    if idx >= 20 and pd.notna(ma60):
        ma60_20ago = df.iloc[idx - 20].get('ma_60')
        if pd.notna(ma60_20ago) and ma60_20ago > 0:
            features['ma60_slope'] = (ma60 - ma60_20ago) / ma60_20ago
        else:
            features['ma60_slope'] = np.nan
    else:
        features['ma60_slope'] = np.nan

    # 10. 연속 상승일
    consec = 0
    for lb in range(1, min(8, idx + 1)):
        if df.iloc[idx - lb + 1]['close'] > df.iloc[idx - lb]['close']:
            consec += 1
        else:
            break
    features['consec_up'] = float(consec)

    # 11. 20일 고점 대비 거리
    high_20d = df['high'].iloc[max(0, idx - 20):idx + 1].max()
    features['dist_from_high'] = (high_20d - close) / high_20d if high_20d > 0 else np.nan

    # 12. 상승장악형 캔들
    features['bullish_engulfing'] = 1.0 if row.get('is_bullish_engulfing', False) else 0.0

    # 13. MA20 기울기 (10일간)
    ma20 = row.get('ma_20')
    if idx >= 10 and pd.notna(ma20):
        ma20_10ago = df.iloc[idx - 10].get('ma_20')
        if pd.notna(ma20_10ago) and ma20_10ago > 0:
            features['ma20_slope'] = (ma20 - ma20_10ago) / ma20_10ago
        else:
            features['ma20_slope'] = np.nan
    else:
        features['ma20_slope'] = np.nan

    # 14. 벤포드 멀티플라이어 (원본 수치)
    p = get_profile(PROFILE_NAME)
    bw = min(p.get('benford_weight', 0.10), BENFORD_INFLUENCE)
    eff_window = max(BENFORD_WINDOW, BENFORD_MIN_HITS)
    try:
        volumes = df['volume'].iloc[max(0, idx - eff_window):idx + 1].values
        prices = df['close'].iloc[max(0, idx - eff_window):idx + 1].values
        vol_bscore, _ = analyze_volume_benford(volumes, window=eff_window)
        pc_bscore, _ = analyze_price_change_benford(prices, window=eff_window)
        benford_mult = 1.0 + min((vol_bscore + pc_bscore) * 0.1, BENFORD_INFLUENCE)
    except Exception:
        benford_mult = 1.0
    features['benford_mult'] = benford_mult

    # 15. buyScore (최종 점수)
    try:
        score, _ = calculate_buy_score(df, idx, BENFORD_WINDOW, PROFILE_NAME,
                                        BENFORD_INFLUENCE, BENFORD_MIN_HITS)
        features['buy_score'] = float(score)
    except Exception:
        features['buy_score'] = np.nan

    return features


def collect_all_trades_with_features():
    """195개 종목 전체 백테스트 + 신호 시점 지표 추출"""
    xml_files = sorted([f for f in os.listdir(XML_DIR) if f.endswith('.xml')])
    all_records = []
    stock_count = 0

    for fname in xml_files:
        filepath = os.path.join(XML_DIR, fname)
        df, sym, name = load_stock(filepath)
        if df is None or len(df) < 100:
            continue

        stock_count += 1
        trades = run_backtest(df, buy_threshold=BUY_THRESHOLD,
                              take_profit=TAKE_PROFIT, stop_loss=STOP_LOSS,
                              cooldown=COOLDOWN, benford_window=BENFORD_WINDOW,
                              profile_name=PROFILE_NAME,
                              benford_influence=BENFORD_INFLUENCE,
                              benford_min_hits=BENFORD_MIN_HITS,
                              rsi_min=RSI_MIN)

        # 백테스트에서 나온 거래를 기반으로, 신호 시점 지표를 다시 추출
        # run_backtest 내부 로직을 재현하여 신호 idx를 찾음
        signal_indices = _find_signal_indices(df, trades)

        for i, trade in enumerate(trades):
            if trade['result'] not in ('WIN', 'LOSS'):
                continue
            if not np.isfinite(trade['return_pct']):
                continue

            sig_idx = signal_indices[i] if i < len(signal_indices) else None
            if sig_idx is None:
                continue

            features = extract_features_at_signal(df, sig_idx)
            features['stock'] = name
            features['symbol'] = sym
            features['entry_date'] = trade['entry_date']
            features['result'] = trade['result']
            features['return_pct'] = trade['return_pct']
            features['win'] = 1 if trade['result'] == 'WIN' else 0
            features['holding_days'] = trade['holding_days']
            all_records.append(features)

    print(f"  종목 {stock_count}개 처리 완료, 총 거래 {len(all_records)}건 수집")
    return pd.DataFrame(all_records)


def _find_signal_indices(df, trades):
    """백테스트 거래 결과에서 신호 발생 idx를 역추적"""
    # 전략: 각 거래의 entry_date 직전에서 신호가 발생한 지점을 찾음
    # PENDING → IN_POSITION 구조이므로 entry_date 이전 1~COOLDOWN일 내에 신호 있음
    indices = []
    for trade in trades:
        entry_date = trade['entry_date']
        found = None
        # entry_date 기준 최대 10일 전까지 검색 (지정가 대기 기간)
        for look_back in range(0, 11):
            target_date = entry_date - pd.Timedelta(days=look_back)
            mask = df['date'] == target_date
            if mask.any():
                idx = df.index[mask][0]
                # 이 날 실제로 신호가 발생했는지 확인
                rsi = df.iloc[idx].get('rsi')
                if pd.notna(rsi) and rsi >= RSI_MIN:
                    score, _ = calculate_buy_score(df, idx, BENFORD_WINDOW, PROFILE_NAME,
                                                    BENFORD_INFLUENCE, BENFORD_MIN_HITS)
                    if score >= BUY_THRESHOLD:
                        found = idx
                        break
        if found is None:
            # fallback: entry_date의 바로 이전 거래일
            mask = df['date'] <= entry_date
            if mask.any():
                found = df.index[mask][-1]
        indices.append(found)
    return indices


# ═══════════════════════════════════════════════════════════════
# PHASE 1: 지표별 수익 기여도 분석
# ═══════════════════════════════════════════════════════════════
def phase1_analysis(df_all):
    print("=" * 65)
    print("  PHASE 1: 지표별 수익 기여도 분석")
    print("=" * 65)

    wins = df_all[df_all['win'] == 1]
    losses = df_all[df_all['win'] == 0]
    print(f"\n총 거래: {len(df_all)}건 (WIN: {len(wins)}건, LOSS: {len(losses)}건)")
    print(f"전체 승률: {len(wins)/len(df_all)*100:.1f}%")
    print(f"평균 수익률: {df_all['return_pct'].mean():.2f}%")

    # ── 상관계수 분석 ──
    feature_cols = [
        'rsi', 'macd_hist', 'macd_positive', 'macd_golden',
        'ret_20d', 'vol_ratio', 'ma200_dist', 'above_ma200',
        'above_cloud', 'tenkan_gt_kijun', 'cloud_bullish',
        'above_bb_mid', 'ma60_slope', 'consec_up', 'dist_from_high',
        'bullish_engulfing', 'ma20_slope', 'benford_mult', 'buy_score'
    ]

    current_weights = {
        'rsi': 2.5, 'macd_hist': 2.0, 'macd_positive': 2.0, 'macd_golden': 2.0,
        'ret_20d': 1.5, 'vol_ratio': 1.0, 'ma200_dist': 1.5, 'above_ma200': 1.5,
        'above_cloud': 1.0, 'tenkan_gt_kijun': 0.5, 'cloud_bullish': 0.3,
        'above_bb_mid': 0.3, 'ma60_slope': 1.0, 'consec_up': 0.5, 'dist_from_high': '-',
        'bullish_engulfing': 1.0, 'ma20_slope': '-', 'benford_mult': '×1.15', 'buy_score': '전체'
    }

    print(f"\n{'─'*65}")
    print(f"  [상관계수 — return_pct 기준]")
    print(f"{'─'*65}")
    print(f"  {'지표':<22} {'상관계수':>8}  {'p-value':>8}  {'현재배점':>8}")
    print(f"  {'─'*54}")

    from scipy import stats

    corr_results = {}
    for col in feature_cols:
        valid = df_all[[col, 'return_pct']].dropna()
        if len(valid) < 30:
            continue
        r, p_val = stats.pearsonr(valid[col], valid['return_pct'])
        corr_results[col] = (r, p_val, len(valid))
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        weight_str = str(current_weights.get(col, '-'))
        print(f"  {col:<22} {r:>+8.4f}  {p_val:>8.4f}{sig:>3}  {weight_str:>8}")

    # ── 승률 상관 ──
    print(f"\n{'─'*65}")
    print(f"  [상관계수 — 승률(win) 기준]")
    print(f"{'─'*65}")
    print(f"  {'지표':<22} {'상관계수':>8}  {'p-value':>8}")
    print(f"  {'─'*46}")

    for col in feature_cols:
        valid = df_all[[col, 'win']].dropna()
        if len(valid) < 30:
            continue
        r, p_val = stats.pointbiserialr(valid['win'], valid[col])
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        print(f"  {col:<22} {r:>+8.4f}  {p_val:>8.4f}{sig:>3}")

    # ── 구간별 승률 ──
    print(f"\n{'─'*65}")
    print(f"  [구간별 승률 분석]")
    print(f"{'─'*65}")

    # RSI 구간
    print(f"\n  ▸ RSI 구간별 승률:")
    for lo, hi in [(60, 65), (65, 70), (70, 75), (75, 80), (80, 85), (85, 100)]:
        sub = df_all[(df_all['rsi'] >= lo) & (df_all['rsi'] < hi)]
        if len(sub) > 0:
            wr = sub['win'].mean() * 100
            avg_ret = sub['return_pct'].mean()
            print(f"    RSI {lo:>3}-{hi:<3}: 승률 {wr:5.1f}% | 평균수익 {avg_ret:>+6.2f}% | {len(sub):>4}건")

    # 거래량 배율 구간
    print(f"\n  ▸ 거래량 배율 구간별 승률:")
    for lo, hi in [(0, 1.3), (1.3, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 100)]:
        sub = df_all[(df_all['vol_ratio'] >= lo) & (df_all['vol_ratio'] < hi)]
        if len(sub) >= 5:
            wr = sub['win'].mean() * 100
            avg_ret = sub['return_pct'].mean()
            print(f"    vol {lo:>4.1f}-{hi:<5.1f}: 승률 {wr:5.1f}% | 평균수익 {avg_ret:>+6.2f}% | {len(sub):>4}건")

    # MA200 위치
    print(f"\n  ▸ MA200 위치별 승률:")
    for label, cond in [('MA200 위', df_all['above_ma200'] == 1), ('MA200 아래', df_all['above_ma200'] == 0)]:
        sub = df_all[cond]
        if len(sub) >= 5:
            wr = sub['win'].mean() * 100
            avg_ret = sub['return_pct'].mean()
            print(f"    {label:<12}: 승률 {wr:5.1f}% | 평균수익 {avg_ret:>+6.2f}% | {len(sub):>4}건")

    # 일목 구름
    print(f"\n  ▸ 일목 구름 위치별 승률:")
    for label, val in [('구름 위', 1.0), ('구름 안', 0.0), ('구름 아래', -1.0)]:
        sub = df_all[df_all['above_cloud'] == val]
        if len(sub) >= 5:
            wr = sub['win'].mean() * 100
            avg_ret = sub['return_pct'].mean()
            print(f"    {label:<12}: 승률 {wr:5.1f}% | 평균수익 {avg_ret:>+6.2f}% | {len(sub):>4}건")

    # MACD 상태
    print(f"\n  ▸ MACD 상태별 승률:")
    for label, cond in [('MACD 양전', df_all['macd_positive'] == 1), ('MACD 음전', df_all['macd_positive'] == 0)]:
        sub = df_all[cond]
        if len(sub) >= 5:
            wr = sub['win'].mean() * 100
            avg_ret = sub['return_pct'].mean()
            print(f"    {label:<12}: 승률 {wr:5.1f}% | 평균수익 {avg_ret:>+6.2f}% | {len(sub):>4}건")

    # 연속 상승일
    print(f"\n  ▸ 연속 상승일별 승률:")
    for days in [1, 2, 3, 4, 5]:
        if days < 5:
            sub = df_all[df_all['consec_up'] == days]
        else:
            sub = df_all[df_all['consec_up'] >= days]
        if len(sub) >= 5:
            wr = sub['win'].mean() * 100
            avg_ret = sub['return_pct'].mean()
            label = f'{days}일' if days < 5 else f'{days}일+'
            print(f"    {label:<12}: 승률 {wr:5.1f}% | 평균수익 {avg_ret:>+6.2f}% | {len(sub):>4}건")

    # 벤포드
    print(f"\n  ▸ 벤포드 멀티플라이어 구간별 승률:")
    for lo, hi in [(1.0, 1.03), (1.03, 1.08), (1.08, 1.15), (1.15, 2.0)]:
        sub = df_all[(df_all['benford_mult'] >= lo) & (df_all['benford_mult'] < hi)]
        if len(sub) >= 5:
            wr = sub['win'].mean() * 100
            avg_ret = sub['return_pct'].mean()
            print(f"    벤포드 {lo:.2f}-{hi:.2f}: 승률 {wr:5.1f}% | 평균수익 {avg_ret:>+6.2f}% | {len(sub):>4}건")

    # buyScore 구간
    print(f"\n  ▸ buyScore 구간별 승률:")
    for lo, hi in [(4, 6), (6, 8), (8, 10), (10, 12), (12, 20)]:
        sub = df_all[(df_all['buy_score'] >= lo) & (df_all['buy_score'] < hi)]
        if len(sub) >= 5:
            wr = sub['win'].mean() * 100
            avg_ret = sub['return_pct'].mean()
            print(f"    score {lo:>2}-{hi:<2}: 승률 {wr:5.1f}% | 평균수익 {avg_ret:>+6.2f}% | {len(sub):>4}건")

    return corr_results


# ═══════════════════════════════════════════════════════════════
# PHASE 2: 로지스틱 회귀 최적 배점 도출
# ═══════════════════════════════════════════════════════════════
def phase2_analysis(df_all):
    print(f"\n\n{'=' * 65}")
    print("  PHASE 2: 최적 배점 도출 (로지스틱 회귀)")
    print("=" * 65)

    # buyScore의 개별 항목들 (buyScore 자체는 제외)
    feature_cols = [
        'rsi', 'macd_positive', 'macd_golden',
        'ret_20d', 'vol_ratio', 'above_ma200',
        'above_cloud', 'tenkan_gt_kijun', 'cloud_bullish',
        'above_bb_mid', 'ma60_slope', 'consec_up',
        'bullish_engulfing', 'benford_mult'
    ]

    current_max_pts = {
        'rsi': 2.5,
        'macd_positive': 1.5,    # MACD양전+가속 합산
        'macd_golden': 1.5,
        'ret_20d': 1.5,
        'vol_ratio': 1.0,
        'above_ma200': 1.5,
        'above_cloud': 1.0,
        'tenkan_gt_kijun': 0.5,
        'cloud_bullish': 0.3,
        'above_bb_mid': 0.3,
        'ma60_slope': 1.0,
        'consec_up': 0.5,
        'bullish_engulfing': 1.0,
        'benford_mult': 0.15,  # ×곱셈이므로 직접 비교 어려움
    }

    # 결측치 제거
    df_clean = df_all[feature_cols + ['win', 'return_pct']].dropna()
    print(f"\n분석 대상: {len(df_clean)}건 (결측 제거 후)")

    X = df_clean[feature_cols].values
    y_win = df_clean['win'].values
    y_ret = df_clean['return_pct'].values

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 로지스틱 회귀 (WIN/LOSS 예측) ──
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_scaled, y_win)
    lr_accuracy = lr.score(X_scaled, y_win)

    print(f"\n{'─'*65}")
    print(f"  [로지스틱 회귀 — WIN/LOSS 예측]")
    print(f"  학습 정확도: {lr_accuracy*100:.1f}%")
    print(f"{'─'*65}")

    # 계수를 원래 스케일로 변환하여 해석
    coefs = lr.coef_[0]
    # 정규화된 계수의 절대값 합 = 1로 만들어 비율 계산
    abs_sum = np.sum(np.abs(coefs))
    normalized = coefs / abs_sum if abs_sum > 0 else coefs

    # 현재 배점 총합 (기본 5.0 제외, 가산점만)
    current_total = sum(v for v in current_max_pts.values() if isinstance(v, (int, float)))

    print(f"\n  {'지표':<22} {'회귀계수':>8} {'기여비율':>8} {'현재배점':>8} {'제안배점':>8} {'변화':>8}")
    print(f"  {'─'*62}")

    suggestions = {}
    for i, col in enumerate(feature_cols):
        coef = coefs[i]
        ratio = abs(normalized[i])
        cur = current_max_pts.get(col, 0)
        # 제안 배점: 기여비율 × 전체 가용점수(~12점)로 환산
        suggested = round(ratio * 12.0, 1)
        # 방향 반영: 음의 계수면 음의 배점 유지
        if coef < 0:
            suggested = -suggested
        change = suggested - cur if isinstance(cur, (int, float)) else '-'
        suggestions[col] = suggested

        cur_str = f'{cur}' if isinstance(cur, (int, float)) else cur
        chg_str = f'{change:>+.1f}' if isinstance(change, (int, float)) else '-'
        direction = '↑' if isinstance(change, (int, float)) and change > 0.3 else ('↓' if isinstance(change, (int, float)) and change < -0.3 else '≈')
        print(f"  {col:<22} {coef:>+8.4f} {ratio:>7.1%}  {cur_str:>8} {suggested:>+8.1f} {chg_str:>6} {direction}")

    # ── 이중 가산 분석 ──
    print(f"\n{'─'*65}")
    print(f"  [이중 가산 분석 — compositeRankScore vs buyScoreV2]")
    print(f"{'─'*65}")
    print(f"""
  compositeRankScore 구조 (100점 만점):
  ┌─────────────────────┬────────┬───────────────────────────┐
  │ 항목                │ 배점   │ buyScore 내 중복          │
  ├─────────────────────┼────────┼───────────────────────────┤
  │ 1. 매수시그널 강도  │ 40점   │ = buyScore 전체 (중복X)   │
  │ 2. RSI 모멘텀       │ 20점   │ RSI +2.5점 → 이중 가산   │
  │ 3. 추세 건전성      │ 15점   │ MA정배열+일목 → 이중 가산 │
  │ 4. 세력감지(벤포드) │ 15점   │ 벤포드 ×곱셈 → 이중 가산 │
  │ 5. 변동성(ATR)      │ 10점   │ 독립 (buyScore에 없음)   │
  └─────────────────────┴────────┴───────────────────────────┘""")

    # buyScore 내 RSI 기여도 추정
    rsi_coef_abs = abs(coefs[feature_cols.index('rsi')])
    total_coef_abs = sum(abs(c) for c in coefs)
    rsi_pct_in_buy = rsi_coef_abs / total_coef_abs * 100 if total_coef_abs > 0 else 0

    # composite에서 RSI가 차지하는 비율
    # buyScore(40점)에서 RSI 기여분 + RSI 독립 배점(20점)
    rsi_in_composite_via_buy = 40 * (rsi_pct_in_buy / 100)
    rsi_total_in_composite = rsi_in_composite_via_buy + 20

    print(f"\n  RSI 이중 가산 영향:")
    print(f"    buyScore 내 RSI 기여도: {rsi_pct_in_buy:.1f}%")
    print(f"    composite 내 RSI (buyScore 경유): {rsi_in_composite_via_buy:.1f}점/100점")
    print(f"    composite 내 RSI (독립 배점):     20.0점/100점")
    print(f"    composite 내 RSI 총 영향력:       {rsi_total_in_composite:.1f}점/100점 ({rsi_total_in_composite:.0f}%)")

    # 일목 이중 가산
    ichi_cols = ['above_cloud', 'tenkan_gt_kijun', 'cloud_bullish']
    ichi_coef_abs = sum(abs(coefs[feature_cols.index(c)]) for c in ichi_cols if c in feature_cols)
    ichi_pct_in_buy = ichi_coef_abs / total_coef_abs * 100 if total_coef_abs > 0 else 0
    ichi_in_composite_via_buy = 40 * (ichi_pct_in_buy / 100)
    # composite 추세건전성 15점 중 구름위(4점) + 기준선위(3점) = 7점이 일목
    ichi_total = ichi_in_composite_via_buy + 7

    print(f"\n  일목균형 이중 가산 영향:")
    print(f"    buyScore 내 일목 기여도: {ichi_pct_in_buy:.1f}%")
    print(f"    composite 내 일목 (buyScore 경유): {ichi_in_composite_via_buy:.1f}점/100점")
    print(f"    composite 내 일목 (추세건전성):    7.0점/100점")
    print(f"    composite 내 일목 총 영향력:       {ichi_total:.1f}점/100점 ({ichi_total:.0f}%)")

    return suggestions, scaler, lr, feature_cols


# ═══════════════════════════════════════════════════════════════
# PHASE 3: Out-of-Sample 검증
# ═══════════════════════════════════════════════════════════════
def phase3_analysis(df_all, suggestions, feature_cols):
    print(f"\n\n{'=' * 65}")
    print("  PHASE 3: Out-of-Sample 검증")
    print("=" * 65)

    # 시간 분할
    df_all['entry_ts'] = pd.to_datetime(df_all['entry_date'])
    df_train = df_all[df_all['entry_ts'] < OOS_CUTOFF]
    df_test = df_all[df_all['entry_ts'] >= OOS_CUTOFF]

    print(f"\n  학습 기간: ~ {OOS_CUTOFF.strftime('%Y-%m-%d')} ({len(df_train)}건)")
    print(f"  검증 기간: {OOS_CUTOFF.strftime('%Y-%m-%d')} ~ ({len(df_test)}건)")

    if len(df_test) < 10:
        print(f"\n  ⚠ OOS 데이터 부족 ({len(df_test)}건). 검증 불가.")
        print(f"  → 전체 데이터 기반 분석 결과만 참고하세요.")
        return

    # 학습 데이터로 회귀 학습
    df_train_clean = df_train[feature_cols + ['win', 'return_pct']].dropna()
    df_test_clean = df_test[feature_cols + ['win', 'return_pct']].dropna()

    if len(df_train_clean) < 30 or len(df_test_clean) < 10:
        print(f"\n  ⚠ 결측 제거 후 데이터 부족. 검증 불가.")
        return

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train_clean[feature_cols].values)
    y_train = df_train_clean['win'].values

    X_test = scaler.transform(df_test_clean[feature_cols].values)
    y_test = df_test_clean['win'].values

    # 현재 배점 기준 성과 (test 데이터)
    current_wr = df_test_clean['win'].mean() * 100
    current_ev = df_test_clean['return_pct'].mean()

    # 최적 배점 기준 성과 (회귀 모델로 필터링)
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_train, y_train)

    # 모델 예측 확률 기반 필터링
    proba = lr.predict_proba(X_test)[:, 1]  # WIN 확률

    # 상위 확률 종목만 선별 (기존 대비 얼마나 개선되는지)
    thresholds = [0.3, 0.4, 0.5, 0.6]

    print(f"\n{'─'*65}")
    print(f"  [현재 배점 성과 (OOS: {OOS_CUTOFF.strftime('%Y')}~)]")
    print(f"{'─'*65}")
    print(f"    전체 거래: {len(df_test_clean)}건")
    print(f"    승률: {current_wr:.1f}%")
    print(f"    평균 수익률: {current_ev:+.2f}%")

    print(f"\n{'─'*65}")
    print(f"  [최적 배점 모델 필터링 효과 (OOS)]")
    print(f"{'─'*65}")
    print(f"  {'WIN확률 임계':>12} {'거래수':>6} {'승률':>8} {'평균수익':>10} {'개선':>8}")
    print(f"  {'─'*50}")

    for th in thresholds:
        mask = proba >= th
        if mask.sum() < 5:
            continue
        sub = df_test_clean.iloc[mask.nonzero()[0]]
        wr = sub['win'].mean() * 100
        avg_ret = sub['return_pct'].mean()
        improve = avg_ret - current_ev
        print(f"  {f'>={th:.0%}':>12} {mask.sum():>6} {wr:>7.1f}% {avg_ret:>+9.2f}% {improve:>+7.2f}%")

    # OOS 학습 정확도
    oos_acc = lr.score(X_test, y_test)
    print(f"\n  OOS 예측 정확도: {oos_acc*100:.1f}%")

    # ── buyScore 구간별 OOS 성과 ──
    print(f"\n{'─'*65}")
    print(f"  [buyScore 구간별 OOS 성과]")
    print(f"{'─'*65}")
    if 'buy_score' in df_test.columns:
        for lo, hi in [(4, 6), (6, 8), (8, 10), (10, 12), (12, 20)]:
            sub = df_test[(df_test['buy_score'] >= lo) & (df_test['buy_score'] < hi)]
            if len(sub) >= 3:
                wr = sub['win'].mean() * 100
                avg_ret = sub['return_pct'].mean()
                print(f"    score {lo:>2}-{hi:<2}: 승률 {wr:5.1f}% | 평균수익 {avg_ret:>+6.2f}% | {len(sub):>3}건")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 65)
    print("  배점 검증 및 최적화 분석")
    print("  195개 종목 × 전체 기간 백테스트 데이터 기반")
    print("=" * 65)
    print("\n데이터 수집 중...")

    df_all = collect_all_trades_with_features()

    if len(df_all) < 50:
        print(f"\n⚠ 수집된 거래가 {len(df_all)}건으로 분석 불가 (최소 50건 필요)")
        sys.exit(1)

    # Phase 1
    corr_results = phase1_analysis(df_all)

    # Phase 2
    suggestions, scaler, lr, feature_cols = phase2_analysis(df_all)

    # Phase 3
    phase3_analysis(df_all, suggestions, feature_cols)

    # ── 최종 요약 ──
    print(f"\n\n{'=' * 65}")
    print("  최종 요약 및 권고사항")
    print("=" * 65)
    print("""
  분석 완료. 위 결과를 바탕으로:
  1. 상관계수가 높고 통계적으로 유의한 지표 → 배점 유지/상향
  2. 상관계수가 낮거나 음수인 지표 → 배점 하향/제거 검토
  3. compositeRankScore의 이중 가산 → 독립 배점 축소 또는 제거 검토
  4. OOS 검증에서 개선이 확인된 경우에만 실제 배점 변경 적용
""")
