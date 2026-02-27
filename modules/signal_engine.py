import numpy as np
import pandas as pd
from modules.benford import (
    analyze_volume_benford,
    analyze_price_change_benford,
    is_near_psychological_level,
)


# ============================================================
# 종목 유형별 프로필 — 대형주와 중소형주는 변동성/패턴이 다름
# ============================================================
STOCK_PROFILES = {
    'default': {  # 중소형 성장주 (카카오페이, 셀트리온 등)
        'high_dist_max': 0.05,    # 20일 고점 대비 최대 거리 (5%)
        'ma20_slope_min': 0.02,   # MA20 10일 최소 기울기 (2%)
        'vol_overheat': 8.0,      # 거래량 과열 상한 (8배)
        'rsi_optimal': (50, 65),  # RSI 최적 구간
        'rsi_forming': (45, 50),  # RSI 모멘텀 형성 구간
        'ret20d_strong': 0.15,    # 20일 수익 강한 모멘텀 (15%)
        'ret20d_mid': 0.10,       # 20일 수익 중간 (10%)
        'ret20d_weak': 0.05,      # 20일 수익 약한 (5%)
        'take_profit': 0.17,      # 추천 익절 (17%)
        'stop_loss': 0.07,        # 추천 손절 (7%)
        'cooldown': 3,            # 추천 쿨다운
        'benford_weight': 0.10,   # 벤포드 최대 가중치 (15%)
    },
    'large_cap': {  # 대형 우량주 (삼성전자, 현대차, SK하이닉스 등)
        'high_dist_max': 0.07,    # 대형주는 고점 대비 7%까지 허용
        'ma20_slope_min': 0.01,   # 대형주는 기울기 1%면 충분
        'vol_overheat': 6.0,      # 대형주는 6배 이상이면 과열
        'rsi_optimal': (45, 60),  # 대형주는 더 낮은 RSI에서도 건강
        'rsi_forming': (40, 45),  # 대형주 모멘텀 형성 구간
        'ret20d_strong': 0.08,    # 대형주는 8%면 강한 모멘텀
        'ret20d_mid': 0.05,       # 대형주는 5%가 중간
        'ret20d_weak': 0.03,      # 대형주는 3%도 의미
        'take_profit': 0.10,      # 추천 익절 (10%)
        'stop_loss': 0.10,        # 추천 손절 (10%)
        'cooldown': 5,            # 추천 쿨다운
        'benford_weight': 0.10,   # 벤포드 최대 가중치 (15%)
    },
    'force_following': {  # 세력 추종형 — 벤포드 이상 + 거래량 급증 중심
        'high_dist_max': 0.08,    # 고점 대비 8%까지 허용 (매집 초기 포착)
        'ma20_slope_min': 0.01,   # MA20 기울기 완화 (매집 초기는 기울기 약함)
        'vol_overheat': 10.0,     # 거래량 10배까지 허용 (세력 진입시 폭발적 거래량)
        'rsi_optimal': (45, 70),  # RSI 범위 확대 (세력은 과매수 구간도 밀어붙임)
        'rsi_forming': (40, 45),  # 모멘텀 형성 구간
        'ret20d_strong': 0.12,    # 20일 강한 모멘텀 기준
        'ret20d_mid': 0.07,       # 20일 중간 기준
        'ret20d_weak': 0.03,      # 20일 약한 기준
        'take_profit': 0.21,      # 익절 +21% (세력 움직임은 큼)
        'stop_loss': 0.07,        # 손절 -7% 유지 (R:R = 1:3)
        'cooldown': 5,            # 쿨다운 5일 (세력 재진입 주기 고려)
        'benford_weight': 0.25,   # 벤포드 최대 가중치 대폭 상향 (35%)
    },
}


def get_profile(profile_name='default'):
    """종목 프로필 반환"""
    return STOCK_PROFILES.get(profile_name, STOCK_PROFILES['default'])


def calculate_buy_score(df, idx, benford_window=30, profile_name='default'):
    """
    매수 시그널 v6 — 종목 유형별 프로필 지원

    프로필별로 필수 조건의 임계값이 달라짐:
    - default(중소형주): 변동성 크고 성장 빠름 → 엄격한 기준
    - large_cap(대형주): 변동성 작고 안정적 → 완화된 기준

    필수 조건 (ALL):
      1. 정배열: MA5 > MA20 > MA60
      2. 종가가 20일 고점의 N% 이상 (프로필별)
      3. MA20이 10일 전 대비 N% 이상 상승 (프로필별)
      4. 양봉 (close > open)
      5. 거래량 < Nx 평균 (프로필별)
    """
    if idx < 60:
        return 0.0, {}

    p = get_profile(profile_name)
    row = df.iloc[idx]
    prev = df.iloc[idx - 1]
    details = {}

    ma5 = row.get('ma_5')
    ma20 = row.get('ma_20')
    ma60 = row.get('ma_60')

    # ============================================================
    # 필수 조건 5가지 (프로필별 임계값)
    # ============================================================

    # 1. 정배열 (MA5 > MA20 > MA60)
    if not (pd.notna(ma5) and pd.notna(ma20) and pd.notna(ma60)):
        return 0.0, {}
    if not (ma5 > ma20 > ma60):
        return 0.0, {}

    # 2. 20일 고점 근접 (프로필별 범위)
    high_20d = df['high'].iloc[max(0, idx - 20):idx + 1].max()
    if high_20d <= 0:
        return 0.0, {}
    dist_from_high = (high_20d - row['close']) / high_20d
    if dist_from_high > p['high_dist_max']:
        return 0.0, {}

    # 3. MA20 상승 (프로필별 최소 기울기)
    if idx < 10:
        return 0.0, {}
    ma20_10ago = df.iloc[idx - 10].get('ma_20')
    if not pd.notna(ma20_10ago) or ma20_10ago <= 0:
        return 0.0, {}
    ma20_slope = (ma20 - ma20_10ago) / ma20_10ago
    if ma20_slope < p['ma20_slope_min']:
        return 0.0, {}

    # 4. 양봉
    if row['close'] <= row['open']:
        return 0.0, {}

    # 5. 거래량 과열 차단 (프로필별 상한)
    vol_ratio = row.get('vol_ratio')
    if pd.notna(vol_ratio) and vol_ratio > p['vol_overheat']:
        return 0.0, {}

    # ============================================================
    # 필수 조건 모두 통과 → 스코어링 (기본 5.0점)
    # ============================================================
    score = 5.0
    details['core'] = f'정배열,고점{dist_from_high*100:.1f}%,MA20↑{ma20_slope*100:.1f}%'

    # === 1. RSI 구간별 점수 (프로필별) ===
    rsi = row.get('rsi')
    rsi_lo, rsi_hi = p['rsi_optimal']
    rsi_form_lo, rsi_form_hi = p['rsi_forming']
    if pd.notna(rsi):
        if rsi_lo <= rsi <= rsi_hi:
            score += 2.0
            details['rsi'] = f'건강모멘텀({rsi:.0f})'
        elif rsi_form_lo <= rsi < rsi_lo:
            score += 1.0
            details['rsi'] = f'모멘텀형성({rsi:.0f})'
        elif rsi_hi < rsi <= 80:
            score += 0.5
            details['rsi'] = f'강모멘텀({rsi:.0f})'
        elif rsi > 80:
            score += 0.0
            details['rsi'] = f'과매수({rsi:.0f})'

    # === 2. MACD 상태 ===
    macd = row.get('macd')
    macd_signal = row.get('macd_signal')
    macd_hist = row.get('macd_hist')
    prev_macd_hist = prev.get('macd_hist')

    if pd.notna(macd_hist) and macd_hist > 0:
        score += 1.0
        if pd.notna(prev_macd_hist) and macd_hist > prev_macd_hist:
            score += 0.5
            details['macd'] = 'MACD가속'
        else:
            details['macd'] = 'MACD양전'

    prev_macd = prev.get('macd')
    prev_macd_signal = prev.get('macd_signal')
    if all(pd.notna(v) for v in [macd, macd_signal, prev_macd, prev_macd_signal]):
        if macd > macd_signal and prev_macd <= prev_macd_signal:
            score += 1.5
            details['macd'] = 'MACD골든크로스'

    # === 3. 20일 수익률 (프로필별 기준) ===
    if idx >= 20:
        price_20ago = df.iloc[idx - 20]['close']
        if price_20ago > 0:
            ret_20d = (row['close'] - price_20ago) / price_20ago
            if ret_20d > p['ret20d_strong']:
                score += 1.5
                details['ret20d'] = f'+{ret_20d*100:.0f}%'
            elif ret_20d > p['ret20d_mid']:
                score += 1.0
                details['ret20d'] = f'+{ret_20d*100:.0f}%'
            elif ret_20d > p['ret20d_weak']:
                score += 0.5

    # === 4. 거래량 품질 ===
    if pd.notna(vol_ratio):
        if profile_name == 'force_following':
            # 세력 추종형: 거래량 급증 구간을 적극적으로 선호
            if 3.0 <= vol_ratio <= 7.0:
                score += 2.0  # 세력 진입 핵심 신호
                details['volume'] = f'세력거래량(x{vol_ratio:.1f})'
            elif 1.5 <= vol_ratio < 3.0:
                score += 1.0
                details['volume'] = f'증가거래량(x{vol_ratio:.1f})'
            elif 7.0 < vol_ratio <= p['vol_overheat']:
                score += 0.5  # 세력형은 초과열도 긍정 신호
                details['volume'] = f'폭발거래량(x{vol_ratio:.1f})'
        else:
            if 1.3 <= vol_ratio <= 3.0:
                score += 1.0
                details['volume'] = f'건전거래량(x{vol_ratio:.1f})'
            elif 3.0 < vol_ratio <= 5.0:
                score += 0.5
                details['volume'] = f'높은거래량(x{vol_ratio:.1f})'
            elif 5.0 < vol_ratio <= p['vol_overheat']:
                score -= 0.5
                details['volume'] = f'과열주의(x{vol_ratio:.1f})'

    # === 5. 캔들/패턴 ===
    if row.get('is_bullish_engulfing', False):
        score += 1.0
        details['candle'] = '상승장악형'

    # 연속 상승
    consec = 0
    for lb in range(1, min(8, idx + 1)):
        if df.iloc[idx - lb + 1]['close'] > df.iloc[idx - lb]['close']:
            consec += 1
        else:
            break
    if 2 <= consec <= 5:
        score += 0.5
        details['streak'] = f'{consec}일연속↑'
    elif consec > 5:
        score -= 0.5  # 너무 연속 상승 = 조정 임박

    # === 6. 볼린저밴드 ===
    bb_mid = row.get('bb_mid')
    if pd.notna(bb_mid) and row['close'] > bb_mid:
        score += 0.3

    # === 7. 20일 신고가 ===
    if row['close'] >= high_20d * 0.99:  # 거의 신고가
        score += 0.5
        details['breakout'] = '20일신고가'

    # === 8. MA60 기울기 (장기 추세 확인) ===
    if idx >= 20:
        ma60_20ago = df.iloc[idx - 20].get('ma_60')
        if pd.notna(ma60_20ago) and ma60_20ago > 0:
            ma60_slope = (ma60 - ma60_20ago) / ma60_20ago
            if ma60_slope > 0.02:
                score += 1.0
                details['long_trend'] = f'MA60도상승(+{ma60_slope*100:.1f}%)'

    # === 9. 벤포드 법칙 ===
    bw = p.get('benford_weight', 0.10)
    if profile_name == 'force_following':
        # 세력 추종형: 단기(15일) + 장기(60일) 이중 윈도우
        short_w = min(15, benford_window)
        long_w = min(60, idx + 1)
        vol_s, _ = analyze_volume_benford(
            df['volume'].iloc[max(0, idx - short_w):idx + 1].values, window=short_w)
        vol_l, _ = analyze_volume_benford(
            df['volume'].iloc[max(0, idx - long_w):idx + 1].values, window=long_w)
        pc_s, _ = analyze_price_change_benford(
            df['close'].iloc[max(0, idx - short_w):idx + 1].values, window=short_w)
        # 단기 이탈도 2배 가중 (최근 세력 흔적 강조)
        combined = (vol_s * 2 + vol_l + pc_s) / 4
        benford_mult = 1.0 + min(combined * bw * 2, bw * 2)
    else:
        volumes = df['volume'].iloc[max(0, idx - benford_window):idx + 1].values
        vol_bscore, _ = analyze_volume_benford(volumes, window=benford_window)
        prices = df['close'].iloc[max(0, idx - benford_window):idx + 1].values
        pc_bscore, _ = analyze_price_change_benford(prices, window=benford_window)
        benford_mult = 1.0 + min((vol_bscore + pc_bscore) * 0.1, 0.15)

    if benford_mult > 1.03:
        details['benford'] = f'벤포드(x{benford_mult:.2f})'

    final_score = score * benford_mult
    return final_score, details
