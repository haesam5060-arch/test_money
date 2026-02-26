import pandas as pd
import numpy as np


def calc_moving_averages(df, windows=[5, 20, 60]):
    """이동평균선 계산"""
    for w in windows:
        df[f'ma_{w}'] = df['close'].rolling(window=w).mean()
    return df


def calc_rsi(df, period=14):
    """RSI (Relative Strength Index) - Wilder 방식"""
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # 첫 번째 평균: SMA
    avg_gain = pd.Series(np.nan, index=df.index, dtype=float)
    avg_loss = pd.Series(np.nan, index=df.index, dtype=float)

    avg_gain.iloc[period] = gain.iloc[1:period + 1].mean()
    avg_loss.iloc[period] = loss.iloc[1:period + 1].mean()

    # Wilder 지수이동평균
    for i in range(period + 1, len(df)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df


def calc_bollinger_bands(df, window=20, num_std=2):
    """볼린저밴드 계산"""
    df['bb_mid'] = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_mid'] + num_std * rolling_std
    df['bb_lower'] = df['bb_mid'] - num_std * rolling_std
    return df


def calc_volume_ratio(df, window=20):
    """거래량 비율 (현재 거래량 / N일 평균 거래량)"""
    df['vol_avg'] = df['volume'].rolling(window=window).mean()
    df['vol_ratio'] = df['volume'] / df['vol_avg']
    return df


def calc_macd(df, fast=12, slow=26, signal=9):
    """MACD 계산"""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df


def detect_candle_patterns(df):
    """캔들스틱 패턴 감지"""
    body = df['close'] - df['open']
    body_abs = body.abs()
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']

    # 평균 몸통 크기 (기준값)
    avg_body = body_abs.rolling(window=20, min_periods=5).mean()

    # 망치형 (Hammer): 아래꼬리가 몸통의 2배 이상, 윗꼬리 작음
    df['is_hammer'] = (
        (lower_shadow > 2 * body_abs) &
        (upper_shadow < body_abs * 0.5) &
        (body_abs > 0)
    )

    # 상승장악형 (Bullish Engulfing): 전일 음봉을 오늘 양봉이 감싸는 패턴
    df['is_bullish_engulfing'] = (
        (body > 0) &
        (body.shift(1) < 0) &
        (df['open'] <= df['close'].shift(1)) &
        (df['close'] >= df['open'].shift(1))
    )

    # 유성형 (Shooting Star): 윗꼬리가 몸통 2배 이상
    df['is_shooting_star'] = (
        (upper_shadow > 2 * body_abs) &
        (lower_shadow < body_abs * 0.5) &
        (body_abs > 0)
    )

    # 하락장악형 (Bearish Engulfing)
    df['is_bearish_engulfing'] = (
        (body < 0) &
        (body.shift(1) > 0) &
        (df['open'] >= df['close'].shift(1)) &
        (df['close'] <= df['open'].shift(1))
    )

    # 도지 (Doji): 몸통이 매우 작음
    df['is_doji'] = body_abs < (avg_body * 0.1)

    return df


def calc_all_indicators(df):
    """모든 기술지표를 한번에 계산"""
    df = calc_moving_averages(df)
    df = calc_rsi(df)
    df = calc_bollinger_bands(df)
    df = calc_volume_ratio(df)
    df = calc_macd(df)
    df = detect_candle_patterns(df)
    return df
