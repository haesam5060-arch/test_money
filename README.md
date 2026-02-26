# 주식 백테스트 및 신호 분석 시스템

벤포드 법칙과 기술적 지표를 결합한 한국 주식 매매 신호 생성 및 백테스트 도구입니다.

---

## 주요 기능

- **매수 신호 생성**: 벤포드 법칙 이상 탐지 + RSI, MACD, 볼린저밴드, 거래량 등 기술적 지표 복합 분석
- **백테스트**: Walk-forward 방식으로 익절(+21%) / 손절(-7%) 시뮬레이션
- **대규모 최적화**: 192개 KOSPI 종목 대상 파라미터 최적화 (평균 승률 77.7%)
- **승률 분석**: 종목별 성과 분석 및 시각화 (`index.html`)

---

## 파일 구조

```
연구/
├── main.py                  # 메인 실행 파일 (단일 종목 분석)
├── optimize_scoring.py      # 대규모 파라미터 최적화
├── analyze_exit_params.py   # 익절/손절 파라미터 분석
├── analyze_winners.py       # 고승률 종목 분석
├── index.html               # 결과 시각화 대시보드
├── optimization_report.txt  # 최적화 결과 리포트
├── optimization_report.json # 최적화 결과 (JSON)
├── modules/
│   ├── data_parser.py       # XML 주가 데이터 파싱
│   ├── indicators.py        # 기술적 지표 계산
│   ├── signal_engine.py     # 매수 신호 엔진
│   ├── backtester.py        # 백테스트 엔진
│   └── benford.py           # 벤포드 법칙 분석
└── xml/                     # 종목별 주가 데이터 (200개+)
```

---

## 사용 방법

### 단일 종목 분석
```bash
python main.py xml/005930.xml
```

### 파라미터 최적화
```bash
python optimize_scoring.py
```

### 승률 분석
```bash
python analyze_winners.py
```

---

## 백테스트 결과 요약

| 항목 | 값 |
|------|-----|
| 분석 종목 수 | 192개 |
| 평균 승률 | 77.7% |
| 테스트 승률 | 79.7% |
| 총 거래 수 | 57,704건 |
| 일관성 (≥60% 종목) | 100% |

> 기본 파라미터: 매수 임계값 4.0 / 익절 +21% / 손절 -7% / 쿨다운 5일

---

## 사용 데이터

- 출처: 한국거래소(KRX) KOSPI 종목 XML 데이터
- 포함 종목: 삼성전자, SK하이닉스, LG에너지솔루션 등 200개+
