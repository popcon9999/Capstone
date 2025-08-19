import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pvlib
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

# --- 한글 폰트 설정 ---
# 맑은 고딕(Malgun Gothic) 폰트를 사용하도록 설정합니다. (윈도우 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지
# ----------------------

def load_and_process_data(power_filepaths, weather_filepaths):
    """
    여러 개의 발전량 CSV와 날씨 CSV를 불러와 하나의 데이터프레임으로 통합하고 전처리합니다.
    """
    print("--- 1. 데이터 통합 및 전처리 시작 ---")
    
    df_list = []
    
    for filepath in power_filepaths:
        try:
            print(f"\n>>> 파일 처리 시도: {filepath}")
            df = pd.read_csv(filepath, encoding='cp949', header=0, skipinitialspace=True)
            print("    - Pandas로 데이터프레임 로딩 성공!")
            
            df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
            if '발전구분' in df.columns:
                df['발전구분'] = df['발전구분'].str.strip()
                
                print(f"    - '발전구분' 컬럼의 고유값: {df['발전구분'].unique()}")
                
                df_gyeongsang = df[df['발전구분'] == '경상대태양광'].copy()
                if not df_gyeongsang.empty:
                    print("    -> '경상대태양광' 데이터 찾음!")
                    df_list.append(df_gyeongsang)
                else:
                    print("    -> '경상대태양광' 데이터를 찾지 못함.")
            else:
                print("    -> '발전구분' 컬럼을 찾을 수 없습니다.")

        except Exception as e:
            print(f"    - 파일 처리 중 심각한 오류 발생: {e}")
            
    if not df_list:
        print("\n최종적으로 처리할 발전량 데이터가 없습니다.")
        return pd.DataFrame()

    combined_power_df = pd.concat(df_list, ignore_index=True)
    
    id_vars = ['일자']
    value_vars = [f'{i}시 발전량(MWh)' for i in range(1, 25)]
    power_long_df = pd.melt(combined_power_df, id_vars=id_vars, value_vars=value_vars, 
                            var_name='시간', value_name='발전량(MWh)')

    power_long_df['시간'] = power_long_df['시간'].str.extract('(\d+)').astype(int)
    power_long_df['일자'] = pd.to_datetime(power_long_df['일자'], errors='coerce')
    power_long_df.loc[power_long_df['시간'] == 24, '일자'] += pd.to_timedelta(1, unit='d')
    power_long_df.loc[power_long_df['시간'] == 24, '시간'] = 0
    power_long_df['일시'] = power_long_df.apply(lambda row: row['일자'].replace(hour=row['시간']), axis=1)
    
    weather_df_list = []
    for filepath in weather_filepaths:
        try:
            df = pd.read_csv(filepath, encoding='cp949')
            weather_df_list.append(df)
        except Exception as e:
            print(f"날씨 파일 로딩 중 오류 발생: {filepath} - {e}")
            
    if not weather_df_list:
        print("처리할 날씨 데이터가 없습니다.")
        return pd.DataFrame()
    
    weather_df = pd.concat(weather_df_list, ignore_index=True)
    
    weather_df.rename(columns={'전운량(10분위)': '운량'}, inplace=True)
    weather_df['일시'] = pd.to_datetime(weather_df['일시'])
    
    # --- 🔧 [핵심 수정] Merge 전, 두 '일시' 컬럼의 형식을 동일하게 만듭니다. ---
    # 초(second) 정보를 모두 제거하여 'YYYY-MM-DD HH:MM' 형식으로 통일
    power_long_df['일시'] = power_long_df['일시'].dt.floor('T')
    weather_df['일시'] = weather_df['일시'].dt.floor('T')
    # --------------------------------------------------------------------

    # 5. 발전량 데이터와 날씨 데이터 병합
    final_df = pd.merge(power_long_df[['일시', '발전량(MWh)']], weather_df, on='일시', how='inner')
    
    final_df.sort_values(by='일시', inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    print(f"--- 데이터 통합 완료. 최종 데이터 수: {len(final_df)}개 ---")
    
    # [디버깅] 병합 후 데이터가 어떻게 생겼는지 확인
    print("\n--- 병합 후 데이터 샘플 ---")
    print(final_df.head())
    
    return final_df

def feature_engineering(df):
    """AI가 학습하기 좋은 특징(Feature)들을 생성합니다."""
    print("--- 2. 특징 공학 시작 ---")
    
    # 기본 전처리
    columns_to_drop = [
        '지점', '지점명', '기온 QC플래그', '습도 QC플래그', 
        '일조 QC플래그', '일사 QC플래그', '운형(운형약어)'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    # 🔧 [수정 1] 결측치 처리 방식 변경: 0으로 채우기 전에 이전 값으로 채우기
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)

    # 기본 시간 특징
    df['hour'] = df['일시'].dt.hour
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['month'] = df['일시'].dt.month
    df['season'] = (df['month'] % 12 + 3) // 3 
    
    # --- 🔧 [개선 1] pvlib를 이용한 물리 모델 특징 추가 ---
    # 진주시 위치 정보 (위도, 경도, 고도)
    location = pvlib.location.Location(latitude=35.19, longitude=128.08, altitude=30, tz='Asia/Seoul')
    
    # '일시' 컬럼을 인덱스로 설정하고, 서울 시간대를 명시적으로 부여합니다.
    times = pd.DatetimeIndex(df['일시']).tz_localize('Asia/Seoul')
    
    # 시간대 정보가 포함된 times를 사용하여 태양 위치와 청천 일사량을 계산합니다.
    solar_position = location.get_solarposition(times)
    clear_sky = location.get_clearsky(times, model='ineichen')
    
    # 계산된 결과를 원래 데이터프레임에 다시 할당합니다.
    df['solar_azimuth'] = solar_position['azimuth'].values
    df['solar_altitude'] = solar_position['apparent_elevation'].values
    df['clearsky_ghi'] = clear_sky['ghi'].values
    
    # --- ✨ [업그레이드] 청명도 지수(Cloudiness Index) 특징 추가 ---
    # '일사(MJ/m2)'는 시간당 에너지, 'clearsky_ghi'는 순간 파워(W/m2)이므로 단위를 맞춰줍니다.
    # 1 MJ/hr = 1,000,000 J / 3600 s = 277.7 W
    MJ_to_W_conversion_factor = 1000000 / 3600
    df['ghi_actual_watt'] = df['일사(MJ/m2)'] * MJ_to_W_conversion_factor
    
    # 청명도 지수 계산 (0으로 나누는 오류 방지)
    # clearsky_ghi가 0보다 클 때만 계산하고, 나머지는 0으로 채웁니다.
    df['cloudiness_index'] = np.where(
        df['clearsky_ghi'] > 0,
        df['ghi_actual_watt'] / df['clearsky_ghi'],
        0
    )
    # 불가능한 값(1 초과)은 1로 제한합니다. (측정 오차 등 보정)
    df['cloudiness_index'] = df['cloudiness_index'].clip(0, 1)

    # 계산에 사용된 중간 컬럼은 삭제
    df.drop(columns=['ghi_actual_watt'], inplace=True)

    print("청명도 지수(cloudiness_index) 특징 추가 완료!")
    
    # feature_engineering 함수에 추가
    # 일사량의 3시간 이동 표준편차를 계산하여 '변동성' 특징으로 활용
    df['irradiance_volatility_3h'] = df['일사(MJ/m2)'].rolling(window=3).std()
    
    # 초반 결측치는 이전 값으로 채우기
    df['irradiance_volatility_3h'].fillna(method='ffill', inplace=True)

    # 주기성 특징
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    print("\n--- 이상치 제거 시작 ---")
    
    clearsky_threshold = 200
    power_threshold = 1.0
    
    # ❗ [수정] df를 사용하고, 제거 후 결과를 다시 df에 할당해야 합니다.
    abnormal_indices = df[
        (df['clearsky_ghi'] > clearsky_threshold) &
        (df['발전량(MWh)'] < power_threshold)
    ].index
    print(f">>> 제거 대상 이상치 {len(abnormal_indices)}개 발견")
    
    abnormal_indices2 = df[
        (df['clearsky_ghi'] > 600) &      # 매우 맑은 날 (정오 부근)
        (df['cloudiness_index'] < 0.25)   # 발전 효율이 25% 미만인 경우
    ].index
    print(f">>> 규칙 2 (낮은 효율) 대상: {len(abnormal_indices2)}개")
    
    abnormal_indices3 = abnormal_indices.union(abnormal_indices2)
    print(f">>> 최종 제거 대상 이상치 {len(abnormal_indices3)}개 발견")
    
    
    df = df.drop(index=abnormal_indices3)
    
    print(f">>> 제거 후 데이터 수: {len(df)}개")
    print("--- 이상치 제거 완료 ---\n")
    

    # 과거 데이터 특징 (Lag Features)
    for i in range(1, 8):
        df[f'power_lag_{i * 24}h'] = df['발전량(MWh)'].shift(i * 24)
    
    for i in range(1, 7): # 1, 2, 3시간 전 Lag 추가
        df[f'power_lag_{i}h'] = df['발전량(MWh)'].shift(i)
    
    def q25(x):
        return x.quantile(0.25)
    def q75(x):
        return x.quantile(0.75)
    
    # 이동평균 특징 추가
    df['temp_roll_mean_3h']= df['기온(°C)'].rolling(window=3, min_periods=1).mean()
    df['humidity_roll_mean_3h']=df['습도(%)'].rolling(window=3, min_periods=1).mean()
    df['cloud_roll_mean_3h']=df['운량'].rolling(window=3, min_periods=1).mean()
    
    df['power_roll_mean_3h']=df['발전량(MWh)'].shift(1).rolling(window=3, min_periods=1).mean()

    # 계절/시간별 통계 특징
    seasonal_hourly_stats = df.groupby(['season', 'hour'])['발전량(MWh)'].agg(['mean', 'median',q25, q75]).reset_index()
    seasonal_hourly_stats.columns=['season', 'hour', 'seasonal_hourly_mean', 'seasonal_hourly_median', 'seasonal_hourly_q25', 'seasonal_hourly_q75']

    df['season'] = df['season'].astype('int64')
    df['hour'] = df['hour'].astype('int64')
    seasonal_hourly_stats['season'] = seasonal_hourly_stats['season'].astype('int64')
    seasonal_hourly_stats['hour'] = seasonal_hourly_stats['hour'].astype('int64')

    df = pd.merge(df, seasonal_hourly_stats, on=['season', 'hour'], how='left')

    df.dropna(inplace=True)
    print("--- 특징 공학 완료 ---\n")
    print(f"--- 최종 결측치 처리 후 데이터 수: {len(df)} ---")
    
    return df


# --- 메인 실행 부분 ---
if __name__ == '__main__':
    power_files = [
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년7월3일~9).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년7월10~16).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년7월17~23).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년7월24~30).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년7월31일~6일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년8월7~13).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년8월14~20).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년8월21~27).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년8월28~3).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년9월4~10).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년9월11~17).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년9월18~24).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년9월25일~1).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년10월2~8).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년10월9~15).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년10월16~22).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년10월23~29).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년10월30일~5일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년11월6~12).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년11월13~19).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년11월20~26).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년11월27~3).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년12월4일~10).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년12월11~17).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년12월18~24).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(24년12월25~31일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(1월1일~1월6일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(1월7일~1월13일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(1월14일~1월20일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(1월21일~1월27일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(1월28일~2월3일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(2월4일~2월10일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(2월11일~2월17일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(2월18일~2월24일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(2월25일~3월3일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(3월4일~3월10일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(3월11일~3월17일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(3월18일~3월24일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(3월25일~3월31일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(4월1일~4월7일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(4월8일~4월14일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(4월15일~4월21일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(4월22일~4월28일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(4월29일~5월5일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(5월6일~5월12일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(5월13일~5월19일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(5월20일~5월26일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(5월27일~6월2일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(6월3일~6월9일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(6월10일~6월16일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(6월17일~6월23일).csv',
        'c:/Users/ms/Desktop/학습자료/한국남동발전_시간대별_태양광_발전실적(6월24일~6월30일).csv'
    ]
    weather_files = ['c:/Users/ms/Desktop/학습자료/OBS_ASOS_TIM_20250725134932.csv',
    'c:/Users/ms/Desktop/학습자료/OBS_ASOS_TIM_20250728101227.csv',
    'c:/Users/ms/Desktop/학습자료/OBS_ASOS_TIM_20250728101447.csv'
    ]

    initial_data = load_and_process_data(power_files, weather_files)
    if not initial_data.empty:
        featured_data = feature_engineering(initial_data)
        
        timestamps_to_remove_str = [
            '2025-04-26 11:00:00', 
            '2025-05-04 12:00:00',
            '2025-05-29 16:00:00',
            '2025-05-29 17:00:00'
        ]
        timestamps_to_remove = pd.to_datetime(timestamps_to_remove_str)
        
        indices_to_drop_manual = featured_data[
            featured_data['일시'].isin(timestamps_to_remove)
        ].index
        
        if not indices_to_drop_manual.empty:
            print(f"\n--- 수동 이상치 제거 시작 ---")
            print(f">>> 수동 제거 대상 {len(indices_to_drop_manual)}개 발견")
            featured_data = featured_data.drop(index=indices_to_drop_manual)
            print(f">>> 제거 후 데이터 수: {len(featured_data)}개")
            print("--- 수동 이상치 제거 완료 ---\n")
        else:
            print("\n>>> 수동으로 지정한 날짜가 데이터에 존재하지 않습니다.\n")

        features = [
            '기온(°C)', '습도(%)', '운량', 'hour', 'dayofweek',
            'hour_sin', 'hour_cos',
            'power_lag_24h', 'power_lag_48h', 'power_lag_72h', 'power_lag_96h',
            'power_lag_120h', 'power_lag_144h', 'power_lag_168h',
            'seasonal_hourly_mean', 'seasonal_hourly_median',
            '일사(MJ/m2)', '일조(hr)','seasonal_hourly_q25', 'seasonal_hourly_q75',
            'solar_azimuth', 'solar_altitude', 'clearsky_ghi' , 'power_lag_1h','power_lag_2h','power_lag_3h',
            'power_lag_4h','power_lag_5h','power_lag_6h','temp_roll_mean_3h','humidity_roll_mean_3h','cloud_roll_mean_3h',
            'power_roll_mean_3h', 'irradiance_volatility_3h', 
        ]
        target = '발전량(MWh)'
        
        X = featured_data[features]
        y_original = featured_data[target]

        # 🔧 [수정 2] 타겟 변수 로그 변환 (0 값을 처리하기 위해 log1p 사용)
        y = np.log1p(y_original)

        split_point = int(len(featured_data) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        print(f"--- 3. 데이터 분리 완료 (시계열 방식). 훈련용: {len(X_train)}개, 테스트용: {len(X_test)}개 ---\n")

        import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# ...

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    model = xgb.XGBRegressor(**params)
    tscv = TimeSeriesSplit(n_splits=5)
    mape_scores = []
    
    y_train_original = np.expm1(y_train)
    threshold = y_train_original.max() * 0.05
    
    for train_index, val_index in tscv.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        model.fit(X_train_fold, y_train_fold)
        
        preds_log = model.predict(X_val_fold)
        preds = np.expm1(preds_log)
        y_val_original = np.expm1(y_val_fold)
        
        meaningful_mask = y_val_original > threshold

        if meaningful_mask.sum() == 0:
            mape_scores.append(1.0)
            continue

        mape = mean_absolute_percentage_error(
            y_val_original[meaningful_mask],
            preds[meaningful_mask]
        )
        mape_scores.append(mape)
    return np.mean(mape_scores)

print("--- 4. 하이퍼파라미터 튜닝 시작 (Optuna with XGBoost) ---")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("튜닝 완료! 최적 하이퍼파라미터:", study.best_params)
print(f"튜닝 후 최소 MAPE (수정된 기준): {study.best_value * 100:.2f}%\n")

print("--- 5. 최종 모델 학습 및 평가 ---")
final_model = xgb.XGBRegressor(**study.best_params, random_state=42)
final_model.fit(X_train, y_train)

final_predictions_log = final_model.predict(X_test)
final_predictions = np.expm1(final_predictions_log)

y_test_original = np.expm1(y_test)
y_train_original = np.expm1(y_train)

r2 = r2_score(y_test_original, final_predictions)

threshold = y_train_original.max() * 0.05
meaningful_mask = y_test_original > threshold

final_mape = 0.0
if meaningful_mask.sum() > 0:
    final_mape = mean_absolute_percentage_error(
        y_test_original[meaningful_mask],
        final_predictions[meaningful_mask]
    )

print("-> 최종 평가 결과:")
print(f"-> R-squared: {r2:.3f}")
print(f"-> 주간 데이터 MAPE (수정된 기준): {final_mape * 100:.2f}% (목표: 15% 이내)")

if final_mape > 0 and final_mape <= 0.15:
    with open('solar_final_model_real_data_xgb.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    print("\n--- 목표 성능 달성! 최종 모델을 'solar_final_model_real_data_xgb.pkl' 파일로 저장했습니다. ---")
    
    # --- 🔧 [핵심 추가] 실시간 예측에 사용할 통계 데이터 저장 ---
    print("\n--- 실시간 예측용 통계 데이터 생성 및 저장 시작 ---")
        
        # 훈련 데이터 전체(X_train, y_train을 합친 데이터)를 사용하여 통계치를 계산합니다.
    train_df = featured_data.iloc[:split_point]

        # feature_engineering 함수에서 사용했던 것과 동일한 로직으로 통계 특징을 계산합니다.
    def q25(x): return x.quantile(0.25)
    def q75(x): return x.quantile(0.75)
    seasonal_hourly_stats = train_df.groupby(['season', 'hour'])['발전량(MWh)'].agg(['mean', 'median', q25, q75]).reset_index()
    seasonal_hourly_stats.columns = ['season', 'hour', 'seasonal_hourly_mean', 'seasonal_hourly_median', 'seasonal_hourly_q25', 'seasonal_hourly_q75']
        
    # 계산된 통계 데이터를 별도의 파일로 저장합니다.
    with open('historical_stats.pkl', 'wb') as f:
        pickle.dump(seasonal_hourly_stats, f)
    print("--- 실시간 예측용 통계 데이터('historical_stats.pkl')를 저장했습니다. ---")
        # ----------------------------------------------------
    
else:
    print("\n--- 목표 성능에 아직 도달하지 못했습니다. ---")




# --- 🔧 [개선] 6. 최종 결과 시각화 ---
print("\n--- 6. 최종 결과 시각화 ---")

# 시각화를 위해 테스트셋 실제값과 예측값을 데이터프레임으로 만듭니다.

test_dates = featured_data.iloc[split_point:]['일시'].reset_index(drop=True)

results_df = pd.DataFrame({
'일시': test_dates,
'actual': np.expm1(y_test).reset_index(drop=True),
'predicted': final_predictions
})
results_df.set_index('일시', inplace=True)

# 1. 예측-실제 비교 산점도
plt.figure(figsize=(8, 8))
sns.scatterplot(x='actual', y='predicted', data=results_df, alpha=0.5, color='darkgreen', edgecolor=None)
plt.plot([0, results_df['actual'].max()], [0, results_df['actual'].max()], 'r--', label='정확 예측선')
plt.title('실제 발전량 vs 예측 발전량 (산점도)', fontsize=14)
plt.xlabel('실제 발전량 (MWh)', fontsize=12)
plt.ylabel('예측 발전량 (MWh)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. 시간에 따른 예측 결과 시계열 그래프 (샘플링하여 일부만 확인)
sample_interval = 1440  # 하루 단위: 1440분, 주 단위: 10080분 등 조정 가능
for i in range(0, len(results_df), sample_interval):
    sample = results_df.iloc[i:i + sample_interval]
    if len(sample) < 10:  # 너무 적은 데이터는 스킵
        continue

    plt.figure(figsize=(18, 5))
    plt.plot(sample.index, sample['actual'], label='실제 발전량', color='steelblue', linewidth=2)
    plt.plot(sample.index, sample['predicted'], label='예측 발전량', color='orangered', linestyle='--', linewidth=2)
    plt.title(f'주간 발전량 예측 결과 (샘플 {i} ~ {i + sample_interval})', fontsize=14)
    plt.xlabel('시간', fontsize=12)
    plt.ylabel('발전량 (MWh)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()



    # 시간 포맷을 자동 조정
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.show()
    
    
from sklearn.linear_model import LinearRegression
import numpy as np

plt.figure(figsize=(8, 8))
sns.scatterplot(x='actual', y='predicted', data=results_df, alpha=0.5, color='darkgreen', edgecolor=None)

# 정확 예측선
max_val = max(results_df['actual'].max(), results_df['predicted'].max())
plt.plot([0, max_val], [0, max_val], 'r--', label='정확 예측선')

# 추세선
lr = LinearRegression()
lr.fit(results_df[['actual']], results_df['predicted'])
x_vals = np.linspace(0, max_val, 100)
y_vals = lr.predict(x_vals.reshape(-1, 1))
plt.plot(x_vals, y_vals, 'b-', label='추세선')

plt.title('실제 발전량 vs 예측 발전량', fontsize=14)
plt.xlabel('실제 발전량 (MWh)')
plt.ylabel('예측 발전량 (MWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 6))
plt.plot(results_df.index, results_df['actual'], label='실제 발전량', color='steelblue', linewidth=1)
plt.plot(results_df.index, results_df['predicted'], label='예측 발전량', color='orangered', linestyle='--', linewidth=1)
plt.title('시간에 따른 전체 발전량 예측 결과', fontsize=14)
plt.xlabel('시간')
plt.ylabel('발전량 (MWh)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# 시간 포맷 자동 조정
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.tight_layout()
plt.show()


results_df['hour'] = results_df.index.hour
hourly_avg = results_df.groupby('hour')[['actual', 'predicted']].mean()

plt.figure(figsize=(10, 5))
hourly_avg.plot(kind='bar', ax=plt.gca(), color=['skyblue', 'salmon'])
plt.title('시간대별 평균 발전량 비교')
plt.xlabel('시간 (Hour of Day)')
plt.ylabel('평균 발전량 (MWh)')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.legend(['실제 발전량', '예측 발전량'])
plt.show()

results_df['residual'] = results_df['actual'] - results_df['predicted']

plt.figure(figsize=(18, 5))
plt.plot(results_df.index, results_df['residual'], color='purple', linewidth=0.8)
plt.title('시간에 따른 예측 오차(Residual)', fontsize=14)
plt.xlabel('시간')
plt.ylabel('오차 (MWh)')
plt.grid(True)
plt.tight_layout()

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.show()

results_df['weekday'] = results_df.index.dayofweek  # 0: 월, ..., 6: 일
results_df['hour'] = results_df.index.hour

pivot_table = results_df.pivot_table(index='hour', columns='weekday', values='actual', aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt=".1f")
plt.title('시간대별 요일별 평균 실제 발전량 (MWh)')
plt.xlabel('요일 (0=월 ~ 6=일)')
plt.ylabel('시간 (Hour)')
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(results_df['actual'], results_df['predicted'])
mape = mean_absolute_percentage_error(results_df['actual'], results_df['predicted']) * 100
rmse = np.sqrt(mean_squared_error(results_df['actual'], results_df['predicted']))

metrics = pd.DataFrame({
    '지표': ['MAE', 'MAPE', 'RMSE'],
    '값': [mae, mape, rmse]
})

plt.figure(figsize=(6, 4))
sns.barplot(x='지표', y='값', data=metrics, palette='Set2')
plt.title('예측 성능 지표')
plt.ylabel('값')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

import matplotlib.dates as mdates

# 결과 DataFrame의 날짜 정보 추출
results_df['month'] = results_df.index.to_period('M')

# 월별로 루프를 돌며 시계열 그래프 출력
for month, group in results_df.groupby('month'):
    if len(group) < 10:  # 너무 적은 데이터는 건너뜀
        continue

    plt.figure(figsize=(18, 5))
    plt.plot(group.index, group['actual'], label='실제 발전량', color='steelblue', linewidth=1.5)
    plt.plot(group.index, group['predicted'], label='예측 발전량', color='orangered', linestyle='--', linewidth=1.5)

    plt.title(f'{month} 발전량 예측 결과', fontsize=14)
    plt.xlabel('시간')
    plt.ylabel('발전량 (MWh)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.tight_layout()
    plt.show()


# 1. 최종 모델을 전체 훈련 데이터로 학습
final_model = xgb.XGBRegressor(**study.best_params, random_state=42)
final_model.fit(X_train, y_train)

# 2. 테스트셋에 대한 예측 수행 (로그 스케일)
predictions_log = final_model.predict(X_test)

# 3. 원래 스케일로 변환
predictions = np.expm1(predictions_log)
y_test_original = np.expm1(y_test)

# 4. 분석을 위한 결과 데이터프레임 생성
# X_test에 있는 모든 특징과 실제값, 예측값을 하나로 합칩니다.
error_analysis_df = X_test.copy()
error_analysis_df['Timestamp'] = featured_data.loc[X_test.index, '일시'] # 시간 정보 추가
error_analysis_df['Actual_Power'] = y_test_original
error_analysis_df['Predicted_Power'] = predictions
error_analysis_df['Error'] = np.abs(error_analysis_df['Actual_Power'] - error_analysis_df['Predicted_Power'])

print("에러 분석용 데이터프레임 생성 완료!")
error_analysis_df.head()

# 에러가 큰 순서대로 정렬하여 상위 50개 확인
worst_predictions_df = error_analysis_df.sort_values(by='Error', ascending=False).head(50)

print("오차가 가장 큰 상위 50개 데이터:")
print(worst_predictions_df)