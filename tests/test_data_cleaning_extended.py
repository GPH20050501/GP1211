import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from data.market_data import MarketDataLoader
import pytz

class TestDataCleaningExtended:
    @pytest.fixture
    def market_loader(self):
        return MarketDataLoader()
    
    @pytest.fixture
    def sample_data_with_problems(self):
        """生成包含各種問題的測試數據"""
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            # 1. 缺失值
            'Open': [100] * 5 + [np.nan] * 3 + [105] * 12,
            # 2. 異常值
            'High': [110] * 8 + [1000] + [115] * 11,  # 1000是異常值
            # 3. 負值
            'Low': [90] * 10 + [-10] + [95] * 9,
            # 4. 價格邏輯錯誤
            'Close': [105] * 15 + [200, 50, 106, 107, 108],  # 價格跳動過大
            # 5. 成交量異常
            'Volume': [1000] * 12 + [1000000] + [1200] * 7  # 1000000是異常值
        }, index=dates)
        return data

    def test_missing_value_patterns(self, market_loader, sample_data_with_problems):
        """測試不同缺失值模式的處理"""
        # 1. 隨機缺失
        random_missing = sample_data_with_problems.copy()
        random_missing.iloc[np.random.choice(len(random_missing), 5), :] = np.nan
        
        # 2. 連續缺失
        consecutive_missing = sample_data_with_problems.copy()
        consecutive_missing.iloc[5:8, :] = np.nan
        
        # 3. 特定列缺失
        column_missing = sample_data_with_problems.copy()
        column_missing['Close'] = np.nan
        
        # 驗證處理結果
        for test_data in [random_missing, consecutive_missing, column_missing]:
            cleaned = market_loader._clean_data(test_data)
            assert not cleaned.isnull().any().any(), "清洗後仍存在缺失值"
            assert len(cleaned) > 0, "清洗後數據為空"

    def test_outlier_detection_methods(self, market_loader, sample_data_with_problems):
        """測試多種異常值檢測方法"""
        # 1. 3σ法則
        def detect_outliers_3sigma(series):
            mean = series.mean()
            std = series.std()
            return series[abs(series - mean) > 3 * std]
        
        # 2. IQR法則
        def detect_outliers_iqr(series):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            return series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]
        
        # 3. Z-score法則
        def detect_outliers_zscore(series):
            z_scores = abs((series - series.mean()) / series.std())
            return series[z_scores > 3]
        
        # 對每個價格列應用三種方法
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            series = sample_data_with_problems[col]
            
            # 收集各方法檢測結果
            outliers_3sigma = detect_outliers_3sigma(series)
            outliers_iqr = detect_outliers_iqr(series)
            outliers_zscore = detect_outliers_zscore(series)
            
            # 驗證檢測結果的一致性
            assert len(outliers_3sigma) > 0, f"{col}列未檢測到3σ異常值"
            assert len(outliers_iqr) > 0, f"{col}列未檢測到IQR異常值"
            assert len(outliers_zscore) > 0, f"{col}列未檢測到Z-score異常值"

    def test_time_related_issues(self, market_loader):
        """測試時間相關的問題"""
        # 使用UTC時區的數據
        dates = pd.date_range('2024-01-01', periods=5, tz='UTC')
        data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000] * 5
        }, index=dates)
        
        # 驗證時區處理
        cleaned_data = market_loader._clean_data(data)
        assert all(cleaned_data.index.tz == pytz.UTC), "時區未統一為UTC"
        
        # 2. 測試交易時間驗證
        market_hours = {
            'start': '09:30',
            'end': '16:00'
        }
        
        def is_market_hours(timestamp):
            time_str = timestamp.strftime('%H:%M')
            return market_hours['start'] <= time_str <= market_hours['end']
        
        # 生成包含非交易時間的數據
        dates_with_time = pd.date_range(
            '2024-01-01 00:00', 
            '2024-01-01 23:59', 
            freq='H',
            tz='America/New_York'
        )
        
        data_with_time = pd.DataFrame({
            'Close': range(len(dates_with_time))
        }, index=dates_with_time)
        
        cleaned_time_data = market_loader._clean_data(data_with_time)
        assert all(is_market_hours(idx) for idx in cleaned_time_data.index), \
            "清洗後數據包含非交易時間"

    @pytest.mark.parametrize("test_case", [
        # 1. 價格跳動過大
        {
            'data': pd.DataFrame({
                'Close': [100, 101, 200, 102, 103]
            }, index=pd.date_range('2024-01-01', periods=5)),
            'expected_length': 4
        },
        # 2. 成交量為零
        {
            'data': pd.DataFrame({
                'Close': [100] * 5,
                'Volume': [1000, 0, 1200, 0, 1400]
            }, index=pd.date_range('2024-01-01', periods=5)),
            'expected_length': 3
        },
        # 3. 價格為零或負數
        {
            'data': pd.DataFrame({
                'Close': [100, -100, 0, 102, 103]
            }, index=pd.date_range('2024-01-01', periods=5)),
            'expected_length': 3
        }
    ])
    def test_specific_edge_cases(self, market_loader, test_case):
        """測試特定的邊界情況"""
        data = test_case['data'].copy()
        
        # 確保數據類型正確
        if 'Close' in data.columns:
            data['Close'] = data['Close'].astype(float)
        if 'Volume' in data.columns:
            data['Volume'] = data['Volume'].astype(float)
        
        cleaned_data = market_loader._clean_data(data)
        
        # 驗證結果
        assert len(cleaned_data) == test_case['expected_length'], \
            f"邊界情況處理錯誤，期望{test_case['expected_length']}行數據，實際得到{len(cleaned_data)}行"

    def test_data_consistency(self, market_loader, sample_data_with_problems):
        """測試數據一致性"""
        cleaned_data = market_loader._clean_data(sample_data_with_problems)
        
        # 1. 檢查OHLC邏輯關係
        assert all(cleaned_data['High'] >= cleaned_data['Open']), "High應大於等於Open"
        assert all(cleaned_data['High'] >= cleaned_data['Close']), "High應大於等於Close"
        assert all(cleaned_data['Low'] <= cleaned_data['Open']), "Low應小於等於Open"
        assert all(cleaned_data['Low'] <= cleaned_data['Close']), "Low應小於等於Close"
        
        # 2. 檢查數據連續性
        price_changes = cleaned_data['Close'].pct_change().abs()
        assert all(price_changes < 0.2), "存在異常的價格跳動"
        
        # 3. 檢查成交量合理性
        assert all(cleaned_data['Volume'] > 0), "成交量應該為正數"
        volume_changes = cleaned_data['Volume'].pct_change().abs()
        assert all(volume_changes < 5), "存在異常的成交量變化"

    def test_extreme_cases(self, market_loader):
        """測試極端情況"""
        # 1. 生成極端測試數據
        extreme_data = pd.DataFrame({
            'Open': [100, 1000000, -1000, np.inf, np.nan],
            'High': [101, 1000001, -999, np.inf, np.nan],
            'Low': [99, 999999, -1001, -np.inf, np.nan],
            'Close': [100, 1000000, -1000, np.nan, np.inf],
            'Volume': [1000, 0, -1, np.inf, np.nan]
        }, index=pd.date_range('2024-01-01', periods=5))
        
        # 2. 驗證處理結果
        cleaned = market_loader._clean_data(extreme_data)
        
        assert len(cleaned) > 0, "清洗後數據不應為空"
        assert not cleaned.isin([np.inf, -np.inf]).any().any(), "數據中不應包含無窮值"
        assert not cleaned.isnull().any().any(), "數據中不應包含空值"
        assert all(cleaned['Volume'] > 0), "成交量應為正值"

    def _handle_timezone(self, data: pd.DataFrame) -> pd.DataFrame:
        """處理時區相關的問題"""
        try:
            if isinstance(data.index, pd.DatetimeIndex):
                if data.index.tz is not None:
                    # 直接轉換到 UTC
                    data.index = data.index.tz_convert('UTC')
                else:
                    # 如果沒有時區信息，假設是 UTC
                    data.index = data.index.tz_localize('UTC')
                
                # 驗證轉換結果
                if not all(idx.tzinfo is not None for idx in data.index):
                    raise ValueError("時區轉換後存在無時區時間戳")
                
                return data
                
        except Exception as e:
            logger.error(f"時區處理失敗: {str(e)}")
            raise

    def _detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """檢測和處理異常值"""
        try:
            # 只處理數值型列
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # 1. 計算統計量
                mean = data[col].mean()
                std = data[col].std()
                
                # 2. 使用 3σ 法則標記異常值
                z_scores = abs((data[col] - mean) / std)
                outliers_3sigma = z_scores > 3
                
                # 3. 使用 IQR 法則標記異常值
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers_iqr = (data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)
                
                # 4. 綜合兩種方法
                outliers = outliers_3sigma | outliers_iqr
                
                if outliers.any():
                    # 使用移動中位數替換異常值
                    window = 5
                    rolling_median = data[col].rolling(
                        window=window, 
                        center=True, 
                        min_periods=1
                    ).median()
                    
                    # 確保替換值在合理範圍內
                    if col in ['High', 'Low', 'Open', 'Close']:
                        rolling_median = rolling_median.clip(lower=0)  # 價格不能為負
                    elif col == 'Volume':
                        rolling_median = rolling_median.clip(lower=0).round()  # 成交量為非負整數
                        
                    data.loc[outliers, col] = rolling_median[outliers]
                    
            return data
            
        except Exception as e:
            logger.error(f"異常值檢測失敗: {str(e)}")
            return data

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清理和標準化數據"""
        try:
            if len(data) == 0:
                return data.copy()
            
            data = data.copy(deep=True)
            original_length = len(data)
            
            # 1. 數據類型轉換
            data = self._convert_datatypes(data)
            
            # 2. 處理異常值（在時區轉換前）
            data = self._detect_outliers(data)
            
            # 3. 處理時區
            data = self._handle_timezone(data)
            
            # 4. 處理缺失值
            data = self._handle_missing_values(data)
            
            # 5. 處理價格邏輯和跳動
            data = self._handle_price_logic(data)
            data = self._handle_price_jumps(data)
            
            # 6. 處理數據長度
            data = self._handle_data_length(data, original_length)
            
            # 7. 最後的清理
            data = data.interpolate(method='linear', limit_direction='both')
            data = data.ffill().bfill()
            
            # 8. 移除無效行
            data = data.dropna(how='all')
            
            return data
            
        except Exception as e:
            logger.error(f"數據清洗失敗: {str(e)}")
            raise 

    def test_data_quality_metrics_extended(self, market_loader, multi_market_data):
        """擴展的數據質量測試"""
        try:
            for market, data in multi_market_data.items():
                # 1. 基本數據檢查
                assert not data.empty, f"市場 {market} 數據為空"
                assert len(data) >= 100, f"市場 {market} 數據量不足"
                
                # 2. 價格邏輯檢查
                price_logic = {
                    'high_vs_open': (data['High'] >= data['Open']).all(),
                    'high_vs_close': (data['High'] >= data['Close']).all(),
                    'low_vs_open': (data['Low'] <= data['Open']).all(),
                    'low_vs_close': (data['Low'] <= data['Close']).all()
                }
                
                # 3. 成交量檢查
                volume_checks = {
                    'positive': (data['Volume'] > 0).all(),
                    'reasonable': (data['Volume'] < data['Volume'].mean() * 10).all()
                }
                
                # 4. 時間序列檢查
                time_checks = {
                    'monotonic': data.index.is_monotonic_increasing,
                    'unique': data.index.is_unique
                }
                
                # 詳細的錯誤信息
                for check_name, result in {**price_logic, **volume_checks, **time_checks}.items():
                    assert result, f"市場 {market} 失敗於 {check_name} 檢查"
                    
        except Exception as e:
            logger.error(f"擴展數據質量測試失敗: {str(e)}")
            raise