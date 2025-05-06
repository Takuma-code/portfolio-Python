# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from datetime import datetime, timedelta

# タイトルとイントロダクション
st.title('株価分析と予測アプリ')
st.write('このアプリは株価データを分析し、将来の価格を予測します。')

# サイドバーの設定
st.sidebar.header('パラメータ設定')
company_name = st.sidebar.text_input('会社名', '株式会社サンプル')
days = st.sidebar.slider('データ期間（日）', 100, 1000, 365)
forecast_days = st.sidebar.slider('予測期間（日）', 7, 90, 30)

# StockAnalyzerクラスの定義（元のコードから必要な部分を抽出）
class StockAnalyzer:
    def __init__(self, company_name="サンプル企業"):
        self.company_name = company_name
        self.model = None
        self.scaler = StandardScaler()
        self.data = None
        
    def generate_sample_data(self, days=365):
        # 元のコードと同じデータ生成ロジック
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        date_list = [start_date + timedelta(days=i) for i in range(days)]
        
        base_price = 10000
        prices = [base_price]
        
        for i in range(1, days):
            seasonal = 500 * np.sin(i / 30 * np.pi)
            trend = i * 5
            random = np.random.normal(0, 200)
            
            new_price = prices[-1] + random + (trend / 100) + (seasonal / 10)
            new_price = max(new_price, 100)
            prices.append(new_price)
        
        self.data = pd.DataFrame({
            '日付': date_list,
            '始値': prices,
            '高値': [p * (1 + np.random.uniform(0, 0.03)) for p in prices],
            '安値': [p * (1 - np.random.uniform(0, 0.03)) for p in prices],
            '終値': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
            '出来高': [int(np.random.uniform(50000, 500000)) for _ in range(days)]
        })
        
        self.data['MA5'] = self.data['終値'].rolling(window=5).mean()
        self.data['MA20'] = self.data['終値'].rolling(window=20).mean()
        self.data['MA50'] = self.data['終値'].rolling(window=50).mean()
        
        self.data['Volatility'] = self.data['終値'].rolling(window=20).std()
        
        return self.data
    
    # 他のメソッド（prepare_features, train_model, forecast_future）も同様に実装
    # ここでは省略しますが、実際のアプリでは必要です

# メイン処理
analyzer = StockAnalyzer(company_name)

# データ生成
with st.spinner('データを生成中...'):
    data = analyzer.generate_sample_data(days=days)
    st.success('データ生成完了！')

# データ表示タブ
tab1, tab2, tab3 = st.tabs(["データ", "分析", "予測"])

with tab1:
    st.subheader('生成された株価データ')
    st.dataframe(data.head())
    
    st.subheader('株価チャート')
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # 株価と移動平均線
    axes[0].plot(data['日付'], data['終値'], label='終値', linewidth=2)
    axes[0].plot(data['日付'], data['MA5'], label='5日移動平均', linewidth=1.5)
    axes[0].plot(data['日付'], data['MA20'], label='20日移動平均', linewidth=1.5)
    axes[0].plot(data['日付'], data['MA50'], label='50日移動平均', linewidth=1.5)
    
    axes[0].fill_between(data['日付'], data['安値'], data['高値'], 
                       alpha=0.2, color='gray', label='価格範囲')
    
    axes[0].set_title(f'{company_name}の株価チャート', fontsize=16)
    axes[0].set_ylabel('価格 (円)', fontsize=12)
    axes[0].legend(loc='upper left')
    axes[0].grid(True)
    
    # 出来高チャート
    axes[1].bar(data['日付'], data['出来高'], color='#1f77b4', alpha=0.7)
    axes[1].set_ylabel('出来高', fontsize=12)
    axes[1].set_xlabel('日付', fontsize=12)
    axes[1].grid(True)
    
    fig.autofmt_xdate()
    st.pyplot(fig)

with tab2:
    if st.button('モデルを訓練'):
        with st.spinner('モデルを訓練中...'):
            # モデル訓練のコードを実装
            # analyzer.prepare_features()
            # analyzer.train_model()
            st.success('モデル訓練完了！')
            
            # 結果の可視化
            st.subheader('特徴量の重要度')
            # 特徴量の重要度を表示するコード
            
            st.subheader('モデル評価')
            # モデル評価指標を表示するコード

with tab3:
    if st.button('将来の株価を予測'):
        with st.spinner(f'将来{forecast_days}日間の株価を予測中...'):
            # 予測のコードを実装
            # forecast = analyzer.forecast_future(days=forecast_days)
            
            # 予測結果の可視化
            st.subheader('株価予測')
            # 予測結果を表示するコード
