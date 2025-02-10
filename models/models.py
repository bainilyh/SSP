import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_curve, average_precision_score
import joblib
from datetime import datetime
from preprocess import get_latest_records, analyze_label_distribution, load_stock_data
import matplotlib.pyplot as plt
from analysis4 import plot_candlestick

class StockPredictor:
    def __init__(self, model_path='stock_model.xgb'):
        self.model = None
        self.model_path = model_path
        self.feature_columns = None
        self.label_col = 'label'
        
    def preprocess_data(self, stock_info):
        """
        数据预处理:
        1. 处理缺失值
        2. 删除每个股票最后n个交易日的数据
        3. 划分训练测试集
        4. 转换数据格式
        """
        # 删除包含无效标签的行
        df = stock_info[stock_info['label'].isin([-1, 0, 1])].copy()
        
        # 将-1和0的标签合并为0
        df['label'] = df['label'].map({-1: 0, 0: 0, 1: 1})
        
        # 删除每个股票最后n个交易日的数据
        n = 5  # 最后5个交易日
        df = df.sort_values(['ts_code', 'trade_date'])
        df = df.groupby('ts_code').apply(
            lambda x: x.iloc[:-n] if len(x) > n else x
        ).reset_index(drop=True)
        # 丢弃缺失值
        df = df.fillna(0)
        
        # 定义特征列（排除非特征列）
        exclude_cols = ['ts_code', 'trade_date', self.label_col]
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # 划分数据集（按时间顺序）
        df.sort_values('trade_date', inplace=True)
        split_idx = int(len(df) * 0.9)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # 准备数据
        X_train = train_df[self.feature_columns]
        y_train = train_df[self.label_col]
        X_test = test_df[self.feature_columns]
        y_test = test_df[self.label_col]
        
        # 打印数据集信息
        print("\nDataset Info:")
        print(f"Total samples: {len(df):,}")
        print(f"Training samples: {len(train_df):,}")
        print(f"Testing samples: {len(test_df):,}")
        print(f"Number of features: {len(self.feature_columns)}")
        print(f"Label distribution:")
        print(df[self.label_col].value_counts(normalize=True).round(4) * 100)
        
        return X_train, y_train, X_test, y_test
    
    def train(self, stock_info, params=None):
        """
        训练模型 - 二分类问题，处理1:5的类别不平衡
        """
        # # 增强正则化
        # default_params = {
        #     'objective': 'binary:logistic',
        #     'eval_metric': ['aucpr', 'auc'],
        #     'eta': 0.05,  # 更小的学习率
        #     'max_depth': 4,  # 更浅的树
        #     'gamma': 0.2,  # 更强的节点分裂阈值
        #     'subsample': 0.6,  # 更低的样本采样率
        #     'colsample_bytree': 0.6,
        #     'scale_pos_weight': 5,  # 增大正样本权重
        #     'min_child_weight': 15,
        #     'lambda': 2.0,  # 增强L2正则
        #     'alpha': 0.8,
        #     'seed': 42
        # }

        # 设置默认参数
        default_params = {
            'objective': 'binary:logistic',  # 二分类
            'eval_metric': ['aucpr', 'auc'],  # 不平衡数据集使用AUC评估
            'eta': 0.05,  # 降低学习率防止过拟合
            'max_depth': 4,  # 减小树的深度防止过拟合
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 5,  # 正负样本比1:5,设置正样本权重为5
            'min_child_weight': 5,  # 增加以减少过拟合
            'gamma': 0.1,  # 增加以减少过拟合
            'seed': 42
        }
        self.params = params or default_params
        
        # 从SQLite加载预处理数据
        train_df = load_stock_data(table_name='preprocessed_train_data')
        test_df = load_stock_data(table_name='preprocessed_test_data')
        feature_df = load_stock_data(table_name='feature_columns')
        
        # 获取特征列名
        self.feature_columns = feature_df['feature_name'].tolist()
        
        # 准备训练和测试数据
        X_train = train_df[self.feature_columns]
        y_train = train_df[self.label_col]
        X_test = test_df[self.feature_columns]
        y_test = test_df[self.label_col]
        
        # 打印数据集信息
        print("\nDataset Info:")
        print(f"Total samples: {len(train_df) + len(test_df):,}")
        print(f"Training samples: {len(train_df):,}")
        print(f"Testing samples: {len(test_df):,}")
        print(f"Number of features: {len(self.feature_columns)}")
        print(f"Label distribution:")
        print(pd.concat([train_df[self.label_col], test_df[self.label_col]]).value_counts(normalize=True).round(4) * 100)
        
        # 转换数据格式
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # 定义学习率衰减函数
        def learning_rate_decay(epoch):
            return 0.02 * pow(0.99, epoch)
        
        # 训练模型
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=3000,
            evals=[(dtrain, 'train'), (dtest, 'eval')],
            early_stopping_rounds=150,  # 增大早停轮数
            verbose_eval=100,
            # callbacks=[xgb.callback.LearningRateScheduler(learning_rate_decay)]  # 动态学习率衰减
        )
        
        # 保存模型
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
        
        # 评估模型
        self.evaluate(X_test, y_test)
        
        # 检查训练集和测试集的AUC差异
        train_pred = self.model.predict(xgb.DMatrix(X_train))
        train_auc = roc_auc_score(y_train, train_pred)
        test_auc = roc_auc_score(y_test, self.model.predict(xgb.DMatrix(X_test)))
        print(f"Train AUC: {train_auc:.4f}  Test AUC: {test_auc:.4f}  Diff: {train_auc-test_auc:.4f}")

        # 如果差异>0.15则可能存在过拟合
        # 如果测试AUC也>0.8且差异小，可能需要检查业务逻辑合理性
    
    def evaluate(self, X, y, threshold=0.5):
        """使用PR曲线评估"""
        if self.model is None:
            self.load_model()
            
        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)
        # 二分类问题，使用0.5作为阈值
        y_pred = (preds >= threshold).astype(int)
        
        print("\nEvaluation Results:")
        print("Accuracy:", accuracy_score(y, y_pred))
        print("预测为买入的个数:", np.sum(y_pred))
        print("真实为买入的个数:", np.sum(y))
        print(classification_report(y, y_pred))
        
        # 计算PR曲线
        precision, recall, thresholds = precision_recall_curve(y, preds)
        avg_precision = average_precision_score(y, preds)
        
        # 找到最佳阈值
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        best_idx = np.argmax(f1_scores)
        print(f"Best threshold: {thresholds[best_idx]:.2f}")
        print(f"Corresponding precision: {precision[best_idx]:.2f}")
    
    def predict(self, new_data, threshold=0.65):  # 提高分类阈值
        """预测时使用动态阈值"""
        if self.model is None:
            self.load_model()
            
        if isinstance(new_data, pd.DataFrame):
            # 确保特征列一致
            new_data = new_data[self.feature_columns]
            dnew = xgb.DMatrix(new_data)
        else:
            dnew = new_data
            
        preds = self.model.predict(dnew)
        # 返回预测类别和概率
        return (preds >= threshold).astype(int), preds
    
    def load_model(self):
        """
        加载已保存的模型
        """
        try:
            self.model = joblib.load(self.model_path)
            print("Model loaded successfully")
        except FileNotFoundError:
            raise Exception("Model file not found. Please train the model first.")
    
    def feature_importance(self):
        """
        显示特征重要性
        """
        if self.model is None:
            self.load_model()
            
        importance = self.model.get_score(importance_type='weight')
        importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\nFeature Importance:")
        for feat, score in importance[:10]:  # 显示前10个重要特征
            print(f"{feat}: {score:.4f}")

    def business_rules_filter(predictions, probabilities, df):
        """业务规则过滤"""
        # 示例规则：同时满足高概率和成交量条件
        mask = (
            (probabilities > 0.7) &
            (df['vol'] > df['vol_ma'] * 1.5) &
            (df['close'] > df['ma20'])
        )
        return predictions & mask

    def save_and_plot_predictions(self, stock_info, X_test, threshold=0.65, n_before=60, n_after=30):
        """
        保存并可视化预测为买点的股票
        
        参数:
        stock_info: DataFrame, 包含完整股票数据
        X_test: DataFrame, 已准备好的预测数据
        threshold: float, 预测阈值
        n_before: int, 向前查看的交易日数量
        n_after: int, 向后查看的交易日数量
        """
        if self.model is None:
            self.load_model()
        
        # 使用已准备好的预测数据
        dnew = xgb.DMatrix(X_test)
        probabilities = self.model.predict(dnew)
        predictions = (probabilities >= threshold).astype(int)
        
        # 获取X_test的索引
        test_indices = X_test.index
        
        # 获取预测为买点的股票信息
        buy_signals = stock_info.loc[test_indices[predictions == 1]].copy()
        buy_signals['probability'] = probabilities[predictions == 1]
        
        # 按预测概率排序
        buy_signals = buy_signals.sort_values('probability', ascending=False)
        
        # 保存预测结果
        result_file = f'buy_signals_{datetime.now().strftime("%Y%m%d")}.csv'
        buy_signals[['ts_code', 'trade_date', 'probability']].to_csv(result_file, index=False)
        print(f"\n预测结果已保存到 {result_file}")
        print(f"共发现 {len(buy_signals)} 个买点")
        
        # 显示前10个最高概率的买点
        print("\n概率最高的10个买点:")
        print(buy_signals[['ts_code', 'trade_date', 'probability']].head(10))
        
        # 绘制K线图
        print("\n开始绘制K线图...")
        for idx, row in buy_signals.head(10).iterrows():
            print(f"\n绘制 {row.ts_code} 在 {row.trade_date} 的K线图 (预测概率: {row.probability:.4f})")
            try:
                plot_candlestick(stock_info, row.ts_code, row.trade_date, 
                               n_before=n_before, n_after=n_after)
            except Exception as e:
                print(f"Error plotting {row.ts_code} - {row.trade_date}: {str(e)}")
                continue
        plt.close()  # 关闭图形，释放内存
        
        return buy_signals

# 使用示例:
if __name__ == '__main__':
    
    # 加载数据
    stock_info = load_stock_data(table_name='stock_info_with_technical_indicators3')
    
    # # 获取最新记录
    # latest_records = get_latest_records(stock_info)
    # print("\n最新记录示例:")
    # print(latest_records[['ts_code', 'trade_date', 'close', 'label']].head())
    
    

    # 初始化模型
    predictor = StockPredictor('./models/stock_model.xgb')
    
    # # 训练模型
    # predictor.train(stock_info)

    # 加载模型
    predictor.load_model()
    
    # 删除包含无效标签的行
    stock_info = stock_info[stock_info['label'].isin([-1, 0, 1])].copy()
        
    # 将-1和0的标签合并为0
    stock_info['label'] = stock_info['label'].map({-1: 0, 0: 0, 1: 1})

    # 设置 feature_columns
    exclude_cols = ['ts_code', 'trade_date', predictor.label_col]
    predictor.feature_columns = [col for col in stock_info.columns if col not in exclude_cols]
    
    
    # # 预测
    # # condition = (stock_info['ts_code']=='300521.SZ') & (stock_info['trade_date']>='20220425') & (stock_info['trade_date']<='20220511')
    # # sample_data = stock_info[condition][predictor.feature_columns]
    # # predictions, probabilities = predictor.predict(sample_data)
    # # print("\nSample Predictions:", predictions)
    # # print("Probabilities:", probabilities)

    # 验证
    latest_records = get_latest_records(stock_info, n=6)
    X_test = latest_records[predictor.feature_columns]
    y_test = latest_records[predictor.label_col]
    predictor.evaluate(X_test, y_test, threshold=0.99)
    

    # 分析标签分布
    label_stats = analyze_label_distribution(latest_records)
    print("\n标签分布统计:")
    print(label_stats)
    
    # # 显示特征重要性
    # predictor.feature_importance()

    # 预测并可视化买点
    buy_signals = predictor.save_and_plot_predictions(stock_info, X_test, threshold=1)

    print('')
