import tushare as ts
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from tqdm import tqdm

def fetch_stock_data(start_date, end_date, batch_days=100):
    """
    获取股票数据并一次性保存到SQLite
    
    参数:
    start_date: str, 开始日期 (YYYYMMDD)
    end_date: str, 结束日期 (YYYYMMDD)
    batch_days: int, 每批处理的天数
    """
    try:
        # 初始化tushare
        ts.set_token('f55086b7b9a5de7a4d04405ab77085004596d1484d6fb7e437334d0d')
        pro = ts.pro_api()
        
        # 创建日期范围
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        
        # 存储所有数据
        all_data = []
        
        # 分批处理日期
        current_date = start
        
        with tqdm(total=(end - start).days) as pbar:
            while current_date <= end:
                try:
                    # 获取数据
                    df = pro.daily(
                        trade_date=current_date.strftime('%Y%m%d')
                    )
                    
                    if df is not None and not df.empty:
                        print(f"日期 {current_date.strftime('%Y%m%d')}: 已获取 {len(df)} 条数据")
                    else:
                        print(f"日期 {current_date.strftime('%Y%m%d')}: 未获取数据")
                    if df is not None and not df.empty:
                        all_data.append(df)
                    
                    # 更新进度条
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Error fetching data for {current_date.strftime('%Y%m%d')}: {str(e)}")
                
                # 移动到下一天
                current_date += timedelta(days=1)
        
        print("数据获取完成，开始合并...")
        
        # 合并所有数据
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"合并完成，共 {len(combined_df)} 条数据")
            
            # 连接数据库并保存数据
            conn = sqlite3.connect('./data/train.nfa')
            
            # 一次性写入数据
            combined_df.to_sql(
                'daily_info', 
                conn, 
                if_exists='append', 
                index=False,
                chunksize=10000,
                method='multi'
            )
            
            conn.close()
            print("数据保存完成!")
            
            return combined_df
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return None

if __name__ == "__main__":
    # 使用示例
    df = fetch_stock_data('20000101', '20250207')

