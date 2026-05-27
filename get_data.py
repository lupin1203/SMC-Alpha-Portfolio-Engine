import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 🌟 新增：證交所產業代碼翻譯字典
TWSE_INDUSTRY_MAP = {
    "01": "水泥工業", "02": "食品工業", "03": "塑膠工業", "04": "紡織纖維", "05": "電機機械",
    "06": "電器電纜", "07": "化學工業", "08": "玻璃陶瓷", "09": "造紙工業", "10": "鋼鐵工業",
    "11": "橡膠工業", "12": "汽車工業", "14": "建材營造", "15": "航運業", "16": "觀光餐旅",
    "17": "金融保險", "18": "貿易百貨", "20": "其他類", "21": "化學工業", "22": "生技醫療",
    "23": "油電燃氣", "24": "半導體業", "25": "電腦及週邊", "26": "光電業", "27": "通信網路",
    "28": "電子零組件", "29": "電子通路", "30": "資訊服務", "31": "其他電子"
}

def generate_stock_list():
    print("📡 正在抓取全台股清單並進行中文翻譯...")
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    
    try:
        response = requests.get(url, verify=False)
        data = response.json()
        
        stock_list = []
        for item in data:
            ticker = item.get('公司代號', '')
            name = item.get('公司簡稱', item.get('公司名稱', ''))
            raw_category = item.get('產業別', '')
            
            # 🌟 透過字典翻譯，如果找不到就顯示原本的代碼
            category_name = TWSE_INDUSTRY_MAP.get(raw_category, f"未分類 ({raw_category})")
            
            if len(ticker) == 4 and ticker.isdigit():
                stock_list.append({
                    "category": category_name,
                    "ticker": f"{ticker}.TW",
                    "name": name
                })
                
        output_df = pd.DataFrame(stock_list)
        output_df.to_csv("stock_list.csv", index=False, encoding="utf-8-sig")
        print(f"✅ 翻譯成功！已產出 {len(output_df)} 檔全中文股票清單！")
        
    except Exception as e:
        print(f"❌ 抓取失敗: {e}")

generate_stock_list()