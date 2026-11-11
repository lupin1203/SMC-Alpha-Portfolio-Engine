import pandas as pd
import requests
import json
import os

def fetch_dynamic_sector_map(save_cache=True, cache_file='sector_map_cache.json'):
    """
    透過 FinMind API 動態抓取台股產業分類，並生成 SECTOR_MAP
    """
    print("🔄 正在從 FinMind 同步最新台股產業分類地圖...")
    
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {
        "dataset": "TaiwanStockInfo"
    }
    # 從環境變數讀 token，提高速率上限（300/hr → 600/hr）
    _token = os.environ.get("FINMIND_TOKEN", "")
    if _token:
        parameter["token"] = _token
    
    try:
        res = requests.get(url, params=parameter)
        data = res.json()
        if 'data' not in data:
            print("❌ FinMind API 回傳格式異常或無資料。")
            return {}
        df = pd.DataFrame(data['data'])
    except Exception as e:
        print(f"❌ FinMind API 讀取失敗: {e}")
        return {}

    sector_map = {}
    
    # 進行資料分組 (以 industry_category 產業類別來 Group)
    for sector, group in df.groupby('industry_category'):
        # 過濾掉無效分類或 ETF
        if pd.isna(sector) or sector in ["", "ETF", "None", "不適用"]:
            continue
            
        tickers = []
        for _, row in group.iterrows():
            stock_id = str(row['stock_id'])
            
            # 過濾雜訊：確保是一般股票 (台股一般股票代號通常是 4 碼數字)
            if len(stock_id) == 4 and stock_id.isdigit():
                # 配合 yfinance 格式，加上 .TW (若你全用 FinMind 抓 K 線，可依需求拿掉 .TW)
                tickers.append(f"{stock_id}.TW") 
                
        # 只要這個產業有股票，就加入地圖中
        if len(tickers) > 0:
            sector_map[sector] = tickers

    # ✅ 合併上市/上櫃造成的重複板塊（FinMind 原始分類會把同類股拆成兩個名稱）
    sector_map = _normalize_sector_map(sector_map)

    print(f"成功生成產業地圖！共掃描到 {len(sector_map)} 個有效產業板塊。")
    
    # 將地圖快取成 JSON 檔案
    if save_cache:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(sector_map, f, ensure_ascii=False, indent=4)
            
    return sector_map


# 重複板塊對應表：把右邊的名稱合併到左邊的標準名稱
_SECTOR_ALIASES = {
    "其他電子業":   ["其他電子類", "其他電子"],
    "化學工業":     ["化學生技醫療"],
    "居家生活":     ["居家生活類"],
    "數位雲端":     ["數位雲端類"],
    "綠能環保":     ["綠能環保類"],
    "創新板股票":   ["創新版股票"],
    "觀光餐旅":     ["觀光事業"],
    "農業科技":     ["農業科技業"],
    "運動休閒":     ["運動休閒類"],
    "金融保險":     ["金融業"],
    "生技醫療":     ["生技醫療業"],
    "資訊服務":     ["資訊服務業"],
    "電子零組件":   ["電子零組件業"],
    "電子通路":     ["電子通路業"],
    "電腦及週邊":   ["電腦及週邊設備業"],
    "通信網路":     ["通信網路業"],
    "油電燃氣":     ["油電燃氣業"],
}

# 掃描時排除的板塊（你的策略不交易這些）
_EXCLUDED_SECTORS = {
    "存託憑證",      # 外國公司 TDR，交易規則不同
    "創新板股票",    # 風險高、流動性差
    "其他",          # FinMind 雜項分類，定義不清
}

def _normalize_sector_map(sector_map: dict) -> dict:
    """把重複/別名的板塊合併到標準名稱，去除重複股票"""
    # 建立 別名 → 標準名 的反查表
    alias_to_std = {}
    for std, aliases in _SECTOR_ALIASES.items():
        for a in aliases:
            alias_to_std[a] = std

    merged = {}
    for sector, tickers in sector_map.items():
        std_name = alias_to_std.get(sector, sector)
        if std_name not in merged:
            merged[std_name] = []
        merged[std_name].extend(tickers)

    # 每個板塊內去除重複股票
    for sec in merged:
        merged[sec] = sorted(set(merged[sec]))

    # ✅ 排除不適合交易的板塊（流動性差/風險高/定義不清）
    for excluded in _EXCLUDED_SECTORS:
        merged.pop(excluded, None)

    return merged


def get_sector_map(force_update=False):
    """
    主程式呼叫的入口。若有快取則讀快取，若無或強制更新則 Call API。
    讀快取時也套用正規化，合併重複板塊。
    """
    cache_file = 'sector_map_cache.json'
    if not force_update and os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return _normalize_sector_map(raw)
    else:
        return fetch_dynamic_sector_map(save_cache=True, cache_file=cache_file)

if __name__ == "__main__":
    # 單獨測試此腳本用
    my_map = get_sector_map(force_update=True)
    print(f"測試印出前 3 個板塊: {list(my_map.keys())[:3]}")