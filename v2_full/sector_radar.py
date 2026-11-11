# sector_radar.py
"""
板塊宏觀雷達：用月營收 YoY 動能替代原本的股價動能作為板塊評分依據。
降級策略：FinMind 資料不足時，自動 fallback 到股價 20 日動能。
"""
import numpy as np
import pandas as pd
import logging
import datetime

logger = logging.getLogger(__name__)


class SectorRadar:
    def __init__(self, sector_map: dict):
        self.sector_map = sector_map

    # ==========================================
    # 主入口：用月營收動能評分
    # ==========================================
    def get_top_sectors_tickers(
        self,
        raw_data_dict: dict,
        top_n: int = 3,
        min_yoy: float = 10.0,
        use_revenue: bool = True,
    ) -> list:
        """
        回傳最強 top_n 個板塊裡，所有標的的 ticker list。
        評分邏輯：
          主力：複合評分（股價動能 × 月營收 YoY）
          降級：純股價動能（FinMind 資料不足時）
        """
        if use_revenue:
            # 有 K 線就用複合評分（股價動能 × 月營收），沒有就用純月營收評分
            if raw_data_dict:
                scores = self._score_composite(raw_data_dict, min_yoy=min_yoy)
                source = "複合評分（股價動能 × 月營收 YoY）"
            else:
                scores = self._score_by_revenue(min_yoy=min_yoy)
                source = "純月營收 YoY 評分（無 K 線）"
        else:
            scores = {}

        # 評分不足 3 個板塊有資料時，才降級到純股價動能（需要 K 線）
        if len(scores) < 3:
            if use_revenue:
                logger.warning(f"月營收評分資料不足（{len(scores)} 個板塊），嘗試股價動能")
            if raw_data_dict:
                scores = self._score_by_price(raw_data_dict)
                source = "股價 60 日動能（降級）"

        if not scores:
            logger.warning("雷達無法評分任何板塊，回傳全部標的")
            return list({t for tickers in self.sector_map.values() for t in tickers})

        sorted_sectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_sectors    = [s[0] for s in sorted_sectors[:top_n]]

        logger.info(f"🏆 [{source}] 宏觀雷達選定最強板塊: {top_sectors}")
        for sec, score in sorted_sectors[:top_n]:
            logger.info(f"   {sec}: {score:.4f}")

        target_tickers = list({t for sec in top_sectors for t in self.sector_map.get(sec, [])})
        return target_tickers

    # ==========================================
    # 評分方法 1：月營收 YoY（純營收，備用）
    # ==========================================
    def _score_by_revenue(self, min_yoy: float) -> dict:
        """板塊分數 = 板塊內通過 min_yoy 的股票比例 × 平均 YoY
        只讀快取，不打 FinMind（掃描時用，避免限流）"""
        try:
            from revenue_fetcher import get_sector_revenue_score
        except ImportError:
            logger.error("revenue_fetcher 未安裝，無法使用月營收評分")
            return {}

        scores = {}
        for sector, tickers in self.sector_map.items():
            if not tickers:
                continue
            score = get_sector_revenue_score(tickers, min_yoy=min_yoy, cache_only=True)
            if score > 0:
                scores[sector] = score
        return scores

    # ==========================================
    # 評分方法 2：股價 20 日動能（降級備援）
    # ==========================================
    def _score_by_price(self, raw_data_dict: dict) -> dict:
        """板塊內個股近 20 日漲幅的平均值"""
        scores = {}
        for sector, tickers in self.sector_map.items():
            mom_list = []
            for t in tickers:
                if t in raw_data_dict and not raw_data_dict[t].empty:
                    df = raw_data_dict[t]
                    if len(df) >= 20:
                        mom = (df["Close"].iloc[-1] / df["Close"].iloc[-20]) - 1
                        mom_list.append(mom)
            if mom_list:
                scores[sector] = float(np.mean(mom_list))
        return scores

    # ==========================================
    # 評分方法 3：複合評分（主力）
    # 股價動能 × 月營收 YoY，兩個訊號都強才入選
    # ==========================================
    def _score_composite(
        self,
        raw_data_dict: dict,
        min_yoy: float = 10.0,
        price_window: int = 60,    # 股價動能看 60 日（季線）
        rev_weight: float = 0.5,   # 月營收權重
        price_weight: float = 0.5, # 股價動能權重
    ) -> dict:
        """
        複合板塊評分：
          revenue_score = 通過 min_yoy 比例 × 平均 YoY（標準化到 0-1）
          price_score   = 板塊平均 N 日漲幅（標準化到 0-1）
          composite     = rev_weight × revenue_score + price_weight × price_score

        兩個訊號都弱（如建材：YoY 爆衝但股價沒動）→ 複合分低
        兩個訊號都強（如半導體多頭）→ 複合分高
        """
        try:
            from revenue_fetcher import get_sector_revenue_score
            has_revenue = True
        except ImportError:
            has_revenue = False
            logger.warning("revenue_fetcher 不可用，複合評分改用純股價動能")

        # ── Step 1：計算各板塊的兩個原始分數 ──
        raw_rev   = {}
        raw_price = {}

        for sector, tickers in self.sector_map.items():
            if not tickers:
                continue

            # 月營收分數
            if has_revenue:
                rev_score = get_sector_revenue_score(tickers, min_yoy=min_yoy)
                if rev_score > 0:
                    raw_rev[sector] = rev_score

            # 股價動能分數
            mom_list = []
            for t in tickers:
                if t in raw_data_dict and not raw_data_dict[t].empty:
                    df = raw_data_dict[t]
                    if len(df) >= price_window:
                        mom = (df["Close"].iloc[-1] / df["Close"].iloc[-price_window]) - 1
                        mom_list.append(mom)
            if mom_list:
                raw_price[sector] = float(np.mean(mom_list))

        if not raw_rev and not raw_price:
            return {}

        # ── Step 2：各自標準化到 0-1（min-max）──
        def _normalize(d: dict) -> dict:
            if not d:
                return {}
            vals = list(d.values())
            lo, hi = min(vals), max(vals)
            if hi == lo:
                return {k: 1.0 for k in d}
            return {k: (v - lo) / (hi - lo) for k, v in d.items()}

        norm_rev   = _normalize(raw_rev)
        norm_price = _normalize(raw_price)

        # ── Step 3：合併計算複合分數 ──
        all_sectors = set(norm_rev) | set(norm_price)
        composite   = {}
        for sector in all_sectors:
            r = norm_rev.get(sector, 0.0)
            p = norm_price.get(sector, 0.0)
            # 兩個都有資料才計算，否則只用有資料的那個（權重補足）
            if sector in norm_rev and sector in norm_price:
                composite[sector] = round(rev_weight * r + price_weight * p, 4)
            elif sector in norm_rev:
                composite[sector] = round(r * 0.7, 4)   # 只有營收，打七折
            else:
                composite[sector] = round(p * 0.5, 4)   # 只有股價，打五折

            logger.info(
                f"  📊 {sector}: rev={r:.2f} price={p:.2f} → composite={composite[sector]:.3f}"
            )

        return composite

    # ==========================================
    # 輔助：取得板塊評分明細（供 UI 顯示用）
    # ==========================================
    def get_sector_scores_df(self, raw_data_dict: dict = None, min_yoy: float = 10.0):
        """回傳所有板塊複合評分的 DataFrame，供 Tab 1 顯示排名"""
        scores = self._score_composite(raw_data_dict or {}, min_yoy=min_yoy)
        if not scores:
            return pd.DataFrame()
        rows = [{"板塊": k, "複合動能分數": v} for k, v in
                sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        return pd.DataFrame(rows)

    # ==========================================
    # 回溯驗證：雷達在歷史各時間點的選擇
    # ==========================================
    def backtest_radar_selections(
        self,
        fundamental_history: dict,
        start_year: int = 2016,
        end_year: int = None,
        top_n: int = 3,
        min_yoy: float = 10.0,
        freq: str = "Q",
        price_history: dict = None,  # {ticker: df} 完整 K 線，供計算歷史股價動能
    ) -> pd.DataFrame:
        """
        回溯驗證雷達在歷史各時間點會選出哪些板塊。
        用當時已公告的月營收 + 當時股價動能（複合評分），無前視偏差。
        """
        from revenue_fetcher import get_fundamental_at_date

        if end_year is None:
            end_year = datetime.date.today().year

        dates = pd.date_range(
            start=f"{start_year}-01-01",
            end=f"{end_year}-12-31",
            freq="QS" if freq == "Q" else "YS"
        )

        records = []

        for query_date in dates:
            # ── 月營收板塊分數（歷史時間點）──
            rev_scores = {}
            for sector, tickers in self.sector_map.items():
                yoys = []
                for raw_ticker in tickers:
                    ticker = str(raw_ticker).replace(".TW", "")
                    fund_data = fundamental_history.get(raw_ticker, {})
                    rev_history = fund_data.get("rev", pd.DataFrame())
                    if rev_history.empty:
                        continue
                    fund = get_fundamental_at_date(
                        ticker, query_date,
                        rev_history=rev_history,
                        margin_history=pd.DataFrame(),
                    )
                    yoy = fund.get("latest_yoy")
                    if yoy is not None:
                        yoys.append(min(yoy, 100.0))   # ✅ YoY 封頂 100%，防景氣循環股灌水

                if not yoys:
                    continue
                pass_rate = sum(1 for y in yoys if y >= min_yoy) / len(yoys)
                avg_yoy   = float(np.median(yoys))     # ✅ 中位數而非平均，防少數爆衝股灌水
                score     = pass_rate * avg_yoy
                if score > 0:
                    rev_scores[sector] = score

            # ── 股價動能板塊分數（歷史時間點）──
            price_scores = {}
            if price_history:
                for sector, tickers in self.sector_map.items():
                    mom_list = []
                    for t in tickers:
                        if t not in price_history:
                            continue
                        df = price_history[t]
                        # 找 query_date 之前 60 個交易日
                        past = df[df.index <= query_date]
                        if len(past) >= 60:
                            mom = (past['Close'].iloc[-1] / past['Close'].iloc[-60]) - 1
                            mom_list.append(mom)
                    if mom_list:
                        price_scores[sector] = float(np.mean(mom_list))

            # ── 複合評分（標準化後合併）──
            def _norm(d):
                if not d:
                    return {}
                vals = list(d.values())
                lo, hi = min(vals), max(vals)
                if hi == lo:
                    return {k: 1.0 for k in d}
                return {k: (v - lo) / (hi - lo) for k, v in d.items()}

            norm_rev   = _norm(rev_scores)
            norm_price = _norm(price_scores)
            all_sectors = set(norm_rev) | set(norm_price)

            composite = {}
            for sector in all_sectors:
                r = norm_rev.get(sector, 0.0)
                p = norm_price.get(sector, 0.0)
                if sector in norm_rev and sector in norm_price:
                    composite[sector] = round(0.5 * r + 0.5 * p, 4)
                elif sector in norm_rev:
                    composite[sector] = round(r * 0.7, 4)
                else:
                    composite[sector] = round(p * 0.5, 4)

            if not composite:
                records.append({
                    "date": query_date.date(),
                    "top_sectors": [],
                    "sector_scores": {},
                    "top_tickers": [],
                    "data_coverage": 0,
                })
                continue

            sorted_sectors = sorted(composite.items(), key=lambda x: x[1], reverse=True)
            top_sectors    = [s[0] for s in sorted_sectors[:top_n]]
            top_tickers    = list({t for sec in top_sectors for t in self.sector_map.get(sec, [])})

            records.append({
                "date":          query_date.date(),
                "top_sectors":   top_sectors,
                "sector_scores": composite,
                "top_tickers":   top_tickers,
                "data_coverage": len(composite),
            })

            logger.info(f"{query_date.date()} 複合雷達: {top_sectors}")

        return pd.DataFrame(records)
        """
        回溯驗證雷達在歷史各時間點會選出哪些板塊。
        用當時已公告的月營收（無前視偏差）計算板塊分數。

        回傳 DataFrame，columns:
            date, top_sectors（list str）, sector_scores（dict）,
            top_tickers（list str）
        """
        from revenue_fetcher import get_fundamental_at_date

        if end_year is None:
            end_year = datetime.date.today().year

        # 建立時間序列
        if freq == "Q":
            dates = pd.date_range(
                start=f"{start_year}-01-01",
                end=f"{end_year}-12-31",
                freq="QS"  # 每季初
            )
        else:
            dates = pd.date_range(
                start=f"{start_year}-01-01",
                end=f"{end_year}-12-31",
                freq="YS"
            )

        records = []

        for query_date in dates:
            sector_scores = {}

            for sector, tickers in self.sector_map.items():
                yoys = []
                for raw_ticker in tickers:
                    ticker = str(raw_ticker).replace(".TW", "")
                    fund_data = fundamental_history.get(raw_ticker, {})
                    rev_history = fund_data.get("rev", pd.DataFrame())

                    if rev_history.empty:
                        continue

                    fund = get_fundamental_at_date(
                        ticker, query_date,
                        rev_history=rev_history,
                        margin_history=pd.DataFrame(),
                    )
                    yoy = fund.get("latest_yoy")
                    if yoy is not None:
                        yoys.append(yoy)

                if not yoys:
                    continue

                # 板塊分數：通過門檻的比例 × 平均 YoY
                pass_rate = sum(1 for y in yoys if y >= min_yoy) / len(yoys)
                avg_yoy   = sum(yoys) / len(yoys)
                score     = round(pass_rate * avg_yoy, 2)
                if score > 0:
                    sector_scores[sector] = score

            if not sector_scores:
                records.append({
                    "date":          query_date.date(),
                    "top_sectors":   [],
                    "sector_scores": {},
                    "top_tickers":   [],
                    "data_coverage": 0,
                })
                continue

            sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
            top_sectors    = [s[0] for s in sorted_sectors[:top_n]]
            top_tickers    = list({t for sec in top_sectors for t in self.sector_map.get(sec, [])})

            records.append({
                "date":          query_date.date(),
                "top_sectors":   top_sectors,
                "sector_scores": sector_scores,
                "top_tickers":   top_tickers,
                "data_coverage": len(sector_scores),
            })

            logger.info(f"{query_date.date()} 雷達選板塊: {top_sectors} (有資料板塊數: {len(sector_scores)})")

        return pd.DataFrame(records)