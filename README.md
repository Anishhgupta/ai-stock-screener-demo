\# AI/ML Real-Time Stock Screener (NSE)



\*\*▶ Live demo: https://ai-stock-screener-demo-hwmhgdfx33svxcf78zrsjx.streamlit.app/\*\*



A Python real-time screening and paper-trading system for NSE equities. It detects SMMA20/SMMA120 crossovers, reads Last Traded Quantity (LTQ) dynamics and 5-level Bid/Ask market depth, and passes each signal through a machine-learned \*\*ACCEPT / AVOID\*\* filter. The goal is to catch good trades and, more importantly, to avoid losing ones.



!\[Demo dashboard](docs/demo\_dashboard.png)



> \*\*Paper trading only.\*\* No real orders are ever placed. Nothing here is investment advice.



\## Demo vs. real



The hosted demo runs the \*\*same engine\*\* as the live system (crossover detection, feature engineering, ML filter, paper-trading books) on a \*\*synthetic price feed\*\*, so it works instantly with no broker credentials.



\- Prices in the demo are a random walk, so no model can have a real edge there. The demo shows the pipeline mechanics, not trading performance.

\- Real results come from live Fyers data collected over multiple trading days (Aug 31 to Sep 14, 2026): 78 real crossovers were combined to train the first properly sized model, with AUC improving from 0.500 to 0.582. That is a small sample and a modest result, reported as it is.



\## How it works



1\. \*\*Ingest:\*\* live ticks from Fyers (WebSocket + REST historical) or Angel One (SmartAPI), or a mock broker for the demo.

2\. \*\*Warm start:\*\* SMMA20/SMMA120 are primed from historical candles, so signals are ready immediately instead of after about two hours of live data.

3\. \*\*Detect:\*\* `CrossoverEngine` flags SMMA20/SMMA120 crossovers.

4\. \*\*Featurize:\*\* 11 broker-agnostic features (SMMA, LTQ acceleration, 5-level bid/ask imbalance and more), so data from different days and brokers can be combined for training.

5\. \*\*Decide:\*\* a `GradientBoostingClassifier` scores each crossover and returns ACCEPT or AVOID with a probability and human-readable reasons.

6\. \*\*Paper trade:\*\* two books run side by side from the same feed.

&#x20;  - \*\*Basic:\*\* enters every crossover.

&#x20;  - \*\*AI-Filtered:\*\* enters only ACCEPTed signals and exits early when the position monitor flags deterioration.



Both books use stop-loss, target and max-hold exits.



\## Run it yourself



```

git clone https://github.com/Anishhgupta/ai-ml-stock-screener.git

cd ai-ml-stock-screener

pip install -r requirements.txt

streamlit run demo\_app.py

```



The demo needs no credentials. To retrain the demo model: `python -m demo.train\_demo\_model`.



\### Live mode (bring your own broker)



Each user runs their own instance with their own Fyers or Angel One credentials. There is no shared data feed, which avoids redistributing broker data and regulated-advice territory.



1\. Put your broker credentials in a local `.env` (never committed).

2\. Fyers tokens expire daily: run `python fyers\_login.py` each trading day.

3\. Real NSE depth rarely reaches the literal 10 lakh bid/ask quantity in the original spec, so set `SCREENER\_MIN\_DEPTH\_QTY` (for example `2000` on Windows CMD: `set SCREENER\_MIN\_DEPTH\_QTY=2000`) before training or running. Otherwise every signal is silently screened out.

4\. Train on saved ticks with `python -m ml.train --tick-csv <path>`, then run `python main.py`.



A Windows `.exe` can be built with `build\_exe.py` (PyInstaller).



\## Bugs found and fixed



Real debugging, documented rather than hidden:



\- \*\*Fyers message schema:\*\* Fyers sends `SymbolUpdate` (LTP/LTQ/volume) and `DepthUpdate` (bid/ask levels, no LTP) as separate messages. The adapter now subscribes to both and merges them per symbol.

\- \*\*Falsy-zero breakeven bug:\*\* `if p.pnl and p.pnl <= 0` silently dropped exact-breakeven trades (`0.0` is falsy in Python) from both win and loss counts. Now handled with explicit `is not None` checks and a separate breakeven bucket.

\- \*\*Mixed clocks in demo trading:\*\* positions were stamped with wall-clock time but compared against an accelerated simulated clock, so every trade looked held for over the max hold time and closed instantly at breakeven. Entry now uses the triggering tick's timestamp.

\- \*\*Basic book never traded without a model:\*\* trade logic was gated on model availability. Basic now always trades, with a placeholder AVOID decision when no model is loaded.

\- \*\*Blank decision display:\*\* a crossover is a one-tick event, so decision columns were empty on over 99% of ticks. The last real value is now carried forward per symbol.

\- \*\*Noisy mock order book:\*\* independently redrawn depth every tick made imbalance flip randomly and tripped the deterioration monitor. Mock depth is now autocorrelated like a real book.

\- \*\*Model portability:\*\* `joblib` models are not guaranteed to load across scikit-learn versions. Retrain locally or pin matching versions.



\## Tech stack



Python 3.12, scikit-learn, joblib, Streamlit, Plotly, pandas, fyers-apiv3, Angel One SmartAPI, PyInstaller. All storage is file-based CSV, with no database.



\## Roadmap



CI with pytest and GitHub Actions, regression tests for the bugs above, and a multi-user deployment (database, auth, Docker).

