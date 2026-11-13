\# Setup guide



This project is \*\*paper trading only\*\*. It never places an order. Each person runs their own copy with their own broker login: nothing is shared, and nothing leaves your computer except the calls to your broker's own API.



\## What you need



\- Docker Desktop (to run it), or Python 3.12 if you prefer to run it without Docker.

\- For live prices: a Fyers account with an API app (or an Angel One SmartAPI account). Without one you can still run the demo and the mock feed.



\## 1. Try it without any login



```

docker compose --profile demo up --build

```



Open http://localhost:8501. It runs on simulated prices, so it shows how the system works, not how a strategy performs. Stop it with Ctrl+C, then `docker compose --profile demo down`.



\## 2. Run the guided setup



```

python setup\_wizard.py

```



It asks which broker you use, your credentials (secrets are typed hidden and never shown again), a minimum depth, and optionally your trading settings and your own watchlist. It writes a `.env` file in the project folder and keeps any other lines already in it. The wizard needs only Python 3.12, no extra packages.



To see later what is set, without showing any values:



```

python setup\_wizard.py --check

```



For the daily Fyers login you also need the packages: `pip install -r requirements.txt`.



\## 3. Fyers: create your app (one time)



1\. Go to https://myapi.fyers.in/dashboard and choose Create App.

2\. Set a Redirect URL, for example https://www.google.com. It only has to match what you type into the wizard.

3\. Copy the App ID and the Secret ID into the wizard.



\## 4. Every trading morning (Fyers)



```

python fyers\_login.py

```



Open the URL it prints, log in, then copy the full address from your browser's address bar (even if the page looks blank) and paste it back. The day's token is saved into `.env` for you and is not printed. Fyers tokens last one trading day. Then recreate the containers so they read it:



```

docker compose --profile live down

docker compose --profile live up -d

docker compose --profile live logs --tail 30 screener

```



The log should say `Warm-started SMMA for` followed by your watchlist size. Open http://localhost:8502.



\*\*Never add `-v` to `down`.\*\* That deletes your saved trade history.



\## 5. What it does during the day



\- It trades on paper from when you start it until \*\*15:15 IST\*\*. Since August 2026, continuous trading in F\&O stocks ends at 15:15 with a closing auction after it, and Fyers starts squaring off intraday positions around then. Anything still open at 15:15 is closed on paper at the last price seen before it.

\- It stops by itself after the feed has been silent for two minutes (about 16:00 IST).

\- Two paper books run side by side: Basic enters every crossover, Filtered enters only what the model ACCEPTs.



\## 6. Your data



\- Trades and decisions live in a Docker volume that survives `down`.

\- On the dashboard, "Results and export" downloads your closed trades and decision log as CSV.

\- To take a backup copy of the database into the project folder (Windows CMD):



```

docker compose --profile live run --rm -v "%cd%:/out" dashboard python -c "import sqlite3;s=sqlite3.connect('/db/screener.db');d=sqlite3.connect('/out/screener\_backup.db');s.backup(d);d.close()"

```



\## 7. Changing settings later



Run `python setup\_wizard.py` again. It only changes what you change. Then `docker compose --profile live down` and `up -d`. Changing the stop, target, amount or hold time changes what the experiment measures, so results saved before and after are not comparable.



\## Troubleshooting



\- \*\*"Token is Invalid or Expired"\*\* in the screener log: run `python fyers\_login.py` again, then recreate the containers.

\- \*\*The dashboard says "Not updating":\*\* the screener has stopped, or the market is closed.

\- \*\*No stock ever gets a verdict:\*\* run `python setup\_wizard.py --check`. If `SCREENER\_MIN\_DEPTH\_QTY` is missing, real NSE order books are too thin for the original 10 lakh default, so set 2000.

\- \*\*"port is already allocated":\*\* another program is using 8501 or 8502. Close it, or stop the other containers first.

\- \*\*"failed to connect to the docker API":\*\* Docker Desktop is not running. Start it and wait until it says it is running.



\## Safety and limits



\- This is paper trading. Nothing here is investment advice, and the model's results so far are indistinguishable from chance. Read the "What I found" section of the README before trusting any number.

\- The dashboard and demo are published on `127.0.0.1` only and have no login. Do not expose them to a network.

\- `.env` holds your secrets. Git ignores it. Never share or commit it. Fyers' terms ask you to keep your credentials to yourself, which is why this project is built to run on your own computer and does not offer a shared hosted version.

