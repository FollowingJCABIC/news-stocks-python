# AI Industry and Stock Tracker Architecture

## Primary objective

The primary goal of this project is to build a living system that tracks **AI-related industries first and individual stocks second**.

The tracker should answer four questions repeatedly:

1. Which parts of the AI economy are strengthening or weakening?
2. Which public companies inside those industries are leading or lagging?
3. What operating evidence connects the industry move to AI demand rather than to a generic market narrative?
4. Are new industries beginning to receive AI-related spending before they are widely thought of as AI industries?

The project therefore treats the AI economy as a transmission network rather than a static watchlist.

`AI adoption -> model demand -> cloud / compute -> networking / memory -> servers -> data centers -> power / cooling / construction -> grid / generation`

A later secondary module can study AI as an income-production technology for small firms, creators, freelancers, and online businesses. That module should feed into the industry tracker only when there is measurable evidence at sector scale.

## Unit of analysis

The hierarchy is:

`Tier -> Industry group -> Company -> Evidence -> Market response`

This is intentionally different from starting with ticker symbols. A company can move for idiosyncratic reasons. A synchronized move across several companies in the same transmission layer is more informative about an industry theme.

The canonical stock universe is `data/company_universe.csv`.

The industry mapping is `data/industry_map.csv`.

## Core tracker modules

### 1. Market module

Prototype source: `yfinance`.

Track for each public company:

- daily adjusted price;
- 1-day, 5-day, 20-day, 60-day, 6-month, 1-year, 3-year, 5-year and 10-year returns where history permits;
- rolling volatility;
- rolling maximum drawdown;
- relative performance versus SPY and a technology benchmark;
- moving-average trend measures;
- industry-group breadth: percent of stocks positive over each window;
- industry-group equal-weight return;
- dispersion within each industry group.

`yfinance` is convenient for a research prototype but should remain behind an abstraction layer so it can later be replaced by a licensed or more authoritative market-data source without changing the analysis code.

### 2. SEC filings and fundamentals module

Primary source: SEC EDGAR / `data.sec.gov`.

The SEC provides RESTful submissions and XBRL Company Facts data for public filers. The tracker should use these feeds for standardized financial history and filing discovery rather than relying only on third-party fundamentals.

Track when available:

- revenue;
- operating income and margin;
- net income;
- operating cash flow;
- capital expenditures;
- free-cash-flow approximations;
- debt and cash;
- segment disclosures where structured data permit;
- filing dates and accession numbers.

A separate text-analysis process should search 10-K, 10-Q and 8-K filings for evidence terms such as:

- artificial intelligence / AI;
- data center;
- accelerated computing;
- liquid cooling;
- high-bandwidth memory / HBM;
- hyperscaler;
- cloud capex;
- backlog;
- power demand;
- switchgear;
- optical / photonics;
- transmission / interconnection.

The important output is not simply an AI keyword count. The program should retain the surrounding passage and classify whether the disclosure describes **revenue, demand, capacity, investment, risk, or marketing language**.

### 3. Industry-demand module

Use public economic data to measure the physical economy surrounding the stock universe.

#### EIA — electricity and generation

Useful for Tier 3 power and grid exposure:

- electricity demand;
- generation by fuel;
- generator capacity;
- regional operating data;
- electricity prices;
- capacity additions.

The EIA provides free open data through its API and bulk files.

#### BEA — industry accounts

Useful for measuring broader industry expansion:

- GDP / value added by industry;
- gross output by industry;
- industry input-output relationships;
- intermediate inputs.

The input-output accounts are especially relevant because this project's central question is how spending flows from one AI layer into suppliers in another layer.

#### FRED / ALFRED

Use as a common interface for macro and industry series such as:

- industrial production;
- semiconductor-related production where available;
- construction spending;
- electricity and utility measures;
- interest rates and credit conditions;
- macro controls.

ALFRED vintages are useful when we later ask whether a signal would have been observable **at the time**, rather than using subsequently revised data.

### 4. AI-adoption module

Primary source: U.S. Census Bureau Business Trends and Outlook Survey (BTOS), supplemented by BLS research and productivity data.

This is where the user's interest in AI creating new forms of income becomes measurable without making anecdotes the core dataset.

The 2026 BTOS AI supplement measures business AI use across industries, geography, firm size, business functions and worker tasks. It can therefore become an **industry adoption signal**.

Possible tracker fields:

- current AI-use rate by sector;
- expected AI-use rate in six months;
- change in adoption rate;
- business functions using AI;
- adoption by firm size;
- sector rank in AI adoption.

This lets us ask whether industries adopting AI more rapidly subsequently show changes in productivity, business formation, software spending, margins, hiring patterns, or the results of public companies serving those industries.

BLS industry productivity and capital-input data can provide another outcome layer.

### 5. Company-evidence module

Some of the most important AI exposure data are not standardized financial fields. The tracker should therefore maintain a structured evidence table.

Suggested schema:

| field | meaning |
|---|---|
| ticker | public ticker |
| evidence_date | when the disclosure became public |
| filing_or_source | 10-K, 10-Q, 8-K, earnings release, investor presentation |
| source_id | accession number or stable identifier |
| industry_group | tracker industry |
| evidence_type | demand, revenue, backlog, capex, capacity, customer, risk |
| ai_link | the mechanism connecting the evidence to AI |
| strength | weak / moderate / strong |
| excerpt | short supporting passage |
| machine_classified | whether the label was generated automatically |
| reviewed | whether a human reviewed the classification |

This table is important for avoiding hindsight. We want to know **when** evidence became visible.

## Industry-level signals

The tracker should calculate signals at the industry-group level before ranking stocks.

### Market strength

Examples:

- equal-weight industry return;
- median stock return;
- percent of constituents outperforming SPY;
- percent above 50- and 200-day moving averages;
- cross-sectional dispersion;
- drawdown from industry high.

### Fundamental acceleration

Examples:

- median year-over-year revenue growth;
- change in operating margin;
- capex growth;
- backlog / RPO growth where relevant;
- number of companies raising capacity or investment.

### External demand

Examples:

- hyperscaler capex growth for Tier 2 and Tier 3 suppliers;
- EIA demand / capacity trends for power industries;
- data-center construction indicators;
- semiconductor investment and production;
- Census AI-adoption rates for application-layer industries.

### Evidence strength

Count and weight disclosures that directly connect operating results to AI-related demand.

Do not turn this directly into a buy/sell signal. It should be a research score showing where the evidence is becoming stronger or weaker.

## Proposed research scores

Keep separate scores rather than one opaque master score:

1. `market_strength_score`
2. `fundamental_acceleration_score`
3. `industry_demand_score`
4. `ai_evidence_score`
5. `valuation_context_score`
6. `non_obviousness_score`

The most interesting research candidates may be industries where demand and operating evidence improve **before** market strength becomes extreme.

## Update cadence

| Data | Suggested cadence |
|---|---|
| Prices / returns | Daily |
| SEC filing discovery | Daily or filing-triggered |
| XBRL fundamentals | After new filings |
| Company evidence text | After new filings / releases |
| Census BTOS | Each release / roughly biweekly when applicable |
| EIA power data | Weekly or monthly depending on series |
| FRED macro / industry data | Per series release |
| BEA industry accounts | Quarterly / release based |
| Full research snapshot | Weekly |

## Program architecture

The notebook phase should evolve into reusable Python modules.

```text
projects/ai_investment_landscape/
├── data/
│   ├── company_universe.csv
│   └── industry_map.csv
├── notebooks/
│   ├── 01_ai_market_returns.ipynb
│   └── 02_industry_signal_map.ipynb
├── outputs/
├── src/
│   ├── universe.py
│   ├── market_data.py
│   ├── sec_data.py
│   ├── industry_data.py
│   ├── filing_text.py
│   ├── signals.py
│   └── snapshots.py
└── TRACKER_ARCHITECTURE.md
```

The final program should have one orchestration command such as:

```bash
python -m ai_tracker update
```

That command would:

1. load the canonical universe;
2. update market prices;
3. discover new filings;
4. update fundamentals;
5. update available macro / industry data;
6. calculate stock signals;
7. aggregate those signals to industries;
8. save a dated snapshot;
9. identify materially changed industries and companies;
10. regenerate dashboard-ready tables.

## Snapshot design

Never overwrite the only copy of a result. Save dated snapshots so we can reconstruct what the system knew at a historical point.

Example:

```text
outputs/snapshots/2026-08-07/
    stock_signals.parquet
    industry_signals.parquet
    evidence.parquet
    data_quality.json
```

This makes later backtesting possible and prevents hindsight leakage.

## Immediate next step

Notebook 02 should establish the industry-first market layer. It should:

1. join `company_universe.csv` to `industry_map.csv`;
2. pull public-stock price histories;
3. compute stock-level momentum and risk metrics;
4. aggregate them into industry-group breadth and equal-weight performance;
5. rank Tier 3 hidden-beneficiary industries;
6. write `outputs/industry_signal_snapshot.csv`;
7. leave clean interfaces for SEC, EIA, BEA, FRED and Census signals to be added next.

After that, Notebook 03 should be the first fundamental/evidence notebook using SEC data.
