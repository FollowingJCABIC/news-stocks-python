# AI Investment Landscape

## Project idea

This project studies the AI economy as an investment ecosystem rather than treating "AI stocks" as a single category.

The central idea is to divide the landscape into three tiers:

1. **AI assistants, models, and software platforms** — companies closest to products such as ChatGPT and Claude.
2. **Core AI technology and infrastructure** — semiconductors, foundries, memory, networking, servers, and semiconductor equipment.
3. **Second-order AI beneficiaries** — power, cooling, electrical equipment, construction, optics, connectors, and other industries whose growth can be driven by AI data-center buildout.

The initial universe contains **15 companies per tier (45 total)**. It deliberately includes companies that are easy to miss if we only search for firms whose descriptions contain the phrase "artificial intelligence."

## Why this project is interesting

The obvious AI investment story is model developers and GPU makers. The less obvious question is:

> **Where does AI spending flow after the model company buys compute?**

An AI data center also requires networking, memory, foundry capacity, electrical distribution, cooling, backup power, grid connections, fiber, optical components, connectors, construction, and maintenance. Some of the strongest business effects may therefore appear in companies that do not look like AI companies at first glance.

The project is designed to distinguish:

- direct AI exposure from indirect exposure;
- public stocks from important private companies;
- historical winners from companies with current AI-driven business exposure;
- genuine operating exposure from companies using AI mainly as marketing language;
- obvious names from less-obvious beneficiaries.

## Initial research questions

1. Which companies in each tier have produced the strongest 1-, 3-, 5-, and 10-year shareholder returns?
2. How much of each company's recent revenue growth can plausibly be connected to AI demand?
3. Which less-obvious companies have experienced accelerating revenue, earnings, backlog, or capital expenditure because of AI infrastructure?
4. Has the market already priced the expected AI growth into each stock?
5. Which parts of the AI supply chain have the strongest margins, scarcity, switching costs, or bottlenecks?
6. Is AI-related performance concentrated in a few dominant firms or spreading outward through the supply chain?
7. Do second-order beneficiaries outperform after large increases in hyperscaler capital expenditure?
8. Which companies would have been identifiable as AI beneficiaries *before* their strongest stock-price moves?

## Three-tier universe

The canonical company list is stored in [`data/company_universe.csv`](data/company_universe.csv).

### Tier 1 — Models, assistants, and AI software

Examples include OpenAI, Anthropic, Alphabet, Microsoft, Meta, Amazon, Palantir, Salesforce, and ServiceNow. Private firms are retained because understanding the competitive landscape matters even when their shares cannot currently be purchased on public exchanges.

### Tier 2 — Core AI technology

This tier covers the physical and computational stack: NVIDIA and AMD accelerators, TSMC manufacturing, Broadcom and Marvell networking/custom silicon, Micron memory, Arista networking, Supermicro/Dell/HPE servers, ASML and Applied Materials semiconductor equipment, and related technology.

### Tier 3 — The less-obvious AI economy

This is the project's distinctive tier. It includes companies such as Vertiv, Eaton, Quanta Services, GE Vernova, Constellation Energy, Comfort Systems USA, EMCOR, Corning, Coherent, Amphenol, and Celestica.

They are not primarily known as AI companies. Their potential exposure comes from the physical requirements created by increasingly large AI clusters and data centers.

## Planned analysis

### Stage 1 — Universe and data quality

- Validate tickers, exchanges, IPO dates, and public/private status.
- Define AI exposure categories consistently.
- Collect adjusted stock prices and benchmark data.
- Document survivorship-bias and listing-history limitations.

### Stage 2 — Historical performance

For public companies, calculate:

- total return;
- CAGR;
- annualized volatility;
- maximum drawdown;
- Sharpe-like risk-adjusted measures;
- performance versus S&P 500 and NASDAQ benchmarks;
- rolling returns rather than relying on a single start date.

### Stage 3 — Fundamental AI exposure

Track where available:

- revenue and revenue growth;
- operating margin and free cash flow;
- capital expenditures;
- data-center or AI-related revenue disclosures;
- backlog / remaining performance obligations;
- hyperscaler customer concentration;
- valuation multiples.

### Stage 4 — AI-capex transmission

Investigate whether hyperscaler AI capital expenditure propagates through the tiers:

`model demand -> compute -> networking/memory -> data centers -> power/cooling/construction`

This creates a possible empirical question: **do changes in hyperscaler capex lead operating results or stock returns in Tier 2 and Tier 3 firms?**

### Stage 5 — "Companies you wouldn't think of first"

Build a separate ranking emphasizing firms where:

- the company's traditional industry label does not say AI;
- management disclosures show meaningful data-center/AI demand;
- fundamentals accelerated during the AI infrastructure buildout;
- there is a defensible mechanism connecting AI spending to the business.

## Important limitations

This is an exploratory investment-research project, not investment advice. Historical returns do not establish that AI caused the return, and a company can benefit from AI while still being a poor investment at a particular valuation. Conversely, a company can be an excellent business without having meaningful AI exposure.

The project should therefore keep **business exposure, stock performance, and valuation** as separate concepts.
