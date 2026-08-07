# Methodology

## Objective

Build a reproducible framework for analyzing how AI-related capital expenditure and adoption affect companies across the economic stack.

The project is **not** designed to label every company that mentions AI as an "AI stock." Instead, each company should have a specific economic mechanism linking AI activity to revenue, margins, backlog, capital expenditure, or strategic value.

## Unit of analysis

The primary unit is the **company**. For public companies, stock-level performance and valuation are added. For private companies, the project tracks competitive relevance without pretending that a public-market return exists.

## Tier definitions

### Tier 1 — Models, assistants, and AI software

A company belongs here when AI models, AI assistants, AI-enabled cloud services, or AI software are directly sold to end users or enterprises.

Typical revenue mechanism:

`AI adoption -> subscriptions / usage / cloud consumption / software upsell`

### Tier 2 — Core AI technology and infrastructure

A company belongs here when its technology is directly required to train, serve, connect, manufacture, or deploy modern AI systems.

Typical revenue mechanism:

`AI compute demand -> accelerators / memory / networking / servers / fabrication / semiconductor equipment`

### Tier 3 — Second-order AI beneficiaries

A company belongs here when AI demand reaches it indirectly through construction of the physical infrastructure needed to operate large-scale compute.

Typical revenue mechanism:

`AI compute buildout -> data-center construction -> power + cooling + grid + optics + electrical equipment`

This tier receives special attention because it is the most likely source of companies whose AI exposure is economically meaningful but not obvious from their industry labels.

## Exposure evidence standard

A company should not receive a high AI-exposure score based only on management using the term "AI." Evidence should preferentially come from:

1. reported AI/data-center revenue;
2. backlog tied to data-center or hyperscaler projects;
3. management disclosure quantifying demand;
4. customer or partnership disclosures;
5. segment growth that can be linked to AI infrastructure;
6. capital expenditure specifically serving AI demand;
7. credible industry evidence where company-level disclosure is unavailable.

## Proposed variables

### Identity

- `company`
- `ticker`
- `exchange`
- `country`
- `sector`
- `industry`
- `public_status`
- `ipo_date`

### AI classification

- `tier`
- `ai_link`
- `ai_exposure_score`
- `non_obviousness_score`
- `exposure_evidence`
- `primary_ai_customer_type`
- `bottleneck_role`

### Market performance

- adjusted price history;
- 1-year total return;
- 3-year total return;
- 5-year total return;
- 10-year total return;
- CAGR;
- annualized volatility;
- maximum drawdown;
- benchmark-relative return.

Companies without enough listing history should be marked explicitly rather than treated as missing observations without explanation.

### Fundamentals

At annual and quarterly frequencies where feasible:

- revenue;
- revenue growth;
- gross margin;
- operating margin;
- EPS;
- free cash flow;
- capex;
- backlog;
- data-center revenue;
- AI revenue;
- valuation multiples.

## Benchmarks

At minimum compare public companies against:

- S&P 500;
- NASDAQ-100 or a broad technology benchmark;
- semiconductor benchmark for Tier 2 where appropriate.

A single 10-year endpoint comparison is useful descriptively but insufficient analytically. Rolling returns and multiple starting dates should be used to reduce start-date dependence.

## Historical-return caveat

A company's stock can have an exceptional ten-year return without AI being the original cause. For example, semiconductor and networking companies may have generated substantial returns before the current generative-AI cycle.

Therefore distinguish:

- **pre-AI-boom return**;
- **AI-boom-period return**;
- **fundamental acceleration during the AI-boom period**.

A useful event boundary can be tested rather than assumed. Candidate dates include the public launch of ChatGPT and subsequent hyperscaler capex acceleration.

## Non-obviousness score

Suggested ordinal scale:

- **1 — Obvious:** AI/model company or household-name accelerator vendor.
- **2 — Visible:** commonly discussed AI beneficiary.
- **3 — Adjacent:** connection is clear after explanation.
- **4 — Hidden:** traditional industry label obscures meaningful AI exposure.
- **5 — Second-order:** connection requires tracing spending through one or more intermediate industries.

The score is not an investment-quality score.

## AI exposure score

A separate 0–5 score should estimate how economically important AI demand is to the business.

Possible rubric:

- **0:** no demonstrated material exposure;
- **1:** experimental or immaterial exposure;
- **2:** identifiable exposure but small relative to company;
- **3:** meaningful contributor to growth;
- **4:** major strategic and financial growth driver;
- **5:** business is predominantly driven by AI demand.

Keeping exposure and non-obviousness separate prevents a hidden but tiny AI connection from being mistaken for a major beneficiary.

## Analysis sequence

### Analysis A — Historical winners

Rank public companies by total return and CAGR over common windows.

### Analysis B — Fundamental acceleration

Measure changes in revenue growth, margins, backlog, and capex before and after the AI infrastructure boom.

### Analysis C — Surprise beneficiaries

Filter for:

- non-obviousness >= 4;
- demonstrated AI/data-center exposure;
- accelerating business fundamentals.

Then investigate the causal mechanism company by company.

### Analysis D — Valuation

Compare current valuation against:

- company history;
- peers;
- growth expectations;
- realized fundamental growth.

This is necessary because the company most exposed to AI is not necessarily the most attractive stock at the current price.

### Analysis E — Supply-chain transmission

Construct a conceptual flow:

`AI applications -> cloud/model compute -> chips -> networking/memory -> racks -> data centers -> cooling/power -> grid/generation`

Test whether investment or demand signals at upstream stages predict fundamentals downstream.

## Reproducibility principles

- Preserve raw source data separately from processed data.
- Record retrieval dates.
- Use adjusted prices for return calculations.
- Avoid silently filling missing history.
- Keep derived metrics in code rather than manually entered spreadsheets.
- Save source URLs and source type for fundamental AI-exposure claims.
- Treat current-company classifications as time-varying data rather than permanent facts.
