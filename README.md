# 📈 Bitcoin Price Prediction Using Market Sentiment and Time Series Models

## 1. Project Overview

This project aims to build a **predictive system for Bitcoin price movements** by combining:

- **Market sentiment analysis** derived from news and thematic keywords
- **Time series forecasting** based solely on historical price data

The core idea is to **compare, complement, and validate** different modeling paradigms in order to better understand:
- the intrinsic dynamics of Bitcoin price
- the added value of market sentiment information

Two main models are developed:
1. A **supervised regression model** using sentiment-based features
2. A **time series model (ARIMA / SARIMA)** using historical Bitcoin prices

---

## 2. Bitcoin as a Financial Asset: Theoretical Context

Bitcoin is a **decentralized digital asset** designed to operate without a central authority. Unlike traditional fiat currencies, Bitcoin follows a **strict and transparent monetary policy**, enforced by its protocol.

### 2.1 Key Characteristics of Bitcoin

- Maximum supply capped at **21 million BTC**
- New bitcoins are introduced through **mining**
- Mining rewards are reduced approximately every four years in events known as **halvings**
- The network is **decentralized**, maintained by distributed nodes and miners

This controlled supply model contrasts sharply with fiat currencies, where supply is regulated by central banks.

---

### 2.2 Market Cycles: Halving, Bull Markets, and Crypto Winters

Historically, Bitcoin has exhibited long-term cycles influenced by:
- **Halving events** (supply shocks)
- Speculative adoption phases
- Prolonged bearish periods commonly referred to as **crypto winters**

These cycles introduce:
- strong non-stationarity
- regime changes
- extreme volatility

As a result, Bitcoin price modeling is particularly challenging and benefits from **multiple modeling perspectives**.

---

## 3. Dataset Design and Temporal Logic

### 3.1 Granularity

- The dataset has an **hourly granularity**
- Each row represents **one hour of data**
- This resolution offers a balance between:
  - capturing short-term dynamics
  - controlling noise
  - computational feasibility

---

### 3.2 Feature Engineering and Temporal Alignment

A critical design decision in this project is the **temporal shift applied to sentiment features**:

- **Sentiment features are computed at time _t-1_**
- **Bitcoin price (target variable) is defined at time _t_**

This ensures:
- no data leakage
- realistic production conditions
- causal consistency

In a real deployment scenario, this allows the system to:
> use news and sentiment from the last hour to predict the Bitcoin price for the next hour.

---

## 4. Model 1: Supervised Regression with Sentiment Features

### 4.1 Objective

The supervised learning model aims to answer the following question:

> *Does aggregated market sentiment help explain and predict short-term Bitcoin price movements beyond pure price inertia?*

---

### 4.2 Input Features

The feature set combines **market sentiment indicators** with **lagged Bitcoin prices**, reflecting both:
- external market perception
- intrinsic price dynamics

Lagged prices are selected to reflect:
- short-term momentum (hours)
- intraday behavior
- daily trading cycles commonly used in crypto markets

---

### 4.3 Target Variable

- Bitcoin price at time _t_

---

### 4.4 Example: Supervised Learning Dataset Structure

Each row represents **one hour of data**.  
Sentiment features are shifted to time *t-1*, while the target corresponds to time *t*.

| timestamp           | sentiment_fear_t-1 | sentiment_greed_t-1 | sentiment_regulation_t-1 | btc_price_t-1 | btc_price_t-4 | btc_price_t-12 | btc_price_t-24 | btc_price_t |
|---------------------|--------------------|---------------------|---------------------------|---------------|---------------|----------------|----------------|-------------|
| 2023-06-01 10:00:00 | 0.12               | -0.34               | 0.05                      | 27150         | 27080         | 26940          | 26810          | 27210       |
| 2023-06-01 11:00:00 | 0.08               | -0.21               | 0.02                      | 27210         | 27120         | 27010          | 26920          | 27180       |
| 2023-06-01 12:00:00 | -0.15              | 0.10                | -0.04                     | 27180         | 27150         | 27080          | 27010          | 27340       |

**Target variable:** `btc_price_t`

**Feature groups:**
- Sentiment indicators at *t-1*
- Lagged prices capturing:
  - immediate momentum (`t-1`)
  - short-term structure (`t-4`)
  - half-day effects (`t-12`)
  - daily trading cycle (`t-24`)

This combination reflects how Bitcoin traders typically analyze price action across multiple time horizons.

---

### 4.5 Modeling Approach

This problem is treated as a **tabular regression task**, enabling the use of:
- linear regression as a baseline
- tree-based regression models
- regularized regression techniques

Model performance is evaluated using standard regression metrics.

---

## 5. Model 2: Time Series Forecasting (ARIMA / SARIMA)

### 5.1 Motivation

The time series model serves as a **price-only baseline**, independent of sentiment data.

It answers the question:
> *How well can Bitcoin price be predicted using only its own historical behavior?*

---

### 5.2 ARIMA as Baseline Model

ARIMA (AutoRegressive Integrated Moving Average) models:
- trends
- short-term dependencies
- autocorrelation structures

ARIMA is selected as the **baseline time series model** because it is:
- interpretable
- widely used in financial forecasting
- a strong reference point for comparison

---

### 5.3 Extending to SARIMA: Capturing Seasonality

With hourly data, Bitcoin prices may exhibit:
- **daily behavioral patterns (24-hour cycles)**
- **weekly effects driven by market participation**

SARIMA extends ARIMA by explicitly modeling **seasonal components**.

Given the stochastic and speculative nature of Bitcoin:
- ARIMA is used first
- SARIMA is introduced only if it demonstrates clear improvements

This approach balances:
- model complexity
- interpretability
- robustness to regime changes

---

### 5.4 Example: Time Series Dataset Structure

The time series model uses **only historical Bitcoin prices**, indexed by time.

| timestamp           | btc_price |
|---------------------|-----------|
| 2023-06-01 08:00:00 | 27080     |
| 2023-06-01 09:00:00 | 27120     |
| 2023-06-01 10:00:00 | 27150     |
| 2023-06-01 11:00:00 | 27210     |

- The timestamp column is used as the time index
- No exogenous variables are included in the baseline ARIMA model

---

## 6. Model Comparison, Competition, and Complementarity

Rather than selecting a single “best” model, the project explores multiple interaction strategies:

### 6.1 Direct Competition

- Forecasting accuracy is compared using metrics such as RMSE and MAE
- Performance stability across different market phases is evaluated

---

### 6.2 Complementary Perspectives

- Time series models capture intrinsic price dynamics
- Sentiment-based models capture external market perception

Together, they provide a more comprehensive understanding of Bitcoin price behavior.

---

### 6.3 Confirmation Logic (Conceptual)

- If both models predict the same direction, confidence increases
- Divergent predictions signal higher uncertainty

This logic aligns with real-world decision-support systems.

---

## 7. Repository Structure and Collaboration

The project uses a collaborative Git workflow with multiple branches:
- `main`: stable integration branch
- individual branches for each contributor

This structure reflects best practices commonly used in professional data science projects.

---

## 8. Project Status and Future Work

This repository represents an **early-stage version** of the project.

Planned extensions include:
- advanced feature engineering
- regime-aware modeling
- ensemble strategies
- production-oriented pipelines

---

## 9. Final Remarks

This project focuses not only on predictive accuracy, but also on:
- understanding Bitcoin as an asset
- respecting temporal causality
- comparing modeling paradigms
- designing solutions with real-world deployment in mind
