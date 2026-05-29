# NeuralRetail — Dataset Catalogue

## Week 1 Source Datasets

| # | Dataset | Source URL | Download Command | Key Schema Columns |
|---|---------|-----------|-----------------|-------------------|
| 1 | **Kaggle RetailRocket** | https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset | `kaggle datasets download -d retailrocket/ecommerce-dataset --unzip -p data/raw/retailrocket` | `event_id` (string), `timestamp` (epoch ms → TimestampType), `visitor_id` (string), `event_type` (string: view/addtocart/transaction), `item_id` (string), `transaction_id` (string, nullable) |
| 2 | **M5 Forecasting** | https://www.kaggle.com/competitions/m5-forecasting-accuracy | `kaggle competitions download -c m5-forecasting-accuracy --unzip -p data/raw/m5` | `item_id` (string), `dept_id` (string), `cat_id` (string), `store_id` (string), `state_id` (string), `d_1`…`d_1941` (int, daily unit sales), `sell_price` (float), `wm_yr_wk` (int, week identifier) |
| 3 | **UCI Online Retail II** | https://archive.ics.uci.edu/ml/datasets/Online+Retail+II | `wget https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx -P data/raw/uci_retail` | `InvoiceNo` (string), `StockCode` (string), `Description` (string), `Quantity` (int), `InvoiceDate` (datetime), `UnitPrice` (float, GBP), `CustomerID` (float → string), `Country` (string) |

## Schema Details

### 1. Kaggle RetailRocket (`data/raw/retailrocket/events.csv`)

```
event_id        STRING         Unique event identifier
timestamp       BIGINT         Unix timestamp in milliseconds
visitor_id      STRING         Anonymous visitor identifier
event_type      STRING         One of: view, addtocart, transaction
item_id         STRING         Product/item identifier
transaction_id  STRING (null)  Set only for event_type=transaction
```

**Notes:**
- ~2.7M events spanning May–Sep 2015
- Convert timestamp: `F.from_unixtime(F.col("timestamp") / 1000).cast("timestamp")`
- ~90% are "view" events; transactions are ~1%

### 2. M5 Forecasting (`data/raw/m5/`)

```
item_id    STRING   e.g. HOBBIES_1_001
dept_id    STRING   e.g. HOBBIES_1
cat_id     STRING   e.g. HOBBIES
store_id   STRING   e.g. CA_1
state_id   STRING   e.g. CA
d_1        INT      Daily unit sales day 1  (2011-01-29)
...
d_1941     INT      Daily unit sales day 1941 (2016-06-19)
sell_price FLOAT    Weekly sell price per item
wm_yr_wk   INT      Walmart year-week identifier
```

**Files:** `sales_train_evaluation.csv`, `sell_prices.csv`, `calendar.csv`

### 3. UCI Online Retail II (`data/raw/uci_retail/online_retail_II.xlsx`)

```
InvoiceNo    STRING    Invoice number; prefix C = cancellation
StockCode    STRING    Product code (5-digit)
Description  STRING    Product description
Quantity     INT       Units per invoice; negative = return
InvoiceDate  DATETIME  Invoice timestamp (UTC)
UnitPrice    FLOAT     Price per unit in GBP
CustomerID   FLOAT     Nullable customer identifier
Country      STRING    Customer country of residence
```

**Notes:**
- Two sheets: Year 2009–2010, Year 2010–2011
- Filter Quantity > 0 and UnitPrice > 0 for sales analysis
- ~1M rows total after combining sheets
