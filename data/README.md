# data/

Git ignores all files in this folder except this README. Do not commit datasets.

## 1. Synthetic data (default, no download)

`autovalue generate` writes two CSV files here. The tests and `autovalue demo` make the same
tables in memory, so they need no file.

| File | Rows (default) | Generator | Seed |
|---|---|---|---|
| `listings.csv` | 4,000 + defects | `autovalue.synthetic.make_listings` | `--seed` (42) |
| `deliveries.csv` | 4,000 | `autovalue.synthetic.make_deliveries` | `--seed` + 1 |

The synthetic tables are not real market data. The extra columns `oracle_price` and
`oracle_minutes` hold the label without noise. The evaluation uses them to show the noise floor.

## 2. Real data (optional)

### 2.1 Used-car listings

| Item | Value |
|---|---|
| Source | "Vehicle dataset" from CarDekho, published on Kaggle |
| URL | https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho |
| Terms | Read the licence on the Kaggle page before you use the data |
| File | `CAR DETAILS FROM CAR DEKHO.csv` |
| Columns used | `name, year, selling_price, km_driven, fuel, transmission, owner` |
| Adapter | `--adapter cardekho` (`autovalue.loaders.from_cardekho`) |

The file has no listing date. The adapter gives all rows one date, so the split is a seeded
random split, and the report says so.

```bash
autovalue train --task price --csv "data/CAR DETAILS FROM CAR DEKHO.csv" --adapter cardekho
```

### 2.2 Delivery orders

| Item | Value |
|---|---|
| Source | "Amazon Delivery Dataset", published on Kaggle |
| URL | https://www.kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset |
| Terms | Read the licence on the Kaggle page before you use the data |
| File | `amazon_delivery.csv` |
| Columns used | `Order_ID, Store_Latitude, Store_Longitude, Drop_Latitude, Drop_Longitude, Order_Date, Order_Time, Pickup_Time, Weather, Traffic, Vehicle, Area, Delivery_Time` |
| Adapter | `--adapter amazon-delivery` (`autovalue.loaders.from_amazon_delivery`) |

```bash
autovalue train --task eta --csv data/amazon_delivery.csv --adapter amazon-delivery
```

## 3. Project schema

A CSV that already uses the project schema needs no adapter.

| Table | Required columns | Optional columns |
|---|---|---|
| listings | `listing_id, listed_at, brand, model, year, odometer_km, fuel, transmission, price` | `owners, accidents, engine_l, body_type, city` |
| deliveries | `order_id, store_lat, store_lon, drop_lat, drop_lon, ordered_at, picked_at, traffic, delivery_minutes` | `weather, area, vehicle` |
