# Willingness-to-Pay Segmentation | Hotel RM

A **Streamlit** dashboard for hotel revenue management that segments bookings by willingness-to-pay (WTP), supports dynamic K-Means clustering, and explores price sensitivity. Built for the Ubud property with IDR pricing.

## Features

- **Overview** — KPIs, monthly bookings vs average rate, WTP tier distribution, channel mix, revenue by room type
- **WTP Distribution** — Score and rate histograms, box plots by segment, configurable WTP heatmaps
- **Segment Profiles** — Tier profile table, radar charts, channel mix by tier, deep dive per WTP tier
- **Dynamic Clustering** — K-Means on configurable features, elbow plot, PCA projection, cluster profiles and channel mix
- **Price Sensitivity** — Demand curves by segment, cancellation rate vs price, booking window vs rate, member vs non-member WTP

Filters in the sidebar: check-in date range, room type, booking channel, rate plan, cancellation status, and membership.

## Setup

### Requirements

- Python 3.9+
- Dependencies in `requirements.txt`

### Install

```bash
cd "d:\Python\Willingness-to-Pay"
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Data

Place your booking data at **`data/Bookings.csv`**. Expected columns include:

- `Booking_ID`, `Booking_Date`, `Check_in_Date`, `Check_out_Date`
- `Room_Type`, `Booked_Rate`, `Rate_Plan`, `Booking_Channel`
- `Cancellation_Status`, `Revenue_Generated`
- `Number_of_Nights`, `Number_of_Guests`

The app derives WTP score (0–100) from booked rate, rate-plan commitment, channel, and booking-window urgency, and assigns quartile tiers: Price-Sensitive, Value-Seeker, Comfort-Buyer, Premium.

### Run

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Tech Stack

- **Streamlit** — UI and caching
- **Pandas** — Data loading and feature engineering
- **Plotly** — Interactive charts
- **scikit-learn** — K-Means, PCA, StandardScaler

## License

Use and modify as needed for your property or analysis.
