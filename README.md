# NFL Ratings System

A statistical analysis project for evaluating NFL team quality using various rating systems and advanced metrics.

## Project Goals

- Build rating systems (Elo, EPA-based) that account for strength of schedule
- Handle NFL's small sample size problem (17 games/season)
- Predict game outcomes and evaluate team quality
- Analyze historical trends and team performance

## Setup

### Prerequisites
- Python 3.9+
- PostgreSQL 14+

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd nfl-ratings

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up database
createdb nfl_stats
psql nfl_stats -f src/database/schema.sql

# Configure environment
cp .env.example .env
# Edit .env with your database credentials
```

## Project Structure

```
├── config/          # Configuration files
├── data/            # Data storage (gitignored)
├── src/             # Source code
│   ├── ingestion/   # Data fetching and loading
│   ├── models/      # Rating systems and models
│   ├── database/    # Database management
│   └── analysis/    # Analysis scripts
├── notebooks/       # Jupyter notebooks for exploration
├── tests/           # Unit tests
└── scripts/         # Utility scripts
```

## Usage

```bash
# Fetch current season data
python -m src.ingestion.nfl_data

# Calculate Elo ratings
python -m src.models.elo

# Run analysis
jupyter notebook notebooks/
```

## Tech Stack

- **Python 3.9+**: Core language
- **PostgreSQL**: Data storage
- **pandas/numpy**: Data manipulation
- **nfl_data_py**: NFL data source
- **psycopg2**: PostgreSQL adapter
- **jupyter**: Interactive analysis

## License

MIT
