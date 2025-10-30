# Setup Guide

## Quick Start

### 1. Configure Git (if needed)
```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 2. Create GitHub Repository
1. Go to https://github.com/new
2. Create a new repository (e.g., `nfl-ratings`)
3. Don't initialize with README (we already have one)

### 3. Push to GitHub
```bash
cd /path/to/nfl-ratings

# Set your GitHub repo as remote
git remote add origin https://github.com/YOUR_USERNAME/nfl-ratings.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 4. Set Up PostgreSQL

**Option A: Local PostgreSQL**
```bash
# Install PostgreSQL (if not installed)
# macOS: brew install postgresql
# Ubuntu: sudo apt-get install postgresql postgresql-contrib

# Start PostgreSQL
# macOS: brew services start postgresql
# Ubuntu: sudo service postgresql start

# Create database
createdb nfl_stats

# Or using psql
psql postgres
CREATE DATABASE nfl_stats;
\q
```

**Option B: Docker PostgreSQL**
```bash
docker run --name nfl-postgres \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=nfl_stats \
  -p 5432:5432 \
  -d postgres:14
```

### 5. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your database credentials
nano .env  # or vim, code, etc.
```

Example `.env`:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nfl_stats
DB_USER=postgres
DB_PASSWORD=yourpassword
```

### 6. Install Python Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 7. Initialize Database
```bash
# Run setup script
python scripts/setup.py

# Or manually:
python -m src.database.db_manager  # Test connection
psql nfl_stats -f src/database/schema.sql  # Create tables
python -m src.ingestion.nfl_data  # Load data
```

## Project Structure

```
nfl-ratings/
├── config/              # Configuration files
│   └── database.py     # DB connection config
├── data/
│   ├── raw/            # Downloaded data (gitignored)
│   └── processed/      # Transformed data (gitignored)
├── src/
│   ├── ingestion/      # Data fetching scripts
│   │   └── nfl_data.py
│   ├── models/         # Rating systems (Elo, EPA, etc.)
│   ├── database/       # DB management
│   │   ├── db_manager.py
│   │   └── schema.sql
│   └── analysis/       # Analysis scripts
├── notebooks/          # Jupyter notebooks
├── tests/             # Unit tests
└── scripts/           # Utility scripts
    └── setup.py       # Initial setup
```

## Next Steps

1. **Verify setup works:**
   ```bash
   python -m src.database.db_manager  # Should print PostgreSQL version
   ```

2. **Load data:**
   ```bash
   python -m src.ingestion.nfl_data
   ```

3. **Build your first model:**
   - Create `src/models/elo.py` with Elo rating system
   - Test on current season data

4. **Explore data:**
   ```bash
   jupyter notebook
   ```

## Common Issues

**"Could not connect to database"**
- Check PostgreSQL is running
- Verify credentials in `.env`
- Test with: `psql -h localhost -U postgres -d nfl_stats`

**"Module not found"**
- Activate virtual environment: `source venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`

**"Table does not exist"**
- Run schema: `psql nfl_stats -f src/database/schema.sql`
- Or: `python scripts/setup.py`
