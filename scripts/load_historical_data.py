"""Load historical NFL season data."""
import argparse
from src.ingestion.nfl_data import NFLDataIngestion


def load_seasons(start_year, end_year=None):
    """
    Load multiple seasons of data.
    
    Args:
        start_year: First year to load
        end_year: Last year to load (inclusive). If None, just loads start_year
    """
    if end_year is None:
        years = [start_year]
    else:
        years = list(range(start_year, end_year + 1))
    
    print(f"Loading data for seasons: {years}")
    print("="*60)
    
    ingestion = NFLDataIngestion()
    
    print("\nLoading schedules...")
    ingestion.fetch_schedule(years)
    
    print("\n" + "="*60)
    print(f"✓ Successfully loaded {len(years)} season(s)")
    print("\nYou can now analyze these seasons with:")
    for year in years:
        print(f"  python -m scripts.run_full_analysis --season {year}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load historical NFL data')
    parser.add_argument('start', type=int, help='Start year (e.g., 2022)')
    parser.add_argument('end', type=int, nargs='?', help='End year (optional, e.g., 2024)')
    
    args = parser.parse_args()
    
    load_seasons(args.start, args.end)
