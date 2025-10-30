#!/usr/bin/env python
"""Setup script to initialize the NFL ratings project."""
import os
import sys
import subprocess


def check_postgres():
    """Check if PostgreSQL is accessible."""
    try:
        import psycopg2
        print("✓ psycopg2 installed")
        return True
    except ImportError:
        print("✗ psycopg2 not installed. Run: pip install -r requirements.txt")
        return False


def check_env_file():
    """Check if .env file exists."""
    if os.path.exists('.env'):
        print("✓ .env file exists")
        return True
    else:
        print("✗ .env file not found. Copy .env.example to .env and configure it.")
        print("  cp .env.example .env")
        return False


def init_database():
    """Initialize the database schema."""
    from src.database.db_manager import DatabaseManager
    
    print("\nInitializing database...")
    db = DatabaseManager()
    
    # Test connection
    if not db.test_connection():
        print("✗ Database connection failed. Check your .env configuration.")
        return False
    
    # Create schema
    try:
        db.execute_sql_file('src/database/schema.sql')
        print("✓ Database schema created")
        return True
    except Exception as e:
        print(f"✗ Failed to create schema: {e}")
        return False


def load_initial_data():
    """Load initial data."""
    from src.ingestion.nfl_data import NFLDataIngestion
    
    print("\nLoading initial data...")
    ingestion = NFLDataIngestion()
    
    try:
        ingestion.load_current_season()
        print("✓ Initial data loaded")
        return True
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return False


def main():
    """Run setup process."""
    print("NFL Ratings Project Setup")
    print("=" * 50)
    
    # Check requirements
    print("\nChecking requirements...")
    if not check_postgres():
        sys.exit(1)
    
    if not check_env_file():
        sys.exit(1)
    
    # Initialize database
    if not init_database():
        sys.exit(1)
    
    # Load data
    response = input("\nLoad current season data? (y/n): ")
    if response.lower() == 'y':
        if not load_initial_data():
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("Setup complete! 🏈")
    print("\nNext steps:")
    print("  1. Run analysis: python -m src.analysis.rankings")
    print("  2. Start Jupyter: jupyter notebook")
    print("  3. Explore the data!")


if __name__ == "__main__":
    main()
