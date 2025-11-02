"""Deep dive analysis of team vectors and their implementation."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.database.db_manager import DatabaseManager
from src.analysis.team_vectors import calculate_team_vectors, analyze_matchup


def analyze_vector_calculation(season=2024):
    """Show how vectors are calculated with sample data."""
    db = DatabaseManager()
    
    print("="*70)
    print("VECTOR CALCULATION BREAKDOWN")
    print("="*70)
    
    # Get sample plays for one team
    query = """
        SELECT 
            p.play_type,
            p.yards_gained,
            p.success,
            p.epa
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.season = %s AND p.posteam = 'BAL'
        LIMIT 10
    """
    sample_plays = db.query_to_dataframe(query, params=(season,))
    
    print("\nSample BAL plays:")
    print(sample_plays.to_string(index=False))
    
    # Show aggregation
    print("\n" + "="*70)
    print("AGGREGATION PROCESS")
    print("="*70)
    
    query = """
        SELECT 
            p.posteam as team,
            p.play_type,
            COUNT(*) as plays,
            AVG(p.yards_gained) as avg_yards,
            AVG(CASE WHEN p.success THEN 1.0 ELSE 0.0 END) as success_rate
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.season = %s AND p.posteam = 'BAL'
        GROUP BY p.posteam, p.play_type
        ORDER BY plays DESC
        LIMIT 5
    """
    agg = db.query_to_dataframe(query, params=(season,))
    print("\nBAL aggregated stats:")
    print(agg.to_string(index=False))
    
    return sample_plays


def analyze_top_teams(season=2024):
    """Identify top teams for each vector metric."""
    vectors = calculate_team_vectors(season)
    
    print("\n" + "="*70)
    print("TOP 5 TEAMS PER VECTOR METRIC")
    print("="*70)
    
    metrics = {
        'Rush Offense': ('rush_ypa', False),
        'Pass Offense': ('pass_ypa', False),
        'Rush Defense': ('rush_ypa_allowed', True),  # Lower is better
        'Pass Defense': ('pass_ypa_allowed', True),
        'Success Rate': ('success_rate', False),
        'Explosive Plays': ('explosive_rate', False)
    }
    
    results = {}
    for name, (col, ascending) in metrics.items():
        print(f"\n{name}:")
        top = vectors.nsmallest(5, col) if ascending else vectors.nlargest(5, col)
        print(top[['team', col]].to_string(index=False))
        results[name] = top
    
    return results, vectors


def analyze_play_by_play_detail(season=2024):
    """Explore detailed play-by-play features for potential new vectors."""
    db = DatabaseManager()
    
    print("\n" + "="*70)
    print("PLAY-BY-PLAY DATA EXPLORATION")
    print("="*70)
    
    # Check available columns
    query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'plays'
        ORDER BY ordinal_position
    """
    columns = db.query_to_dataframe(query)
    print("\nAvailable columns:")
    print(columns.to_string(index=False))
    
    # Sample detailed stats
    query = """
        SELECT 
            p.posteam,
            COUNT(*) as total_plays,
            COUNT(CASE WHEN p.play_type = 'run' THEN 1 END) as runs,
            COUNT(CASE WHEN p.play_type = 'pass' THEN 1 END) as passes,
            AVG(p.yards_gained) as avg_yards,
            AVG(p.epa) as avg_epa,
            SUM(CASE WHEN p.yards_gained >= 10 THEN 1 ELSE 0 END)::float / COUNT(*)::float as first_down_rate
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.season = %s AND p.posteam IS NOT NULL
        GROUP BY p.posteam
        ORDER BY avg_epa DESC
        LIMIT 10
    """
    detailed = db.query_to_dataframe(query, params=(season,))
    print("\nTop 10 teams by EPA:")
    print(detailed.to_string(index=False))
    
    return detailed


def analyze_vector_contribution():
    """Show how vectors modify Elo predictions."""
    print("\n" + "="*70)
    print("HOW VECTORS MODIFY PREDICTIONS")
    print("="*70)
    
    print("\nCurrent formula:")
    print("  1. Calculate matchup advantage (yards/play difference)")
    print("  2. vector_boost = advantage * BOOST (e.g., 0.60)")
    print("  3. adjustment = vector_boost * WEIGHT (e.g., 0.35)")
    print("  4. final_prob = elo_prob + adjustment")
    
    # Example calculation
    print("\n" + "-"*70)
    print("EXAMPLE: BAL @ MIA")
    print("-"*70)
    
    vectors = calculate_team_vectors(2024)
    matchup = analyze_matchup('MIA', 'BAL', vectors)
    
    if matchup:
        print(f"\nMatchup analysis:")
        print(f"  MIA run advantage: {matchup['home_run_advantage']:+.3f} yards")
        print(f"  MIA pass advantage: {matchup['home_pass_advantage']:+.3f} yards")
        print(f"  BAL run advantage: {matchup['away_run_advantage']:+.3f} yards")
        print(f"  BAL pass advantage: {matchup['away_pass_advantage']:+.3f} yards")
        print(f"  Total advantage: {matchup['total_advantage']:+.3f} yards (positive = MIA)")
        
        boost = 0.60
        weight = 0.35
        vector_boost = matchup['total_advantage'] * boost
        adjustment = vector_boost * weight
        
        print(f"\nWith BOOST={boost}, WEIGHT={weight}:")
        print(f"  Vector boost: {vector_boost:+.3f} (= {matchup['total_advantage']:+.3f} * {boost})")
        print(f"  Final adjustment: {adjustment:+.3f} (= {vector_boost:+.3f} * {weight})")
        print(f"  Effect: {adjustment*100:+.1f}% win probability shift for home team")


def visualize_vectors(season=2024):
    """Create comprehensive vector visualizations."""
    vectors = calculate_team_vectors(season)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{season} Team Vector Analysis', fontsize=16, fontweight='bold')
    
    # 1. Rush Offense vs Defense
    ax = axes[0, 0]
    scatter = ax.scatter(vectors['rush_ypa'], vectors['rush_ypa_allowed'], 
                        s=100, alpha=0.6, c=vectors['rush_ypa'], cmap='RdYlGn')
    for _, row in vectors.iterrows():
        ax.annotate(row['team'], (row['rush_ypa'], row['rush_ypa_allowed']), 
                   fontsize=8, alpha=0.7)
    ax.axhline(vectors['rush_ypa_allowed'].mean(), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(vectors['rush_ypa'].mean(), color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Rush Yards/Attempt (Offense)')
    ax.set_ylabel('Rush Yards/Attempt Allowed (Defense)')
    ax.set_title('Rush Offense vs Defense')
    ax.grid(True, alpha=0.3)
    
    # 2. Pass Offense vs Defense
    ax = axes[0, 1]
    scatter = ax.scatter(vectors['pass_ypa'], vectors['pass_ypa_allowed'],
                        s=100, alpha=0.6, c=vectors['pass_ypa'], cmap='RdYlGn')
    for _, row in vectors.iterrows():
        ax.annotate(row['team'], (row['pass_ypa'], row['pass_ypa_allowed']),
                   fontsize=8, alpha=0.7)
    ax.axhline(vectors['pass_ypa_allowed'].mean(), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(vectors['pass_ypa'].mean(), color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Pass Yards/Attempt (Offense)')
    ax.set_ylabel('Pass Yards/Attempt Allowed (Defense)')
    ax.set_title('Pass Offense vs Defense')
    ax.grid(True, alpha=0.3)
    
    # 3. Success Rate Distribution
    ax = axes[0, 2]
    vectors_sorted = vectors.sort_values('success_rate', ascending=False)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(vectors_sorted)))
    ax.barh(range(len(vectors_sorted)), vectors_sorted['success_rate'], color=colors)
    ax.set_yticks(range(len(vectors_sorted)))
    ax.set_yticklabels(vectors_sorted['team'], fontsize=8)
    ax.set_xlabel('Success Rate')
    ax.set_title('Offensive Success Rate by Team')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Explosive Play Rate
    ax = axes[1, 0]
    vectors_sorted = vectors.sort_values('explosive_rate', ascending=False)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(vectors_sorted)))
    ax.barh(range(len(vectors_sorted)), vectors_sorted['explosive_rate'], color=colors)
    ax.set_yticks(range(len(vectors_sorted)))
    ax.set_yticklabels(vectors_sorted['team'], fontsize=8)
    ax.set_xlabel('Explosive Play Rate (20+ yards)')
    ax.set_title('Explosive Plays by Team')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 5. Total Offense (Rush + Pass)
    ax = axes[1, 1]
    vectors['total_ypa'] = (vectors['rush_ypa'] + vectors['pass_ypa']) / 2
    vectors_sorted = vectors.sort_values('total_ypa', ascending=False).head(15)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(vectors_sorted)))
    ax.bar(range(len(vectors_sorted)), vectors_sorted['total_ypa'], color=colors)
    ax.set_xticks(range(len(vectors_sorted)))
    ax.set_xticklabels(vectors_sorted['team'], rotation=45, fontsize=9)
    ax.set_ylabel('Average Yards/Attempt')
    ax.set_title('Top 15 Total Offenses')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Correlation heatmap
    ax = axes[1, 2]
    corr_data = vectors[['rush_ypa', 'pass_ypa', 'rush_ypa_allowed', 
                         'pass_ypa_allowed', 'success_rate', 'explosive_rate']].corr()
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Vector Metric Correlations')
    
    plt.tight_layout()
    output_path = 'data/plots/vector_deep_dive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")
    plt.show()


def suggest_new_vectors():
    """Suggest additional vector features to capture more nuance."""
    print("\n" + "="*70)
    print("POTENTIAL NEW VECTOR FEATURES")
    print("="*70)
    
    suggestions = {
        'Situational': [
            '3rd down conversion rate',
            'Red zone scoring rate',
            'Goal-to-go success',
            'Short yardage (3rd/4th & 1-2) conversion'
        ],
        'Field Position': [
            'Average starting field position',
            'Yards per drive',
            'Points per drive',
            'Drive success rate'
        ],
        'Game Script': [
            'Performance when leading',
            'Performance when trailing',
            'First quarter scoring',
            'Fourth quarter scoring'
        ],
        'Personnel': [
            '11 personnel (1 RB, 1 TE) efficiency',
            '12 personnel (1 RB, 2 TE) efficiency',
            'Play action success rate',
            'RPO efficiency'
        ],
        'Play Characteristics': [
            'Deep pass rate (15+ air yards)',
            'Deep pass completion %',
            'Outside run vs inside run split',
            'Pre-snap motion usage'
        ]
    }
    
    for category, features in suggestions.items():
        print(f"\n{category}:")
        for feature in features:
            print(f"  • {feature}")


if __name__ == "__main__":
    season = 2024
    
    # 1. Show calculation process
    analyze_vector_calculation(season)
    
    # 2. Top teams analysis
    top_teams, vectors = analyze_top_teams(season)
    
    # 3. Explore play-by-play detail
    detailed_stats = analyze_play_by_play_detail(season)
    
    # 4. Show prediction contribution
    analyze_vector_contribution()
    
    # 5. Visualize
    visualize_vectors(season)
    
    # 6. Suggest improvements
    suggest_new_vectors()
