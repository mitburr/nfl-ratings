"""Visualization tools for NFL Elo ratings."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.models.elo import EloRatingSystem
from src.database.db_manager import DatabaseManager

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_elo_over_time(elo_system, teams=None, title="Elo Ratings Over Season"):
    """
    Plot Elo rating evolution over the season.
    
    Args:
        elo_system: EloRatingSystem with calculated ratings
        teams: List of team codes to plot (None = all teams)
        title: Plot title
    """
    # Extract rating history
    history_data = []
    for game in elo_system.history:
        history_data.append({
            'week': game['week'],
            'team': game['home_team'],
            'rating': game['home_rating_after']
        })
        history_data.append({
            'week': game['week'],
            'team': game['away_team'],
            'rating': game['away_rating_after']
        })
    
    df = pd.DataFrame(history_data)
    
    # Filter to specific teams if requested
    if teams:
        df = df[df['team'].isin(teams)]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for team in df['team'].unique():
        team_data = df[df['team'] == team].sort_values('week')
        ax.plot(team_data['week'], team_data['rating'], 
                marker='o', label=team, linewidth=2, markersize=4)
    
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Elo Rating', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axhline(y=1500, color='gray', linestyle='--', alpha=0.5, label='Average (1500)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_k_factors(season, k_factors, teams=None):
    """
    Compare rankings across different k-factors.
    
    Args:
        season: Season to analyze
        k_factors: List of k-factors to compare
        teams: Optional list of teams to highlight
    """
    results = {}
    
    for k in k_factors:
        elo = EloRatingSystem(k_factor=k)
        elo.calculate_season(season, save_to_db=False)
        rankings = elo.get_rankings()
        results[f'K={k}'] = rankings.set_index('team')['rank']
    
    df = pd.DataFrame(results)
    
    # Filter if requested
    if teams:
        df = df.loc[teams]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create heatmap
    sns.heatmap(df, annot=True, fmt='g', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Rank'}, ax=ax,
                vmin=1, vmax=32, linewidths=0.5)
    
    ax.set_title('Team Rankings Across Different K-Factors', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('K-Factor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Team', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_ranking_comparison(season, k1=20, k2=30):
    """
    Side-by-side comparison of two k-factors.
    
    Shows which teams move up/down between different k-factors.
    """
    # Calculate both
    elo1 = EloRatingSystem(k_factor=k1)
    elo1.calculate_season(season, save_to_db=False)
    rank1 = elo1.get_rankings()
    
    elo2 = EloRatingSystem(k_factor=k2)
    elo2.calculate_season(season, save_to_db=False)
    rank2 = elo2.get_rankings()
    
    # Merge
    comparison = rank1.merge(rank2, on='team', suffixes=(f'_k{k1}', f'_k{k2}'))
    comparison['rank_change'] = comparison[f'rank_k{k1}'] - comparison[f'rank_k{k2}']
    comparison = comparison.sort_values(f'rank_k{k1}')
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Left plot: Rankings comparison
    y_pos = np.arange(len(comparison))
    
    ax1.barh(y_pos, comparison[f'rating_k{k1}'], alpha=0.6, label=f'K={k1}')
    ax1.barh(y_pos, comparison[f'rating_k{k2}'], alpha=0.6, label=f'K={k2}')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(comparison['team'])
    ax1.set_xlabel('Elo Rating', fontsize=12, fontweight='bold')
    ax1.set_title(f'Rating Comparison: K={k1} vs K={k2}', 
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.invert_yaxis()
    ax1.axvline(x=1500, color='gray', linestyle='--', alpha=0.5)
    
    # Right plot: Biggest movers
    movers = pd.concat([
        comparison.nlargest(10, 'rank_change', keep='all'),
        comparison.nsmallest(10, 'rank_change', keep='all')
    ])
    movers = movers.sort_values('rank_change')
    
    colors = ['green' if x > 0 else 'red' for x in movers['rank_change']]
    ax2.barh(movers['team'], movers['rank_change'], color=colors, alpha=0.7)
    ax2.set_xlabel('Rank Change (+ = Better with lower K)', fontsize=12, fontweight='bold')
    ax2.set_title('Biggest Rank Changes', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    return fig


def plot_prediction_accuracy(elo_system):
    """
    Scatter plot: Predicted win probability vs actual outcome.
    
    Shows calibration - are 70% predictions actually winning 70% of the time?
    """
    predictions = []
    actuals = []
    
    for game in elo_system.history:
        pred_prob = game['expected_home_win_prob']
        actual = 1 if game['home_score'] > game['away_score'] else 0
        
        predictions.append(pred_prob)
        actuals.append(actual)
    
    # Create bins for calibration curve
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_actuals = []
    
    for i in range(len(bins) - 1):
        mask = (np.array(predictions) >= bins[i]) & (np.array(predictions) < bins[i+1])
        if mask.sum() > 0:
            bin_actuals.append(np.array(actuals)[mask].mean())
        else:
            bin_actuals.append(np.nan)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Scatter plot
    ax1.scatter(predictions, actuals, alpha=0.3, s=50)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    ax1.set_xlabel('Predicted Win Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actual Outcome (1=Win, 0=Loss)', fontsize=12, fontweight='bold')
    ax1.set_title('Prediction vs Reality', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Calibration curve
    ax2.plot(bin_centers, bin_actuals, 'o-', linewidth=2, markersize=8, label='Actual')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    ax2.set_xlabel('Predicted Probability Bin', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actual Win Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Calibration Curve', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig


def plot_strength_of_schedule(elo_system, top_n=32):
    """
    Bar chart showing strength of schedule.
    """
    # Calculate SOS
    sos = {}
    for game in elo_system.history:
        home_team = game['home_team']
        away_team = game['away_team']
        
        if home_team not in sos:
            sos[home_team] = []
        sos[home_team].append(game['away_rating_before'])
        
        if away_team not in sos:
            sos[away_team] = []
        sos[away_team].append(game['home_rating_before'])
    
    sos_df = pd.DataFrame([
        {'team': team, 'avg_opponent_rating': np.mean(ratings)}
        for team, ratings in sos.items()
    ]).sort_values('avg_opponent_rating', ascending=False).head(top_n)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['darkred' if x > 1500 else 'darkgreen' for x in sos_df['avg_opponent_rating']]
    ax.barh(sos_df['team'], sos_df['avg_opponent_rating'], color=colors, alpha=0.7)
    ax.axvline(x=1500, color='black', linestyle='--', linewidth=2, label='League Average')
    ax.set_xlabel('Average Opponent Elo Rating', fontsize=12, fontweight='bold')
    ax.set_ylabel('Team', fontsize=12, fontweight='bold')
    ax.set_title('Strength of Schedule', fontsize=14, fontweight='bold')
    ax.legend()
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


def plot_upset_analysis(elo_system, top_n=15):
    """
    Visualize the biggest upsets of the season.
    """
    upsets = elo_system.get_biggest_upsets(top_n)
    
    if len(upsets) == 0:
        print("No upsets found")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create labels
    upsets['label'] = (upsets['winner'] + ' over ' + upsets['loser'] + 
                       ' (Week ' + upsets['week'].astype(str) + ')')
    
    colors = plt.cm.RdYlGn_r(upsets['upset_factor'] / upsets['upset_factor'].max())
    
    ax.barh(upsets['label'], upsets['upset_factor'], color=colors, alpha=0.8)
    ax.set_xlabel('Upset Factor (Higher = Bigger Upset)', fontsize=12, fontweight='bold')
    ax.set_title('Biggest Upsets of the Season', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


def create_comparison_dashboard(season, k_factors=[20, 30]):
    """
    Create a multi-panel dashboard comparing two k-factors.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    elos = {}
    for k in k_factors:
        elo = EloRatingSystem(k_factor=k)
        elo.calculate_season(season, save_to_db=False)
        elos[k] = elo
    
    # Top left: Elo evolution for top 8 teams (K1)
    ax1 = fig.add_subplot(gs[0, 0])
    top_teams = elos[k_factors[0]].get_rankings().head(8)['team'].tolist()
    plot_elo_evolution_on_axis(ax1, elos[k_factors[0]], top_teams, 
                                f'Top 8 Teams Evolution (K={k_factors[0]})')
    
    # Top right: Elo evolution for top 8 teams (K2)
    ax2 = fig.add_subplot(gs[0, 1])
    top_teams = elos[k_factors[1]].get_rankings().head(8)['team'].tolist()
    plot_elo_evolution_on_axis(ax2, elos[k_factors[1]], top_teams, 
                                f'Top 8 Teams Evolution (K={k_factors[1]})')
    
    # Middle: Ranking comparison
    ax3 = fig.add_subplot(gs[1, :])
    plot_ranking_bars_on_axis(ax3, elos[k_factors[0]], elos[k_factors[1]], 
                               k_factors[0], k_factors[1])
    
    # Bottom left: SOS comparison
    ax4 = fig.add_subplot(gs[2, 0])
    plot_sos_comparison_on_axis(ax4, elos[k_factors[0]], elos[k_factors[1]], 
                                 k_factors[0], k_factors[1])
    
    # Bottom right: Accuracy comparison
    ax5 = fig.add_subplot(gs[2, 1])
    plot_accuracy_comparison_on_axis(ax5, elos[k_factors[0]], elos[k_factors[1]], 
                                     k_factors[0], k_factors[1])
    
    fig.suptitle(f'Elo Analysis Comparison: K={k_factors[0]} vs K={k_factors[1]}', 
                 fontsize=16, fontweight='bold')
    
    return fig


# Helper functions for dashboard
def plot_elo_evolution_on_axis(ax, elo_system, teams, title):
    """Plot Elo evolution on given axis."""
    history_data = []
    for game in elo_system.history:
        history_data.append({
            'week': game['week'],
            'team': game['home_team'],
            'rating': game['home_rating_after']
        })
        history_data.append({
            'week': game['week'],
            'team': game['away_team'],
            'rating': game['away_rating_after']
        })
    
    df = pd.DataFrame(history_data)
    df = df[df['team'].isin(teams)]
    
    for team in df['team'].unique():
        team_data = df[df['team'] == team].sort_values('week')
        ax.plot(team_data['week'], team_data['rating'], marker='o', label=team, linewidth=2)
    
    ax.axhline(y=1500, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Week')
    ax.set_ylabel('Elo Rating')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_ranking_bars_on_axis(ax, elo1, elo2, k1, k2):
    """Plot ranking comparison bars."""
    rank1 = elo1.get_rankings()
    rank2 = elo2.get_rankings()
    
    comparison = rank1.merge(rank2, on='team', suffixes=(f'_{k1}', f'_{k2}'))
    comparison['diff'] = comparison[f'rank_{k2}'] - comparison[f'rank_{k1}']
    comparison = comparison.sort_values(f'rank_{k1}').head(16)
    
    x = np.arange(len(comparison))
    width = 0.35
    
    ax.bar(x - width/2, comparison[f'rank_{k1}'], width, label=f'K={k1}', alpha=0.8)
    ax.bar(x + width/2, comparison[f'rank_{k2}'], width, label=f'K={k2}', alpha=0.8)
    
    ax.set_ylabel('Rank')
    ax.set_title('Top 16 Rankings Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison['team'], rotation=45, ha='right')
    ax.legend()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='y')


def plot_sos_comparison_on_axis(ax, elo1, elo2, k1, k2):
    """Plot SOS comparison."""
    # Get SOS for both
    def get_sos(elo):
        sos = {}
        for game in elo.history:
            for team, opp_rating in [(game['home_team'], game['away_rating_before']),
                                      (game['away_team'], game['home_rating_before'])]:
                if team not in sos:
                    sos[team] = []
                sos[team].append(opp_rating)
        return {team: np.mean(ratings) for team, ratings in sos.items()}
    
    sos1 = get_sos(elo1)
    sos2 = get_sos(elo2)
    
    teams = sorted(sos1.keys(), key=lambda t: sos1[t], reverse=True)[:10]
    
    x = np.arange(len(teams))
    width = 0.35
    
    ax.barh([i - width/2 for i in x], [sos1[t] for t in teams], width, 
            label=f'K={k1}', alpha=0.8)
    ax.barh([i + width/2 for i in x], [sos2[t] for t in teams], width, 
            label=f'K={k2}', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(teams)
    ax.set_xlabel('Avg Opponent Rating')
    ax.set_title('Toughest Schedules', fontweight='bold')
    ax.axvline(x=1500, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.invert_yaxis()


def plot_accuracy_comparison_on_axis(ax, elo1, elo2, k1, k2):
    """Plot accuracy stats comparison."""
    def get_accuracy(elo):
        correct = sum(1 for g in elo.history 
                     if (g['expected_home_win_prob'] > 0.5 and g['home_score'] > g['away_score']) or
                        (g['expected_home_win_prob'] < 0.5 and g['home_score'] < g['away_score']))
        return correct / len(elo.history)
    
    acc1 = get_accuracy(elo1)
    acc2 = get_accuracy(elo2)
    
    ax.bar([f'K={k1}', f'K={k2}'], [acc1, acc2], color=['blue', 'orange'], alpha=0.7)
    ax.set_ylabel('Accuracy')
    ax.set_title('Prediction Accuracy', fontweight='bold')
    ax.set_ylim(0.5, 0.7)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Guess')
    
    # Add percentage labels
    for i, (label, val) in enumerate([(f'K={k1}', acc1), (f'K={k2}', acc2)]):
        ax.text(i, val + 0.01, f'{val:.1%}', ha='center', fontweight='bold')
    
    ax.legend()
