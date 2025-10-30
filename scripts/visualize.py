"""Generate visualizations for NFL Elo analysis."""
import os
import argparse
from src.models.elo import EloRatingSystem
from src.analysis import visualizations as viz
import matplotlib.pyplot as plt


def ensure_output_dir():
    """Create output directory for plots."""
    os.makedirs('data/plots', exist_ok=True)


def generate_all_plots(season=2024, k_factor=20, home_advantage=65, save=True, show=False):
    """
    Generate all standard visualizations.
    
    Args:
        season: Season to analyze
        k_factor: K-factor for Elo
        home_advantage: Home advantage in Elo points
        save: Whether to save plots
        show: Whether to display plots interactively
    """
    ensure_output_dir()
    
    print(f"Generating visualizations for {season} season (K={k_factor}, Home={home_advantage})...")
    
    # Calculate Elo
    print("\n[1/7] Calculating Elo ratings...")
    elo = EloRatingSystem(k_factor=k_factor, home_advantage=home_advantage)
    elo.calculate_season(season, save_to_db=False)
    
    # 1. Elo evolution - top 8 teams
    print("[2/7] Plotting Elo evolution (top 8)...")
    top_teams = elo.get_rankings().head(8)['team'].tolist()
    fig1 = viz.plot_elo_over_time(elo, teams=top_teams, 
                                   title=f'Top 8 Teams - Elo Evolution (K={k_factor})')
    if save:
        fig1.savefig(f'data/plots/elo_evolution_top8_k{k_factor}.png', dpi=300, bbox_inches='tight')
        print(f"  → Saved to data/plots/elo_evolution_top8_k{k_factor}.png")
    if show:
        plt.show()
    plt.close()
    
    # 2. All teams evolution
    print("[3/7] Plotting Elo evolution (all teams)...")
    fig2 = viz.plot_elo_over_time(elo, teams=None, 
                                   title=f'All Teams - Elo Evolution (K={k_factor})')
    if save:
        fig2.savefig(f'data/plots/elo_evolution_all_k{k_factor}.png', dpi=300, bbox_inches='tight')
        print(f"  → Saved to data/plots/elo_evolution_all_k{k_factor}.png")
    if show:
        plt.show()
    plt.close()
    
    # 3. Prediction accuracy
    print("[4/7] Plotting prediction accuracy...")
    fig3 = viz.plot_prediction_accuracy(elo)
    if save:
        fig3.savefig(f'data/plots/prediction_accuracy_k{k_factor}.png', dpi=300, bbox_inches='tight')
        print(f"  → Saved to data/plots/prediction_accuracy_k{k_factor}.png")
    if show:
        plt.show()
    plt.close()
    
    # 4. Strength of schedule
    print("[5/7] Plotting strength of schedule...")
    fig4 = viz.plot_strength_of_schedule(elo)
    if save:
        fig4.savefig(f'data/plots/strength_of_schedule_k{k_factor}.png', dpi=300, bbox_inches='tight')
        print(f"  → Saved to data/plots/strength_of_schedule_k{k_factor}.png")
    if show:
        plt.show()
    plt.close()
    
    # 5. Biggest upsets
    print("[6/7] Plotting biggest upsets...")
    fig5 = viz.plot_upset_analysis(elo, top_n=15)
    if fig5:
        if save:
            fig5.savefig(f'data/plots/biggest_upsets_k{k_factor}.png', dpi=300, bbox_inches='tight')
            print(f"  → Saved to data/plots/biggest_upsets_k{k_factor}.png")
        if show:
            plt.show()
        plt.close()
    
    print("[7/7] Done!")
    print(f"\nGenerated 5 visualizations in data/plots/")


def compare_k_factors_visual(season=2024, k1=20, k2=30, save=True, show=False):
    """
    Generate comparison visualizations for two k-factors.
    
    Args:
        season: Season to analyze
        k1: First k-factor
        k2: Second k-factor
        save: Whether to save plots
        show: Whether to display plots
    """
    ensure_output_dir()
    
    print(f"Comparing K={k1} vs K={k2} for {season} season...")
    
    # 1. Ranking comparison
    print("\n[1/3] Creating ranking comparison...")
    fig1 = viz.plot_ranking_comparison(season, k1, k2)
    if save:
        fig1.savefig(f'data/plots/ranking_comparison_k{k1}_vs_k{k2}.png', 
                     dpi=300, bbox_inches='tight')
        print(f"  → Saved to data/plots/ranking_comparison_k{k1}_vs_k{k2}.png")
    if show:
        plt.show()
    plt.close()
    
    # 2. K-factor heatmap
    print("[2/3] Creating k-factor heatmap...")
    top_teams = EloRatingSystem(k_factor=k1)
    top_teams.calculate_season(season, save_to_db=False)
    teams = top_teams.get_rankings().head(16)['team'].tolist()
    
    fig2 = viz.compare_k_factors(season, [k1, k2], teams=teams)
    if save:
        fig2.savefig(f'data/plots/kfactor_heatmap_k{k1}_vs_k{k2}.png', 
                     dpi=300, bbox_inches='tight')
        print(f"  → Saved to data/plots/kfactor_heatmap_k{k1}_vs_k{k2}.png")
    if show:
        plt.show()
    plt.close()
    
    # 3. Full dashboard
    print("[3/3] Creating comparison dashboard...")
    fig3 = viz.create_comparison_dashboard(season, k_factors=[k1, k2])
    if save:
        fig3.savefig(f'data/plots/dashboard_k{k1}_vs_k{k2}.png', 
                     dpi=300, bbox_inches='tight')
        print(f"  → Saved to data/plots/dashboard_k{k1}_vs_k{k2}.png")
    if show:
        plt.show()
    plt.close()
    
    print(f"\nGenerated 3 comparison visualizations in data/plots/")


def quick_viz(season=2024, k_factor=20):
    """Quick visualization for exploration (shows plots interactively)."""
    print(f"Quick visualization: {season} season, K={k_factor}")
    
    elo = EloRatingSystem(k_factor=k_factor)
    elo.calculate_season(season, save_to_db=False)
    
    # Show top teams evolution
    top_teams = elo.get_rankings().head(8)['team'].tolist()
    fig1 = viz.plot_elo_over_time(elo, teams=top_teams, 
                                   title=f'Top 8 Teams Evolution (K={k_factor})')
    plt.show()
    
    # Show prediction accuracy
    fig2 = viz.plot_prediction_accuracy(elo)
    plt.show()
    
    # Show upsets
    fig3 = viz.plot_upset_analysis(elo, top_n=10)
    if fig3:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate NFL Elo visualizations')
    parser.add_argument('--season', type=int, default=2024, help='Season to analyze')
    parser.add_argument('--k', type=int, default=20, help='K-factor')
    parser.add_argument('--home', type=int, default=65, help='Home advantage')
    parser.add_argument('--compare', nargs=2, type=int, metavar=('K1', 'K2'),
                       help='Compare two k-factors (e.g., --compare 20 30)')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    parser.add_argument('--nosave', action='store_true', help='Don\'t save plots to disk')
    parser.add_argument('--quick', action='store_true', help='Quick interactive visualization')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_viz(args.season, args.k)
    elif args.compare:
        compare_k_factors_visual(args.season, args.compare[0], args.compare[1],
                                save=not args.nosave, show=args.show)
    else:
        generate_all_plots(args.season, args.k, args.home, 
                          save=not args.nosave, show=args.show)
    
    if not args.show and not args.nosave:
        print("\nTo view plots, open the PNG files in data/plots/")
        print("Or run with --show to display them interactively")
