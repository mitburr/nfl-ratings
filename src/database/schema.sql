-- NFL Stats Database Schema

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    team_id VARCHAR(3) PRIMARY KEY,
    team_name VARCHAR(100) NOT NULL,
    conference VARCHAR(3),
    division VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Games table
CREATE TABLE IF NOT EXISTS games (
    game_id VARCHAR(20) PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_type VARCHAR(4),  -- REG, WC, DIV, CONF, SB
    game_date DATE NOT NULL,
    home_team VARCHAR(3) REFERENCES teams(team_id),
    away_team VARCHAR(3) REFERENCES teams(team_id),
    home_score INTEGER,
    away_score INTEGER,
    stadium VARCHAR(100),
    roof VARCHAR(20),
    surface VARCHAR(50),
    temp NUMERIC,
    wind NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Elo ratings table (historical tracking)
CREATE TABLE IF NOT EXISTS elo_ratings (
    id SERIAL PRIMARY KEY,
    team_id VARCHAR(3) REFERENCES teams(team_id),
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    rating NUMERIC NOT NULL,
    games_played INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, season, week)
);

-- Play-by-play data (for EPA analysis)
CREATE TABLE IF NOT EXISTS plays (
    play_id VARCHAR(50) PRIMARY KEY,
    game_id VARCHAR(20) REFERENCES games(game_id),
    play_type VARCHAR(50),
    posteam VARCHAR(3) REFERENCES teams(team_id),
    defteam VARCHAR(3) REFERENCES teams(team_id),
    quarter INTEGER,
    down INTEGER,
    yards_to_go INTEGER,
    yardline_100 INTEGER,
    epa NUMERIC,
    wpa NUMERIC,
    success BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_games_season_week ON games(season, week);
CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_elo_ratings_season ON elo_ratings(season, week);
CREATE INDEX IF NOT EXISTS idx_plays_game ON plays(game_id);
CREATE INDEX IF NOT EXISTS idx_plays_team ON plays(posteam);

-- View for quick game results
CREATE OR REPLACE VIEW game_results AS
SELECT 
    g.game_id,
    g.season,
    g.week,
    g.game_type,
    g.game_date,
    g.home_team,
    ht.team_name as home_team_name,
    g.home_score,
    g.away_team,
    at.team_name as away_team_name,
    g.away_score,
    CASE 
        WHEN g.home_score > g.away_score THEN g.home_team
        WHEN g.away_score > g.home_score THEN g.away_team
        ELSE NULL
    END as winner,
    ABS(g.home_score - g.away_score) as point_differential
FROM games g
JOIN teams ht ON g.home_team = ht.team_id
JOIN teams at ON g.away_team = at.team_id
WHERE g.home_score IS NOT NULL;
