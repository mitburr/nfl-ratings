# Setting Up Claude Project: NFL Rating System

## Quick Setup

1. **Create New Project** in Claude
   - Name: "NFL Elo Rating System"
   
2. **Add Project Description:**
   ```
   NFL game prediction system using Elo ratings + team performance vectors.
   Python/PostgreSQL. Data from nflverse. Currently testing vector approaches
   to improve 64% Elo baseline. See PROJECT_STATE.md for full context.
   ```

3. **Upload These Files:**
   - PROJECT_STATE.md (comprehensive overview)
   - CODE_PATTERNS.md (⭐ how to write code that integrates properly)
   - src/models/elo.py (Elo system)
   - src/analysis/team_vectors.py (vector calculations)
   - src/database/db_manager.py (database interface)
   - src/ingestion/nfl_data.py (data loading)
   - scripts/predict_game.py (main prediction)
   - scripts/test_vectors_proper.py (vector testing)
   - README.md (project overview)
   - SETUP_GUIDE.md (installation)

4. **Set Custom Instructions:**
   Copy from CLAUDE_PROJECT_SETUP.md "Custom Instructions for Claude" section

## Key Context for Claude

**Critical Rules:**
- NO data leakage: use through_week parameter
- Test across multiple seasons, not just one
- nflreadpy returns Polars → must call .to_pandas()
- Be skeptical of improvements that seem too good
- **⭐ ALWAYS integrate with existing code - never create standalone scripts!**

**Code Integration (see CODE_PATTERNS.md):**
- Import from `src/` modules, don't duplicate classes
- Use `DatabaseManager`, not raw psycopg2
- Never hardcode credentials
- Follow existing script patterns in `scripts/`

**Current Focus:**
Testing weighted vectors (25% prior + 75% current season) vs cumulative
(current season only) to improve prediction accuracy beyond Elo baseline.

**Tech Stack:**
Python 3.13, PostgreSQL, pandas, matplotlib, nflreadpy

That's it! Claude will have full context on the project state and can help
with code, analysis, debugging, and research.