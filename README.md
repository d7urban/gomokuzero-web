# GomokuZero Web

Play Gomoku (Five in a Row) against a neural network trained with AlphaZero-style self-play.

**[Play online](https://gomokuzero-web.vercel.app)**

## Features

- **Human vs AI** with adjustable difficulty (250–5000 MCTS simulations)
- **Human vs Human** mode
- **Analysis heatmap** — MCTS visit counts and win% overlay
- Undo, save/load games, switch sides mid-game
- 15×15 board, standard Gomoku rules (first to five in a row)

## Architecture

The app is designed for stateless serverless deployment (Vercel):

- **Client** (JS) — owns all game state, renders the board on canvas, manages game logic
- **Server** (Python/Flask) — two stateless compute endpoints:
  - `POST /api/ai_move` — given move history, runs MCTS and returns the AI's move
  - `POST /api/analyze` — given move history, runs MCTS and returns policy/value heatmap data
- **Model** — TFLite (~12 MB), 10-block residual CNN with 128 filters

## Run locally

```bash
pip install flask numpy ai-edge-litert
python play_web.py
```

Open http://127.0.0.1:5000

## Deploy to Vercel

1. Fork or clone this repo
2. Import the project in [Vercel](https://vercel.com)
3. Deploy — no configuration needed (`vercel.json` is included)

## License

[GPL-3.0](LICENSE)
