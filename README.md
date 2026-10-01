# Chess_JEV_vs_Laya
Chess with JEV vs Laya

python replay_game.py games.pgn --theme walnut
python replay_game.py picked_games.pgn --theme walnut

# final
python replay_game.py picked_games_jev_vs_clm_ft.pgn --theme walnut
python replay_game.py picked_games_jev_vs_clm_base.pgn --theme walnut
python replay_game.py picked_games_clm_base_v_random.pgn --theme walnut

python replay_game.py picked_games_jev_vs_clm_base.pgn --one-game --seconds 0.3 --record jev_vs_clm_base.mp4
python replay_game.py picked_games_jev_vs_clm_ft.pgn --one-game --seconds 0.3 --record jev_vs_clm_ft.mp4
python replay_game.py picked_games_clm_base_v_random.pgn --one-game --seconds 0.3 --record clm_base_vs_random.mp4




Thank you. I am super happy with this result and Agree with the "Five findings" you mentioned above.

# steps

python3.12 -m venv laya_jev_test_env
source laya_jev_test_env/bin/activate
python chess_game.py --white laya --black human
python chess_game.py --white laya --laya-path laya_chess_20k --black random
python chess_game.py --white laya --black random
python chess_game.py --white laya --laya-path laya_chess_20k --black laya


python laya_arena.py --games 20 --laya-path laya_chess_20k


python make_chess_dataset.py --positions 1000 --sample-rate 0.1 --out chess_1k.jsonl
python train_laya_chess.py --data chess_1k.jsonl --out laya_smoke --smoke

caffeinate -i python train_laya_chess.py --data chess_1k.jsonl --out laya_chess_1k --epochs 3


# TO DO
DeepMind fine tuned 'Stockfish' database to RL train for chess --- Laya repo includes a fine-tuning notebook for this.


<!-- Concrete next steps -->
1. Zero-shot baseline: Build the evaluate-every-move player with the stock checkpoint and play it against the random bot. This measures what Laya knows about chess without training.
2. Fine-tune: Generate training data from real positions labeled by Stockfish, then fine-tune on Kaggle or Colab. Your MacBook can run the model but isn't practical for training at this scale.
3. Compare: Play the fine-tuned model against the zero-shot model and the random bot, and evaluate the games with Stockfish.
