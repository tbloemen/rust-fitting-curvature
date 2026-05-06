cargo run -p fitting-optimizer --release -- --mode pareto --output results/off_3.jsonl --experiment all_off --dataset real
cargo run -p fitting-optimizer --release -- --mode pareto --output results/centering.jsonl --experiment centering_only --dataset real
cargo run -p fitting-optimizer --release -- --mode pareto --output results/norm.jsonl --experiment norm_only --dataset real
cargo run -p fitting-optimizer --release -- --mode pareto --output results/global.jsonl --experiment global_only --dataset real
cargo run -p fitting-optimizer --release -- --mode pareto --output results/free.jsonl --experiment all_free --dataset real
cargo run -p fitting-optimizer --release -- --mode pareto --output results/off_3_5k.jsonl --experiment all_off --dataset real --n_samples 5000
cargo run -p fitting-optimizer --release -- --mode pareto --output results/centering_5k.jsonl --experiment centering_only --dataset real --n_samples 5000
cargo run -p fitting-optimizer --release -- --mode pareto --output results/norm_5k.jsonl --experiment norm_only --dataset real --n_samples 5000
cargo run -p fitting-optimizer --release -- --mode pareto --output results/global_5k.jsonl --experiment global_only --dataset real --n_samples 5000
cargo run -p fitting-optimizer --release -- --mode pareto --output results/free_5k.jsonl --experiment all_free --dataset real --n_samples 5000
