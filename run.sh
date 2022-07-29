
python3 main.py --is_debug_mode 1 --seed 42 --model_name 'roberta-base' --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 2 --learning_rate 5e-5 --num_train_epochs 5 --train_file_path raw_data/qg_train.json --valid_file_path raw_data/qg_valid.json --output_dir ./test1905/