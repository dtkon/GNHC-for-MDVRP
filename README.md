# GNHC-for-MDVRP

This repo implements our paper: Generalizable neural hybrid construction for multi-depot vehicle routing problems

## Dependencies
* Python>=3.12
* PyTorch>=2.8
* tqdm

## Usage

### Training

Normal scale:
```bash
python -m src --log_step 10 --problem mdvrp --model LEHDca_att_da --customer_num 50 --depot_num 3 --batch_size 64 --epoch_size 160000 --backward_len 15 --run_name HDda
```

Large scale:
```bash
python -m src --log_step 1 --problem mdvrp --model LEHDca_att_da --customer_num 500 --depot_num 3 --batch_size 16 --epoch_size 16000 --epoch_end 10 --backward_len 5 --run_name HDda --load_path saved_model/mdvrp-50-3/HDda_20250813T180352/epoch-97.pt
```

Need specify `--load_path` to transfer from normal scale model.

### Inference

```bash
--eval_only 
--load_path '{add model to load here}'
--val_range 0 100 
--val_batch_size 100
--val_dataset '{add dataset here}' 
--min_seg_len 4
--max_seg_len 15
--val_LC_iter 10
--eval_type greedy_aug
--sample_times 10
--enable_LC
```

#### Examples

For inference 100 MDVRP instances with 50 customers and 3 depots:

```bash
python -m src --no_log --eval_only --problem mdvrp --model LEHDca_att_da --val_batch_size 100 --val_range 0 100 --min_seg_len 4 --max_seg_len 22 --val_LC_iter 10 --eval_type greedy_aug --sample_times 10 --run_name eval_LEHDca_att_da_lc10 --load_path saved_model/mdvrp-50-3/HDda_20250813T180352/epoch-97.pt --customer_num 50 --depot_num 3 --enable_LC
```

For inference 10 MDVRP instances with 1000 customers and 4 depots:

```bash
python -m src --no_log --eval_only --problem mdvrp --model LEHDca_att_da --val_batch_size 10 --val_range 0 10 --min_seg_len 4 --max_seg_len 200 --val_LC_iter 10 --eval_type greedy_aug --sample_times 10 --run_name eval_LEHDca_att_da_lc10 --load_path saved_model/mdvrp-500-3/HDda_20250914T112649/epoch-8.pt --customer_num 1000 --depot_num 4 --enable_LC
```

Run ```python -m src -h``` for detailed help on the meaning of each argument.
