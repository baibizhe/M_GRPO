
# verl-M-GRPO

First, download the MATH dataset and prepare it using the following Python script:

```bash
python examples/data_preprocess/math_dataset_ours.py --model Qwen2.5-3B
```

Then, run the following command to start the training (Modify the WANDB_KEY in the `math_intuitor.sh` script to your own WANDB key.):

```bash
bash math_intuitor.sh
```

## 📚 References

This project builds upon the following open-source repositories:

- [intuitor](https://github.com/sunblaze-ucb/Intuitor) License: [Apache License 2.0](https://github.com/volcengine/verl/blob/main/LICENSE)

- [open-r1](https://github.com/huggingface/open-r1) License: [Apache License 2.0](https://github.com/huggingface/open-r1/blob/main/LICENSE)

- [verl](https://github.com/volcengine/verl) License: [Apache License 2.0](https://github.com/volcengine/verl/blob/main/LICENSE)
