
# M-GRPO: Stabilizing Self-Supervised ReinforcementLearning for Large Language Models with Momentum-Anchored Policy Optimization


<img width="707" height="219" alt="m_grpo drawio" src="https://github.com/user-attachments/assets/01243d50-ce1f-43ea-8b74-e22a66127ab1" />

## Motivation: Existing self-supervised reinforcement learning (RL) are prone to "policy collapse.
Self-supervised reinforcement learning (RL) presents a promising avenue for enhancing the reasoning capabilities of LLMs without reliance on expensive human-annotated data. However, existing methods are prone to "policy collapse," a phenomenon where the learning process becomes unstable during extended training, leading to a sharp degradation in both reward and task performance. This paper diagnoses this instability, attributing it to the lack of a stable target in self-rewarding systems. To address this, we introduce M-GRPO, a  momentum-anchored method that leverages a slowly evolving momentum model to provide a consistent and reliable training signal, stabilizing the generation of pseudo-labels for policy optimization. Our experiments, conducted on the MATH dataset, demonstrate that M-GRPO effectively prevents policy collapse, maintaining a stable training reward and consistently high validation accuracy. 
First, download the MATH dataset and prepare it using the following Python script:
![Xnip2025-09-12_00-58-07](https://github.com/user-attachments/assets/f232251f-ed62-41df-bdbc-9c6ba7b50ee3)

## How to use
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
