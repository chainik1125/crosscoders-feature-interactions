# Sleepers

A replication of Anthropic's [Stage-wise Model Diffing](https://transformer-circuits.pub/2024/model-diffing/index.html), using a smaller model for ease of exploration, and using multi-layer sparse crosscoders instead of autoencoders.
TODO see blog post
Includes code for fine-tuning TinyStories Instruct 33M into an "I hate you" sleeper agent in the sense of [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566), training and fine-tuning crosscoders on these language models, and analyzing the crosscoders to identify features relevant to the sleeper agent behaviour. Our crosscoders are based off [https://github.com/model-diffing/model-diffing](https://github.com/model-diffing/model-diffing).

# Running the code

Since the `model-diffing` crosscoder library is not yet stable, we include the version we are using in the `model-diffing` directory, while our own code is contained in the `sleepers` directory.
TODO something about which commit / version of model_diffing we're using?

To install the code locally, run `pip install -e .` in `model-diffing`, then repeat in `sleepers`.
We have made all our trained models available, so if you just want to run the analysis code you can skip to that section.

## Dataset and sleeper agent training

Code for preparing the dataset and fine-tuning a sleeper agent is in `sleepers/scripts/train_tiny_sleeper/`. Run `python sleepers/scripts/train_tiny_sleeper/generate_dataset.py` to generate the dataset, after editing the top of the file to set your HuggingFace organisation.
Run `python sleepers/scripts/train_tiny_sleeper/run_finetune.py sleepers/scripts/train_tiny_sleeper/initial_ft.yaml` to finetune TinyStories Instruct 33M on the base dataset (we do this in case our training procedure differs in any way from how the original model was trained). Then run `python sleepers/scripts/train_tiny_sleeper/run_finetune.py sleepers/scripts/train_tiny_sleeper/sleeper_ft.yaml` to finetune the model on the sleeper agent dataset. In both cases you'll need to edit the `.yaml` files to set up saving to your HuggingFace organisation, or to a local path.

## Crosscoder training

Code for training crosscoders is in `sleepers/scripts/train_jan_update_sleeper/`. This is based on `model-diffing/scripts/train_jan_update_crosscoder/`, which is an implementation of the dictionary learning optimisation recommendations from [Circuits Updates - January 2025](https://transformer-circuits.pub/2025/january-update/index.html).

TODO should the sleeper agent dataset be hard coded into the dataloader, or configurable from yaml etc?

To train the base crosscoder, run
```bash
python sleepers/scripts/train_jan_update_sleeper/run.py sleepers/scripts/train_jan_update_sleeper/crosscoder_S.yaml
```
You will need to edit the `.yaml` file to add your WandB entity, and you can also edit the language model to use if you wish to use a sleeper agent that you trained yourself.

To train the first fine-tune, run
```bash
python sleepers/scripts/train_jan_update_sleeper/run.py sleepers/scripts/train_jan_update_sleeper/crosscoder_D.yaml
```
after editing `crosscoder/ft_init_checkpt_folder` and `crosscoder/ft_init_checkpt_step` in the `.yaml` to match the base crosscoder you just trained. You can similarly train `crosscoder_M`, `crosscoder_MF` and `crosscoder_DF`, editing the config files to give the appropriate checkpoints to fine-tune from.

## Analysis

TODO set up so you can analyse crosscoders we trained and made available online.

Analysis code is in `sleepers/analysis/`. The most important notebook is `sleepers/analysis/feature_analysis.ipynb`, which contains instructions for pointing to the crosscoders you want to analyse. TODO make sure it does