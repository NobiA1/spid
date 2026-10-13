# P01: Semantic Indistinguishability and Controllability

P01 tests whether investor attributes expressed in natural language map to trading behavior in a stable, monotonic, and controllable way. The same single-step portfolio task is evaluated through three control interfaces:

- **Natural-language risk levels** (`L0`–`L9`)
- **Numeric parameters injected into prompts**
- **Explicit utility control** (LLM emits semantic judgment; utility layer generates actions)

## Protocol

- **MFT**: strong bullish/bearish scenarios to verify basic directional judgment
- **DIR**: monotonicity and local reversals across the three interfaces
- **INV**: paraphrase stability at the same nominal risk level

## Results

Prompt-only control is unreliable at the micro level. Natural-language risk levels show a weak average trend but heavy overlap and local reversals between adjacent levels. Numeric prompt injection does not fully remove this instability. Explicit utility control is more stable because the risk parameter enters the trading response function directly.

The invariance test shows that semantically similar prompts do not guarantee behaviorally similar outputs, motivating SUM's separation between semantic interpretation and utility-based action generation.

## Figures

![Natural-language risk gradient](assets/P01_1_nl_gradient.png)

![Three-interface controllability](assets/P01_2_three_condition.png)

![Behavioral distance across risk levels](assets/P01_3_inv_jsd.png)

![Semantic vs behavioral distance](assets/P01_4_sem_beh_scatter.png)

## Data

The full reproduction data is not redistributed in this anonymized release. Model
outputs are written under a `data/controllability/P01/` results tree by `run.py`;
fake credentials and synthetic inputs can be produced with
`../../00_prompts_and_protocol/fake_data/make_fake_data.py`.
