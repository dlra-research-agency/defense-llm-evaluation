# Defense LLM Evaluation Framework

A systematic evaluation framework for benchmarking large language models on defense and intelligence analysis tasks. Developed by the Defence Language Research Agency (DLRA, Singapore), this framework addresses the gap between general-purpose LLM benchmarks and the specialized requirements of national security and defense applications, providing reproducible evaluation pipelines for threat classification, entity extraction, and maritime intelligence analysis.

## Relationship to Defense NLP Benchmarks

This evaluation framework is complementary to the [Defense NLP Benchmarks](https://github.com/dlra-research/defense-nlp-benchmarks) repository:

- **defense-nlp-benchmarks** defines the benchmark task specifications, curated evaluation datasets, and baseline metrics for defense-domain NLP tasks.
- **defense-llm-evaluation** provides the evaluation harness for systematically benchmarking large language models against those specifications, with support for multiple model providers and extensible metrics computation.

Use **defense-nlp-benchmarks** to understand the benchmark definitions and download evaluation data. Use **defense-llm-evaluation** to run reproducible evaluations of LLM systems against those benchmarks.

## Why Domain-Specific Evaluation Matters

General-purpose benchmarks such as MMLU (Hendrycks et al., 2021), HELM (Liang et al., 2022), and BIG-Bench (Srivastava et al., 2023) provide useful measures of broad language understanding. However, they fall short when applied to defense and intelligence domains for several reasons:

**Vocabulary and terminology gaps.** Defense reports use domain-specific terminology, classification markings, abbreviations, and codenames that rarely appear in general training data. A model scoring well on MMLU's professional knowledge subtasks may still fail to correctly parse an intelligence summary referencing SIGINT collection methods or maritime vessel identification codes.

**Structured output requirements.** Intelligence analysis requires models to produce structured, actionable outputs -- threat severity assessments, entity relationship graphs, and standardized classification labels. General benchmarks primarily evaluate free-form text generation or multiple-choice selection, neither of which captures the structured reasoning that analysts depend on.

**Domain-specific reasoning chains.** Threat assessment involves multi-step reasoning across geopolitical context, historical precedent, and technical indicators. The reasoning patterns required differ substantially from those tested by academic benchmarks. For example, determining whether a cyber intrusion report indicates a state-sponsored actor requires synthesizing technical indicators (TTPs, infrastructure overlap) with geopolitical context -- a reasoning chain that MMLU's cybersecurity questions do not capture.

**Evaluation metric alignment.** Standard NLP metrics like accuracy or perplexity do not map well to the operational requirements of defense applications. An intelligence triage system that misclassifies a critical threat as low-severity has a fundamentally different cost than one that over-classifies routine activity. This framework uses weighted F1 scores, entity-level F1, and ROUGE-L to capture these distinctions.

**Data sensitivity.** Real defense evaluation data cannot be publicly shared. This framework uses synthetically generated evaluation samples that mirror the structure, complexity, and linguistic patterns of real intelligence products without containing classified or sensitive information.

## Evaluation Tasks

### 1. Threat Report Classification

Given an unstructured intelligence report, the model must classify it by threat category and assign a severity level. Categories include cyber-intrusion, terrorism, state-sponsored activity, criminal operations, and insider threats. Severity levels follow a five-point scale: critical, high, medium, low, and informational.

This task evaluates a model's ability to parse complex, multi-paragraph reports and extract the salient indicators that determine threat type and urgency. Reports are synthetically generated to reflect realistic structures including source attribution, temporal markers, and indicator-of-compromise (IOC) descriptions.

Evaluation uses weighted F1 score across all category-severity combinations, with weighting that penalizes critical/high misclassification more heavily than informational-level errors.

### 2. Defense Entity Extraction

Given a defense or intelligence text passage, the model must identify and extract entities of the following types: military units, weapon systems, geographic locations, persons of interest, organizations, operation names, and classification levels.

This task goes beyond standard Named Entity Recognition (NER) benchmarks by including defense-specific entity types (weapon systems, operation codenames, classification markings) that are absent from datasets like CoNLL-2003 or OntoNotes. The evaluation measures exact-match and partial-match entity-level F1, with separate scores for each entity type.

Both zero-shot and few-shot (3-shot, 5-shot) prompting strategies are evaluated to assess how effectively models leverage in-context examples for domain adaptation.

### 3. Maritime Intelligence Analysis

Given structured and semi-structured maritime data -- including AIS (Automatic Identification System) signals, vessel registry information, and patrol reports -- the model must produce analytical summaries that identify vessels of interest, detect anomalous behavior patterns, and generate threat assessments.

This task evaluates generative capabilities rather than classification, measuring the quality of produced analytical text using ROUGE-L for summary quality, BLEU for structural consistency, and entity-overlap F1 to verify that generated summaries correctly reference the vessels, locations, and events present in the source data.

Maritime analysis represents a particularly challenging evaluation domain because it requires integrating structured data (coordinates, timestamps, vessel identifiers) with unstructured text (patrol narratives, historical context) into a coherent analytical product.

## Methodology

All evaluations use a prompt-based approach, sending formatted prompts to model APIs and collecting structured responses. This mirrors the operational deployment pattern where LLMs are integrated into analyst workflows via API calls rather than fine-tuned on classified data.

**Zero-shot evaluation** presents the task description and input data without examples. This measures the model's baseline capability for each task using only its pre-training knowledge and instruction-following ability.

**Few-shot evaluation** (where applicable) provides 3 or 5 labeled examples before the test input. This measures the model's ability to adapt to domain conventions through in-context learning, which is the most practical deployment strategy for defense applications where fine-tuning data is limited or restricted.

**Controlled parameters.** Temperature is set to 0.0 for all evaluations to ensure reproducibility. Maximum token limits are set per task to prevent runaway generation. Each model is evaluated on the identical set of synthetic samples, with sample ordering randomized but consistent across models (fixed seed).

**Statistical reporting.** Results are reported with mean scores and 95% confidence intervals computed via bootstrap resampling (n=1000). All evaluation runs are logged with timestamps, model versions, and API response metadata to ensure reproducibility.

## Results Summary

Evaluation results across all three tasks (200 samples per task, zero-shot prompting):

| Model | Threat Classification (F1) | Entity Extraction (F1) | Maritime Analysis (ROUGE-L) | Average |
|-------|---------------------------|----------------------|---------------------------|---------|
| GPT-4o | 0.87 | 0.82 | 0.74 | 0.81 |
| Claude Sonnet 4 | 0.85 | 0.84 | 0.72 | 0.80 |
| Gemini 2.5 Flash | 0.83 | 0.80 | 0.70 | 0.78 |
| Mistral Large | 0.81 | 0.78 | 0.68 | 0.76 |
| Qwen 2.5 72B | 0.80 | 0.77 | 0.69 | 0.75 |
| Llama 3.1 70B | 0.79 | 0.76 | 0.67 | 0.74 |

**Key observations:**
- All models perform best on threat classification, suggesting this task aligns most closely with general pre-training data distributions.
- Claude Sonnet 4 achieves the highest entity extraction score, indicating strong structured output capabilities.
- Maritime analysis scores are uniformly lower across all models, reflecting the difficulty of integrating structured data with analytical reasoning.
- The gap between proprietary and open-weight models (5-7 points average) is smaller than observed on general benchmarks, suggesting that defense-domain performance is less correlated with model scale than with training data composition.

## Quick Start

### Installation

```bash
git clone https://github.com/dlra-research/defense-llm-evaluation.git
cd defense-llm-evaluation
pip install -r requirements.txt
```

### Set API Keys

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Run Evaluations

```bash
# Run threat classification evaluation
python eval/run_eval.py --config configs/threat-classification.yaml

# Run entity extraction evaluation
python eval/run_eval.py --config configs/entity-extraction.yaml

# Run maritime analysis evaluation
python eval/run_eval.py --config configs/maritime-analysis.yaml

# Run all evaluations
python eval/run_eval.py --config configs/threat-classification.yaml configs/entity-extraction.yaml configs/maritime-analysis.yaml
```

Results are saved to the `results/` directory as timestamped JSON files.

## Configuration

Evaluation configurations are defined in YAML files under `configs/`. Each configuration specifies:

- **task**: Task name, description, and version.
- **model_configs**: List of models to evaluate, with provider, model identifier, temperature, and max token settings.
- **evaluation**: Evaluation mode (zero-shot or few-shot), metrics to compute, number of samples, and task-specific parameters (categories, entity types, sub-tasks).

To add a new model, append an entry to the `model_configs` list in the relevant YAML file:

```yaml
model_configs:
  - provider: openai
    model: gpt-4o
    temperature: 0.0
    max_tokens: 256
  - provider: your-provider
    model: your-model-id
    temperature: 0.0
    max_tokens: 256
```

Supported providers: `openai`, `anthropic`. Additional providers can be added by extending `eval/models.py`.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{goh2024defense,
  title={Defense LLM Evaluation Framework},
  author={Goh, Brandon Zhi Hao and Ong, Cheryl Hui Wen and Tan, Shu Ling and Lim, Wei Jie},
  year={2024},
  publisher={DLRA (Defence Language Research Agency)},
  url={https://github.com/dlra-research/defense-llm-evaluation},
  version={1.0}
}
```

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

Areas where contributions are particularly valued:
- Additional evaluation tasks (e.g., OSINT analysis, signals intelligence summarization)
- Support for additional model providers
- Improved synthetic data generation pipelines
- Multilingual evaluation samples (especially Southeast Asian languages)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## About DLRA

The Defence Language Research Agency (DLRA) is a Singapore-based research organization founded in 2024 that develops natural language processing tools for defense and intelligence applications. DLRA's products include DLRA Threat Lens for intelligence triage, DLRA Maritime NLP for vessel tracking and anomaly detection, and DLRA SynthBrief for automated intelligence summary generation. For more information, visit [dlra-research.org](https://dlra-research.org).
