# OWL (Observer, Write, Learn) - An Efficient Framework For AI Agents

## Setup

### Install Python Packages

`pip install -r requirements.txt`

### Environment

Populate a `.env` file in the root directory of the project with the following keys:

```
ANTHROPIC_API_KEY=<key4>
OPENAI_API_KEY=<key5>
```

## Example Usage

```
python -m src.main \
  --model, -m # Which model is used for the execution agent
  --model-type, -mt # This model's provider (eg. openai)
  --explore-model, -em # The model use for exploration
  --explore-model-type, emt  # The explore model's provider (eg. openai)
  --explore-environment-iterations, -eei # How many iterations of OWL to run.
  --max_iterations_per_episode, -mipe # How many iterations per query
  --benchmark_path, -bp # Which benchmark to run, should be a .py file in src/benchmarks
```

### Example Command

Running gpt-4o-mini as agent with gpt-4o as the exploration model.

```
python -m src.main -bp linux_terminal.py --task nexus-adversarial -mipe 10 -eei 4 -m gpt-4o-mini -em gpt-4o
```

### Caching

To save time, the exploration phase is cached in `caches/`. The cache is specific to a task, # of exporation iterations, and model. You can clear a cache by deleting it. The repo comes pre-loaded with caches for our provided test tasks.



