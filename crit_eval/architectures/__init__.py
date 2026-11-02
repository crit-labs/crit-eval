"""Architecture-specific code the reference needs, one place for all of it: GGUF checkpoints read in the layout
transformers' modeling code expects and the config.json their metadata derives (the config rules crit-core reads
through its registry, gguf_config), per architecture, the chat templates crit ships for families whose checkpoints
carry none (CHAT_TEMPLATES, crit-core's chat_templates), and faster forms of modeling code that would otherwise run
per token in Python."""
from pathlib import Path

# config.json model_type -> the chat template of a family whose checkpoints carry none: DeepSeek V4 publishes a
# Python encoder (encoding_dsv4.py) instead, which templates/deepseek-v4.jinja ports (its golden tests:
# tests/test_eval_deepseek4_template.py).
CHAT_TEMPLATES = {"deepseek_v4": Path(__file__).resolve().parent / "templates" / "deepseek-v4.jinja"}
