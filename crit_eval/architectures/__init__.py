"""Architecture-specific code the reference needs, one place for all of it: GGUF checkpoints read in the layout
transformers' modeling code expects, per architecture, and faster forms of modeling code that would otherwise run per
token in Python."""
