from datasets import load_dataset

ds = load_dataset("badayvedat/VCTK", split="train")
print(ds.column_names)