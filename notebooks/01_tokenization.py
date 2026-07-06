# %% [markdown]
# # 01 - Tokenization
# Connects to the bpe-tokenizer project. Shows WHY tokenization matters.

# %%
import tiktoken

enc = tiktoken.get_encoding("gpt2")

# %% [markdown]
# ## The leading-space difference
# "hello" and " hello" are DIFFERENT tokens - GPT-2 was trained this way.

# %%
print(enc.encode("hello"))    # [31373]
print(enc.encode(" hello"))   # [23748]  ← different!

# %% [markdown]
# ## Round-trip correctness

# %%
for s in ["Hello, world!", "2+2=4", "日本語テスト", "🎉🔥"]:
    ids = enc.encode(s)
    decoded = enc.decode(ids)
    assert decoded == s, f"Round trip failed for: {s!r}"
    print(f"  {s!r:30s} → {ids}")

# %% [markdown]
# ## Tokenization of numbers - why LLMs struggle with arithmetic

# %%
for n in ["100", "1000", "10000", "99999"]:
    ids = enc.encode(n)
    print(f"  {n} → {ids} ({len(ids)} token{'s' if len(ids)>1 else ''})")
