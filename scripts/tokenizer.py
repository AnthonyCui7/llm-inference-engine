"""
GPT-2 tokenizer helper for the C++ driver, which takes raw token ids.

  python3 scripts/tokenizer.py encode "some text"
  python3 scripts/tokenizer.py decode 464 3290 ...
"""

import sys

from transformers import GPT2Tokenizer


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("encode", "decode"):
        print("usage: tokenizer.py encode <text> | decode <id> [id ...]", file=sys.stderr)
        sys.exit(1)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if sys.argv[1] == "encode":
        ids = tokenizer.encode(" ".join(sys.argv[2:]))
        print(" ".join(str(t) for t in ids))
    else:
        print(tokenizer.decode([int(t) for t in sys.argv[2:]]))


if __name__ == "__main__":
    main()
