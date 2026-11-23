# Commit attribution

When preparing a commit that includes code written by Codex, append a
`Co-authored-by` trailer at the end of the commit message, with a blank line
before the trailer block. Include the exact model identifier after `Codex` when
the model that wrote the code is known.

```text
Co-authored-by: Codex <model identifier> <codex@openai.com>
```

Replace `<model identifier>` with the actual identifier, without angle brackets.
If the model cannot be verified, use `Co-authored-by: Codex <codex@openai.com>`.

# Test organization

Prefer extending existing tests with parameters and adding coverage to existing
test files before creating a new test or test file. When a new test is needed,
make it parameterized where practical so future cases can extend it.

# Formatting

Before committing Python changes, run Black using the version pinned in
`.github/workflows/formatting.yml`, then run `black --diff --check .` from the
repository root to match the formatting CI.
