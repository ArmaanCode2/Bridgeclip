# CI Python tools

Hash-locked test and audit tools for CI and release jobs. They never ship in
the app; they are installed on the runner (pytest also into a throwaway
`--target` directory for the packaged interpreter). Regenerate after changing
a `.txt` file, from this directory:

    uv pip compile pytest.txt --universal --python-version 3.12 --generate-hashes --output-file pytest.lock
    uv pip compile pip-audit.txt --universal --python-version 3.12 --generate-hashes --output-file pip-audit.lock
