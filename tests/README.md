# Run Tests

Set this environment variable to a working directory that can store the Hugging Face cache and checkpoints created by the tests:

```bash
export MISTRAL_TEST_HOME=/path/to/mistral-test-working-dir
```

From the `tests` directory, run this command to run tests in single node/single GPU mode:

```bash
export CUDA_VISIBLE_DEVICES=0
cd tests
pytest
```
