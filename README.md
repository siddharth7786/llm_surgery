# LLM Surgery: Modifying Model Behavior at Runtime
## Project Structure

- `original.py`: Demonstrates the baseline refusal behavior of the original model.
- `modified.py`: Performs the activation engineering "surgery" to bypass the refusal mechanism.
- `diagnose.py`: A diagnostic tool to analyze the model's internal state and locate the refusal signal across its layers.

## How to Run

**Prerequisites:**
Ensure you have the necessary libraries installed:
```bash
pip install mlx mlx-lm
```

To see the demonstration, run the scripts in the following order:

1.  **See the original model's refusal:**
    ```bash
    python original.py
    ```

2.  **Analyze the model to find the refusal signal:**
    ```bash
    python diagnose.py
    ```

3.  **See the modified model's bypassed response:**
    ```bash
    python modified.py
    ```

---
