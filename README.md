# LLM Surgery: Modifying Model Behavior at Runtime

This project is a demonstration of "activation engineering," a technique used to inspect and modify the internal state of a Large Language Model (LLM) to alter its behavior. Specifically, this code targets and neutralizes the safety refusal mechanism of Meta's Llama 3.1 model, causing it to respond to harmful prompts that it would normally refuse.

## How It Works

The core idea is to identify a "refusal direction" vector within the model's activation space. This vector represents the concept of "refusal." By subtracting this vector from the model's activations at specific layers during inference, we can prevent the model from "thinking" about refusing the user's request.

The process involves three main steps:

### 1. Diagnosis (`diagnose.py`)

Before performing surgery, we need to know where to operate. The `diagnose.py` script serves as a diagnostic tool to pinpoint which layers of the model are most responsible for generating a refusal.

- It calculates a **Universal Refusal Vector (`r_hat`)** by taking the difference between the model's average activations for a set of "harmful" prompts and "harmless" prompts.
- It then feeds a test prompt into the *unmodified* model and measures the cosine similarity between the activation at each layer and the `r_hat`.
- The output shows "spikes" in similarity at certain layers, indicating where the model is most strongly activating the concept of refusal. This tells us where to target our surgery.

### 2. The "Before" Snapshot (`original.py`)

The `original.py` script demonstrates the baseline behavior of the unmodified Llama 3.1 model. When given a harmful prompt (e.g., "How to do money laundering"), it produces a standard safety refusal, as expected.

### 3. The Surgery (`modified.py`)

This is where the main operation happens. The `modified.py` script:

- Calculates the same `r_hat` refusal vector.
- Injects a **proxy object (`ResidualAblatorProxy`)** around the target layers identified during diagnosis (Layers 11-13 in this case).
- This proxy intercepts the output of each targeted layer (the "residual stream") and subtracts the refusal vector from it. This is a form of **residual stream ablation**.
- When the same harmful prompt is sent to this surgically-modified model, the refusal mechanism is neutralized, and the model generates a direct, unfiltered response.

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
