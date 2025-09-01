# 文件名: dynamic_few_shot.py

import torch
from typing import List, Tuple

class FewShotManager:
    """
    Manages the creation and injection of dynamic few-shot examples
    based on incorrect predictions from the LLM.
    """
    def __init__(self, max_examples: int = 3):
        """
        Initializes the FewShotManager.

        Args:
            max_examples (int): The maximum number of few-shot examples to generate and store.
        """
        self.max_examples = max_examples
        self.examples = []
        print(f"FewShotManager initialized with a capacity of {max_examples} examples.")

    def can_add_more(self) -> bool:
        """Checks if there is space to add a new example."""
        return len(self.examples) < self.max_examples

    def get_few_shot_context(self) -> str:
        """
        Returns all stored few-shot examples formatted as a single string
        to be prepended to the main prompt.
        """
        if not self.examples:
            return ""
        
        header = "### Expert Reasoning Examples\n"
        header += "Here are some previous cases with expert analysis. Learn from them:\n\n"
        return header + "\n\n".join(self.examples)

    def generate_and_add_example(
        self,
        prompt_preamble: str,
        failed_prompt_template: str,
        q_head: int,
        v_true: int,
        v_neg: int,
        model, # The LLM model
        tokenizer,
        device: str
    ):
        """
        Generates a reasoning for a failed prediction and adds it as a new few-shot example.
        """
        if not self.can_add_more():
            print("Few-shot example capacity reached. No new example will be generated.")
            return

        print("\n--- Generating a new few-shot example from a failed prediction... ---")

        # 1. Construct the prompt for generating the reasoning
        reasoning_question = (
            f"Question: Which node is more likely to link with node {q_head}?\n  A: {v_true}\n  B: {v_neg}\n"
            f"The correct answer was A: {v_true}. Based on the context provided in the prompt, please provide a brief analysis explaining why node {v_true} was the more likely connection.\n"
            "Reasoning: "
        )
        
        messages_for_reasoning = [
            {"role": "system", "content": prompt_preamble},
            {"role": "user", "content": failed_prompt_template + reasoning_question}
        ]

        text_for_reasoning = tokenizer.apply_chat_template(
            messages_for_reasoning,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # <-- 核心改动：关闭思考功能
        )

        inputs = tokenizer([text_for_reasoning], return_tensors="pt").to(device)
        
        # 2. Generate the reasoning from the LLM
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the newly generated reasoning part
        newly_generated_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        reasoning_text = tokenizer.decode(newly_generated_ids, skip_special_tokens=True).strip()

        # 3. Format the complete few-shot example
        # The prompt part of the example includes the original question
        original_question = f"Question: Which node is more likely to link with node {q_head}?\n  A: {v_true}\n  B: {v_neg}\nAnswer: "
        
        complete_example = (
            f"--- Example Start ---\n"
            f"**Prompt Context:**\n{failed_prompt_template}\n"
            f"**{original_question}A**\n"
            f"**Expert Analysis:**\n{reasoning_text}\n"
            f"--- Example End ---\n"
        )

        self.examples.append(complete_example)
        print(f"--- New few-shot example added. Total examples: {len(self.examples)} ---")