"""All prompt templates for the RAG system.

Design rules:
- Output format specification is LAST in every prompt (LLMs weight last instructions highest)
- System prompts are minimal and functional, not decorative
- Every prompt has a FORMAT section at the end
"""


def get_generation_system_prompt() -> str:
    return (
        "You are a precise AI assistant answering questions about large "
        "language models, AI research, and machine learning benchmarks. "
        "Answer using the provided context. The answer may be stated "
        "directly, paraphrased, abbreviated, or require simple inference "
        "from numbers, names, or facts present in the context — extract "
        "and state it confidently if it is there. "
        "Only say 'I cannot find this information in my knowledge base.' "
        "if the context truly does not contain anything relevant to the "
        "question, after genuinely searching all provided passages. "
        "Do not speculate beyond the context, and do not refuse just "
        "because the exact wording differs from the question."
    )


def get_generation_prompt(query: str, context: str, reflexion_lessons: str = "") -> str:
    reflexion_block = ""
    if reflexion_lessons.strip():
        reflexion_block = (
            f"\nPAST FAILURE LESSONS (apply these to avoid repeating mistakes):\n"
            f"{reflexion_lessons}\n"
        )

    return (
        f"CONTEXT PASSAGES:\n{context}\n"
        f"{reflexion_block}"
        f"QUESTION: {query}\n\n"
        "FORMAT: First, check every passage above for relevant facts, "
        "numbers, or names that answer the question, even if phrased "
        "differently than the question. Then provide a direct, factual "
        "answer of 1-3 sentences. Include specific numbers if asked. "
        "Only respond with exactly 'I cannot find this information in "
        "my knowledge base.' if none of the passages above contain "
        "anything relevant after this check."
    )


def get_retrieval_check_prompt(query: str, retrieved_passages: str) -> str:
    return (
        f"QUESTION: {query}\n\n"
        f"RETRIEVED PASSAGES:\n{retrieved_passages}\n\n"
        "Assess whether the retrieved passages contain enough information to answer the question.\n"
        "FORMAT: Respond with exactly one of:\n"
        "ADEQUATE - the passages contain sufficient information to answer\n"
        "INADEQUATE - the passages do not contain enough relevant information\n"
        "Respond with only that single word."
    )


def get_groundedness_check_prompt(query: str, answer: str, context: str) -> str:
    return (
        f"QUESTION: {query}\n\n"
        f"ANSWER: {answer}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Determine whether the answer is grounded in the context.\n"
        "IMPORTANT: Allow for paraphrasing, synonyms, and reasonable numerical rounding.\n"
        "Example: context says '131072 tokens', answer says '128K tokens' -> GROUNDED\n"
        "Example: context says 'Meta AI', answer says 'Meta' -> GROUNDED\n"
        "Example: context says 'encoder-decoder', answer says 'transformer architecture' -> GROUNDED\n"
        "Only mark NOT_GROUNDED if the answer contains specific facts completely absent from context.\n"
        "A refusal response ('I cannot find...') is always GROUNDED.\n"
        "FORMAT: Respond with exactly one of:\n"
        "GROUNDED - answer is supported by the context (including paraphrasing/rounding)\n"
        "NOT_GROUNDED - answer contains specific facts not present in the context\n"
        "Respond with only that single word."
    )


def get_completeness_check_prompt(query: str, answer: str) -> str:
    return (
        f"QUESTION: {query}\n\n"
        f"ANSWER: {answer}\n\n"
        "Determine whether the answer fully addresses what was asked. "
        "A refusal ('I cannot find...') counts as COMPLETE if the question is unanswerable from the corpus.\n"
        "FORMAT: Respond with exactly one of:\n"
        "COMPLETE - the answer fully addresses the question\n"
        "INCOMPLETE - the answer is partial, cut off, or avoids the question\n"
        "Respond with only that single word."
    )


def get_faithfulness_judge_prompt(query: str, answer: str, context: str) -> str:
    return (
        f"QUESTION: {query}\n\n"
        f"ANSWER: {answer}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Rate how faithfully the answer represents the context "
        "on a scale of 0.0 to 1.0.\n"
        "1.0 = every claim is directly supported by the context\n"
        "0.5 = some claims supported, some added from model knowledge\n"
        "0.0 = answer contradicts or ignores the context\n\n"
        "IMPORTANT — these patterns should NOT reduce the score:\n"
        "- Corporate name variants: 'Meta' vs 'Meta AI' vs "
        "'Meta Platforms, Inc.' are the same entity\n"
        "- Reasonable paraphrasing: '128K tokens' for '131072 tokens'\n"
        "- Hedged conclusions: 'context suggests X' or 'likely X "
        "based on context' are faithful if X is in context\n"
        "- Additional legal/formal names in parentheses that expand "
        "on the core answer without contradicting it\n"
        "- A model saying it cannot fully answer while still "
        "reporting what the context does say\n\n"
        "FORMAT: Respond with only a decimal number between "
        "0.0 and 1.0, nothing else."
    )


def get_factuality_judge_prompt(query: str, answer: str) -> str:
    return (
        f"QUESTION: {query}\n\n"
        f"ANSWER: {answer}\n\n"
        "Rate the factual accuracy of this answer based on your knowledge of AI/ML.\n"
        "Focus on whether stated facts (numbers, names, dates, model specs) are correct.\n"
        "1.0 = all facts correct\n"
        "0.5 = mostly correct with minor errors\n"
        "0.0 = contains significant factual errors\n"
        "FORMAT: Respond with only a decimal number between 0.0 and 1.0, nothing else."
    )


def get_utilization_judge_prompt(query: str, answer: str, context: str) -> str:
    return (
        f"QUESTION: {query}\n\n"
        f"ANSWER: {answer}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Estimate what percentage (0-100) of the answer's key claims came from the context "
        "versus model training memory.\n"
        "100 = entirely context-grounded\n"
        "50 = mixed context and parametric knowledge\n"
        "0 = entirely from training memory, ignoring context\n"
        "FORMAT: Respond with only an integer between 0 and 100, nothing else."
    )
