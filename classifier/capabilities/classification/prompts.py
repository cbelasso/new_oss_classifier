from typing import Any, Dict


def standard_classification_prompt(node_config: Dict[str, Any]) -> str:
    """
    Generate a standard single-node classification prompt.

    This prompt focuses on binary classification of whether a text discusses
    the target topic as a main subject (not just in passing).

    Args:
        node_config: Dictionary containing node configuration with keys:
            - name: Topic name
            - description: Detailed topic description
            - keywords: List of relevant keywords
            - scope: Scope definition

    Returns:
        Formatted prompt string for the LLM
    """
    name = node_config.get("name", "[No Name]")
    description = node_config.get("description", "[No Description]")
    keywords = ", ".join(node_config.get("keywords", [])) or "[None]"
    scope = node_config.get("scope", "[None]")

    return f"""
You are an expert binary classifier. You are classifying an employee comment from a post-training survey.
The goal is to determine if a specific target topic — {name} — appears as one of the **main topics** in the comment.

**Topic**: {name}
**Description**: {description}
**Keywords**: {keywords}
**Scope**: {scope}

# Key Rules
- Use only the information **explicitly present** in the comment and the topic definition above.
- DO NOT infer intent, sentiment, or unstated implications.
- DO NOT guess based on general knowledge, stereotypes, or assumptions.
- A "main topic" means the comment directly and substantially discusses it, not just a brief or indirect mention.
- If evidence is insufficient, classify as False.

# Output
Return a JSON object with the following fields:
- **is_relevant**: True or False
- **reasoning**: One or two sentences. It must:
    * Directly reference or quote the relevant part(s) of the comment.
    * Only use explicit evidence from the text and topic definition.
    * Avoid general statements not grounded in the comment.
- **confidence**: Integer 1–5, where:
    1 = Very uncertain, weak or no evidence
    2 = Low confidence, weak evidence
    3 = Moderate confidence, ambiguous or partial evidence
    4 = High confidence, clear direct evidence
    5 = Very certain, strong direct evidence
- **excerpt**: Exact span of the comment that supports the classification if True, otherwise empty string.
"""
