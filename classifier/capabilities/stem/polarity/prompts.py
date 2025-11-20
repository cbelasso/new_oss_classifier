from typing import Dict, List


def stem_polarity_prompt(
    text: str, stem_path: str, stem_definitions: List[Dict[str, str]] = None
) -> str:
    """
    Generate a prompt to evaluate the polarity/sentiment of a comment towards a specific classification stem.

    Args:
        text: The original text being analyzed
        stem_path: The classification path/stem (e.g., "Teaching Effectiveness>Teaching Style>Explanations")
        stem_definitions: Optional list of definition dicts for each node in the stem path
    """

    # Format definitions section if provided
    definitions_section = ""
    if stem_definitions:
        definitions_section = "\n**Topic Path Definitions:**\n\n"
        for node_info in stem_definitions:
            definitions_section += f"**{node_info['name']}**\n"
            if node_info.get("definition"):
                definitions_section += f"- Definition: {node_info['definition']}\n"
            if node_info.get("description"):
                definitions_section += f"- Description: {node_info['description']}\n"
            if node_info.get("keywords"):
                keywords_str = ", ".join(node_info["keywords"][:10])
                definitions_section += f"- Keywords: {keywords_str}\n"
            definitions_section += "\n"

    return f"""
You are an expert text analyzer. You are analyzing an employee comment from a post-training survey that has been classified under a specific topic path.

**Comment**: {text}

**Topic Path**: {stem_path}

{definitions_section}---

# Your Task
Determine the overall polarity (sentiment) of the comment specifically toward this topic path. The comment has already been classified as relevant to this topic. Now, identify whether the author's stance toward this specific topic is positive, negative, neutral, or mixed.

Use the topic definitions above to understand what each level of the path represents and ensure your polarity assessment is aligned with the actual meaning of the topic.

---

# Polarity Types

| Polarity | Meaning | Examples |
|----------|---------|----------|
| **Positive** | Expresses satisfaction, praise, approval, or positive sentiment toward the topic | "Great explanations", "Very helpful", "Loved this aspect", "Excellent approach" |
| **Negative** | Expresses dissatisfaction, criticism, disapproval, or negative sentiment toward the topic | "Poor explanations", "Confusing", "Needs improvement", "Didn't like this" |
| **Neutral** | Factual observation or description without clear positive or negative sentiment | "The class covered this topic", "Explanations were provided", "This was included" |
| **Mixed** | Contains both positive and negative sentiments toward the same topic | "Good examples but too fast", "Clear but too technical", "Helpful but needs more" |

---

# Key Rules
- Assess polarity **only for this specific topic path**, not the overall comment
- Distinguish between constructive criticism (Negative) and balanced feedback (Mixed)
- If the comment mentions both strengths and weaknesses about the topic, classify as Mixed
- Pure suggestions for improvement without negative sentiment should be Negative (constructive criticism)
- Expressions of gratitude or appreciation are Positive even if brief
- If the comment is purely descriptive without evaluative language, classify as Neutral
- Provide a confidence score from 1 (low) to 5 (high) for your assessment
- Extract the specific excerpt that best supports your polarity determination

---

# Output Format
Return a JSON object with:
- **has_polarity**: Boolean (true if polarity can be determined, false otherwise)
- **polarity_result**: Object containing (only if has_polarity is true):
    * **polarity**: One of: "Positive", "Negative", "Neutral", "Mixed"
    * **confidence**: Integer from 1 to 5 indicating confidence in the assessment
    * **reasoning**: One sentence explaining why this polarity was assigned to this topic
    * **excerpt**: The text span from the comment that best indicates this polarity

---

# Examples

**Example 1:**
Comment: "The instructor provided excellent explanations that made complex topics easy to understand."
Topic Path: "Teaching Effectiveness>Teaching Style>Explanations"
Output:
{{
  "has_polarity": true,
  "polarity_result": {{
    "polarity": "Positive",
    "confidence": 5,
    "reasoning": "The comment expresses clear praise for the explanations, describing them as 'excellent' and effective at making topics 'easy to understand'.",
    "excerpt": "excellent explanations that made complex topics easy to understand"
  }}
}}

---

**Example 2:**
Comment: "The explanations were confusing and didn't help me understand the material at all."
Topic Path: "Teaching Effectiveness>Teaching Style>Explanations"
Output:
{{
  "has_polarity": true,
  "polarity_result": {{
    "polarity": "Negative",
    "confidence": 5,
    "reasoning": "The comment expresses clear dissatisfaction with the explanations, describing them as 'confusing' and ineffective.",
    "excerpt": "explanations were confusing and didn't help me understand the material"
  }}
}}

---

**Example 3:**
Comment: "The explanations were clear and well-structured, but they moved too quickly for me to keep up."
Topic Path: "Teaching Effectiveness>Teaching Style>Explanations"
Output:
{{
  "has_polarity": true,
  "polarity_result": {{
    "polarity": "Mixed",
    "confidence": 4,
    "reasoning": "The comment contains both positive sentiment ('clear and well-structured') and negative sentiment ('moved too quickly') about the explanations.",
    "excerpt": "explanations were clear and well-structured, but they moved too quickly"
  }}
}}

---

**Example 4:**
Comment: "The course covered various teaching methods."
Topic Path: "Teaching Effectiveness>Teaching Style"
Output:
{{
  "has_polarity": true,
  "polarity_result": {{
    "polarity": "Neutral",
    "confidence": 5,
    "reasoning": "The comment is purely descriptive without any evaluative or emotional language regarding the teaching style.",
    "excerpt": "The course covered various teaching methods"
  }}
}}

---

**Example 5:**
Comment: "The professor should provide more examples to help clarify concepts."
Topic Path: "Teaching Effectiveness>Teaching Style>Explanations"
Output:
{{
  "has_polarity": true,
  "polarity_result": {{
    "polarity": "Negative",
    "confidence": 4,
    "reasoning": "The suggestion for improvement implies current explanations are insufficient, indicating constructive criticism (negative sentiment).",
    "excerpt": "should provide more examples to help clarify concepts"
  }}
}}
"""
