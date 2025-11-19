from typing import Any, Dict, List


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


def bundled_classification_prompt(nodes: List[Dict[str, Any]]) -> str:
    """
    Generate a bundled classification prompt for multiple sibling nodes.

    This prompt evaluates multiple topics simultaneously, allowing the LLM
    to see relationships between adjacent topics for richer context and
    potentially better comparative evaluation.

    Args:
        nodes: List of node configurations, each containing:
            - name: Topic name
            - description: Detailed topic description
            - keywords: List of relevant keywords
            - scope: Scope definition

    Returns:
        Formatted prompt string for the LLM
    """
    # Build topic definitions table
    topics_section = ""
    for i, node in enumerate(nodes, 1):
        name = node.get("name", "[No Name]")
        description = node.get("description", "[No Description]")
        keywords = ", ".join(node.get("keywords", [])) or "[None]"
        scope = node.get("scope", "[None]")

        topics_section += f"""
**Topic {i}: {name}**
- **Description**: {description}
- **Keywords**: {keywords}
- **Scope**: {scope}
"""

    return f"""
You are an expert multi-topic classifier. You are classifying an employee comment from a post-training survey.
The goal is to determine which of the following topics appear as **main topics** in the comment.

You will evaluate **{len(nodes)} topics simultaneously**. This allows you to compare and contrast the topics to make more informed decisions about relevance.

---

# Topics to Evaluate
{topics_section}

---

# Key Rules
- Evaluate **each topic independently** — a comment can be relevant to multiple topics, one topic, or none.
- Use only the information **explicitly present** in the comment and the topic definitions above.
- DO NOT infer intent, sentiment, or unstated implications.
- DO NOT guess based on general knowledge, stereotypes, or assumptions.
- A "main topic" means the comment directly and substantially discusses it, not just a brief or indirect mention.
- If evidence is insufficient for a topic, classify it as False.
- **Compare topics**: Use the context of seeing multiple related topics to make better distinctions about which ones truly apply.

---

# Output Format
Return a JSON object with:
- **node_results**: A dictionary where each key is a topic name and each value is an object containing:
    * **is_relevant**: True or False
    * **reasoning**: One or two sentences that:
        - Directly reference or quote the relevant part(s) of the comment
        - Only use explicit evidence from the text and topic definition
        - Avoid general statements not grounded in the comment
    * **confidence**: Integer 1–5, where:
        1 = Very uncertain, weak or no evidence
        2 = Low confidence, weak evidence
        3 = Moderate confidence, ambiguous or partial evidence
        4 = High confidence, clear direct evidence
        5 = Very certain, strong direct evidence
    * **excerpt**: Exact span of the comment that supports the classification if True, otherwise empty string

---

# Example 1: Multiple Topics Relevant

**Comment**: "The instructor's teaching style was excellent - clear explanations and engaging delivery. The course content was also very relevant to my job."

**Topics**:
1. Teaching Style
2. Course Content  
3. Assessment Methods

**Output**:
{{
  "node_results": {{
    "Teaching Style": {{
      "is_relevant": true,
      "confidence": 5,
      "reasoning": "The comment explicitly praises 'teaching style' with specific details about 'clear explanations and engaging delivery'.",
      "excerpt": "The instructor's teaching style was excellent - clear explanations and engaging delivery"
    }},
    "Course Content": {{
      "is_relevant": true,
      "confidence": 5,
      "reasoning": "The comment directly states 'course content was also very relevant to my job', explicitly mentioning content relevance.",
      "excerpt": "The course content was also very relevant to my job"
    }},
    "Assessment Methods": {{
      "is_relevant": false,
      "confidence": 5,
      "reasoning": "No mention of assessments, tests, evaluations, or grading in the comment.",
      "excerpt": ""
    }}
  }}
}}

---

# Example 2: Distinguishing Similar Topics

**Comment**: "The hands-on exercises were great, but the lecture portions were too long and boring."

**Topics**:
1. Practical Activities
2. Lecture Format
3. Course Duration

**Output**:
{{
  "node_results": {{
    "Practical Activities": {{
      "is_relevant": true,
      "confidence": 5,
      "reasoning": "The comment explicitly mentions 'hands-on exercises' which directly corresponds to practical activities.",
      "excerpt": "The hands-on exercises were great"
    }},
    "Lecture Format": {{
      "is_relevant": true,
      "confidence": 5,
      "reasoning": "The comment specifically discusses 'lecture portions' and their quality, directly addressing lecture format.",
      "excerpt": "the lecture portions were too long and boring"
    }},
    "Course Duration": {{
      "is_relevant": false,
      "confidence": 3,
      "reasoning": "While 'too long' is mentioned, it refers to lecture portions specifically, not overall course duration.",
      "excerpt": ""
    }}
  }}
}}

---

# Example 3: No Topics Relevant

**Comment**: "The coffee was cold."

**Topics**:
1. Teaching Effectiveness
2. Course Materials
3. Learning Outcomes

**Output**:
{{
  "node_results": {{
    "Teaching Effectiveness": {{
      "is_relevant": false,
      "confidence": 5,
      "reasoning": "The comment only discusses coffee temperature with no mention of teaching or instruction.",
      "excerpt": ""
    }},
    "Course Materials": {{
      "is_relevant": false,
      "confidence": 5,
      "reasoning": "No mention of course materials, resources, handouts, or learning content.",
      "excerpt": ""
    }},
    "Learning Outcomes": {{
      "is_relevant": false,
      "confidence": 5,
      "reasoning": "No discussion of learning, skills acquired, or educational outcomes.",
      "excerpt": ""
    }}
  }}
}}

---

Now evaluate the comment for all {len(nodes)} topics and return your response in the exact JSON format shown above.
"""
