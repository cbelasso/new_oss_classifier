from typing import Dict, List


def stem_trends_prompt(
    text: str, stem_path: str, stem_definitions: List[Dict[str, str]] = None
) -> str:
    """
    Generate a prompt to evaluate what types of trends apply to a specific classification stem.

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
                keywords_str = ", ".join(
                    node_info["keywords"][:10]
                )  # Limit to first 10 keywords
                definitions_section += f"- Keywords: {keywords_str}\n"
            definitions_section += "\n"

    return f"""
You are an expert text analyzer. You are analyzing an employee comment from a post-training survey that has been classified under a specific topic path.

**Comment**: {text}

**Topic Path**: {stem_path}

{definitions_section}---

# Your Task
Determine which trends apply to this specific topic path based on the comment. The comment has already been classified as relevant to this topic. Now, identify what temporal changes or patterns (if any) the author is describing about this specific topic.

Use the topic definitions above to understand what each level of the path represents and ensure your trend analysis is aligned with the actual meaning of the topic.

---

# What Counts as a Trend
A trend for this topic path is a statement that indicates:
- A temporal comparison between past and present states **of this topic**
- An observation of change (or consistency) over time **regarding this topic**
- An evolution or trajectory **of some aspect of this topic**

**Trends require temporal markers** such as:
- Explicit time references: "last year", "in 2023", "this quarter", "recently"
- Comparative language: "used to be", "no longer", "has become", "still"
- Change verbs: "improved", "declined", "increased", "decreased", "remains"
- Sequential language: "now", "anymore", "these days", "currently"

---

# Direction Types

| Direction | Meaning | Example |
|-----------|---------|---------|
| **increasing** | Growing, rising, or expanding | "Workload has increased" |
| **decreasing** | Shrinking, falling, or contracting | "Support has decreased" |
| **improving** | Getting better in quality/effectiveness | "Teaching has improved" |
| **deteriorating** | Getting worse in quality/effectiveness | "Quality has worsened" |
| **stable_positive** | Something good remains consistent | "Still excellent as always" |
| **stable_negative** | Something bad remains consistent | "Still inadequate" |
| **fluctuating** | Varying or inconsistent over time | "Sometimes good, sometimes not" |

---

# Valence Types

| Valence | Meaning |
|---------|---------|
| **positive** | The trend is favorable, beneficial, or desirable |
| **negative** | The trend is unfavorable, problematic, or undesirable |
| **neutral** | The trend is neither clearly good nor bad |
| **mixed** | The trend has both positive and negative aspects |

**Important**: Valence should be determined from the **commenter's perspective** as expressed in the text.

---

# Confidence Scoring

For each trend, assign a **confidence score** from 1 to 5:

| Score | Meaning | When to Use |
|-------|---------|-------------|
| **5** | Very certain | Explicit temporal markers, clear change, unambiguous direction, clearly relevant to topic |
| **4** | Quite certain | Strong temporal indicators, clear trend with minor ambiguity about topic relevance |
| **3** | Moderately certain | Temporal markers present but trend or topic relevance requires interpretation |
| **2** | Somewhat uncertain | Weak temporal markers or ambiguous direction/valence or topic relevance |
| **1** | Very uncertain | Minimal temporal indicators, highly ambiguous interpretation or topic relevance |

**Confidence Guidelines**:
- Higher confidence for explicit time references ("since 2023", "compared to last year")
- Higher confidence when trend is clearly about this specific topic path
- Lower confidence for vague temporal markers ("recently", "these days")
- Lower confidence when it's unclear if trend applies to this specific topic vs. related topics
- Higher confidence when direction and valence are both clear

---

# Key Rules
- A comment may contain **multiple trends** for the same topic path
- Only identify trends that are **explicitly stated** in the comment with temporal markers
- Each trend must have a corresponding excerpt that supports it
- The excerpt must be directly relevant to the topic path
- Focus on trends **about this specific topic**, not other topics
- If no trends are present for this topic, return has_trends: false
- DO NOT infer trends from static descriptions without temporal comparison

---

# Output Format
Return a JSON object with:
- **has_trends**: Boolean (true if at least one trend applies, false otherwise)
- **trends**: List of objects, each containing:
    * **excerpt**: The exact text span from the comment containing the trend.
    * **reasoning**: One sentence explaining why this qualifies as a trend for this topic path.
    * **subject**: What aspect of the topic is changing (keep concise and aligned with the topic path).
    * **direction**: One of: "increasing", "decreasing", "improving", "deteriorating", "stable_positive", "stable_negative", "fluctuating"
    * **valence**: One of: "positive", "negative", "neutral", "mixed"
    * **confidence**: Integer from 1 to 5

---

# Examples

**Example 1:**
Comment: "The instructor's explanations used to be clear, but have become more confusing recently."
Topic Path: "Teaching Effectiveness>Teaching Style>Explanations"
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "The instructor's explanations used to be clear, but have become more confusing recently",
      "reasoning": "Indicates a decline in explanation clarity over time, directly relevant to the Explanations topic.",
      "subject": "explanation clarity",
      "direction": "deteriorating",
      "valence": "negative",
      "confidence": 5
    }}
  ]
}}

---

**Example 2:**
Comment: "The course content was excellent and very relevant to my work."
Topic Path: "Course Content>Relevance"
Output:
{{
  "has_trends": false,
  "trends": []
}}

---

**Example 3:**
Comment: "The amount of hands-on practice has increased significantly compared to last year, which is great. However, the quality of feedback on those exercises has declined."
Topic Path: "Teaching Effectiveness>Practical Application"
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "The amount of hands-on practice has increased significantly compared to last year",
      "reasoning": "Shows an increase in the quantity of practical application opportunities with explicit temporal reference.",
      "subject": "hands-on practice quantity",
      "direction": "increasing",
      "valence": "positive",
      "confidence": 5
    }},
    {{
      "excerpt": "the quality of feedback on those exercises has declined",
      "reasoning": "Indicates deterioration in feedback quality for practical exercises over time.",
      "subject": "feedback quality",
      "direction": "deteriorating",
      "valence": "negative",
      "confidence": 4
    }}
  ]
}}

---

**Example 4:**
Comment: "The instructor remains consistently engaging and motivating, just like in previous sessions I attended."
Topic Path: "Teaching Effectiveness>Teaching Style"
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "The instructor remains consistently engaging and motivating, just like in previous sessions I attended",
      "reasoning": "Indicates stable positive quality in teaching style with explicit temporal comparison to past sessions.",
      "subject": "instructor engagement",
      "direction": "stable_positive",
      "valence": "positive",
      "confidence": 5
    }}
  ]
}}

---

**Example 5:**
Comment: "Support from the teaching assistants has been really hit or miss lately - sometimes very helpful, sometimes unavailable."
Topic Path: "Course Support>Teaching Assistant Support"
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "Support from the teaching assistants has been really hit or miss lately - sometimes very helpful, sometimes unavailable",
      "reasoning": "Describes inconsistent quality of TA support over recent time period, directly relevant to the Teaching Assistant Support topic.",
      "subject": "TA support consistency",
      "direction": "fluctuating",
      "valence": "mixed",
      "confidence": 4
    }}
  ]
}}

---

**Example 6:**
Comment: "The workload has increased dramatically this semester, making it hard to keep up. Meanwhile, the teaching quality has remained excellent throughout."
Topic Path: "Teaching Effectiveness>Teaching Style"
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "the teaching quality has remained excellent throughout",
      "reasoning": "Indicates consistent high quality in teaching style over the semester period, directly relevant to Teaching Style.",
      "subject": "teaching quality",
      "direction": "stable_positive",
      "valence": "positive",
      "confidence": 5
    }}
  ]
}}

Note: The workload trend is not included because it's not relevant to the "Teaching Effectiveness>Teaching Style" topic path.

---

**Example 7:**
Comment: "Things have gotten somewhat better recently, I think."
Topic Path: "Course Content>Organization"
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "Things have gotten somewhat better recently",
      "reasoning": "Suggests improvement over time, though the specifics and certainty are vague.",
      "subject": "course organization",
      "direction": "improving",
      "valence": "positive",
      "confidence": 2
    }}
  ]
}}
"""
