from typing import Dict, List


def stem_recommendations_prompt(
    text: str, stem_path: str, stem_definitions: List[Dict[str, str]] = None
) -> str:
    """
    Generate a prompt to evaluate what types of recommendations apply to a specific classification stem.

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
Determine which types of recommendations apply to this specific topic path based on the comment. The comment has already been classified as relevant to this topic. Now, identify what kind of recommendation (if any) the author is making about this specific topic.

Use the topic definitions above to understand what each level of the path represents and ensure your recommendations are aligned with the actual meaning of the topic.

---

# Recommendation Types

Evaluate whether the comment suggests any of the following actions regarding the topic:

| Type | Meaning | Example Phrases |
|------|---------|----------------|
| **start** | Begin doing something new that isn't currently done | "Should start", "Need to begin", "Introduce" |
| **stop** | Cease or discontinue something currently being done | "Should stop", "Eliminate", "Don't do", "Remove" |
| **do_more** | Increase or expand something that's already being done | "More of", "Increase", "Expand", "Add more" |
| **do_less** | Reduce or decrease something that's currently being done | "Less of", "Reduce", "Too much", "Cut back" |
| **continue** | Keep doing something that's working well | "Keep doing", "Continue", "Maintain", "Don't change" |
| **change** | Modify or alter how something is currently done | "Change how", "Do differently", "Improve", "Adjust" |

---

# Key Rules
- A comment may contain **multiple recommendation types** for the same topic path
- Only identify recommendations that are **explicitly stated** in the comment
- Each recommendation type must have a corresponding excerpt that supports it
- The excerpt must be directly relevant to the topic path
- Focus on what is being recommended **for this specific topic**, not other topics
- If no recommendations are present for this topic, return has_recommendations: false

---

# Output Format
Return a JSON object with:
- **has_recommendations**: Boolean (true if at least one recommendation type applies, false otherwise)
- **recommendations**: List of objects, each containing:
    * **recommendation_type**: One of: "start", "stop", "do_more", "do_less", "continue", "change"
    * **excerpt**: The exact text span from the comment that indicates this recommendation type
    * **reasoning**: One sentence explaining why this recommendation type applies to this topic path
    * **paraphrased_recommendation**: A clear, concise restatement of the recommendation in natural language.

---

# Examples

**Example 1:**
Comment: "The instructor should provide more real-world examples during lectures."
Topic Path: "Teaching Effectiveness>Teaching Style"
Output:
{{
  "has_recommendations": true,
  "recommendations": [
    {{
      "recommendation_type": "do_more",
      "excerpt": "should provide more real-world examples",
      "reasoning": "Suggests increasing the use of real-world examples in the teaching approach.",
      "paraphrased_recommendation": "Increase the use of real-world examples in lectures."
    }}
  ]
}}

---

**Example 2:**
Comment: "The training was excellent and very helpful."
Topic Path: "Course Content>Relevance"
Output:
{{
  "has_recommendations": false,
  "recommendations": []
}}

---

**Example 3:**
Comment: "We should stop using PowerPoint slides and start incorporating more interactive activities. The current lecture format needs to be more engaging."
Topic Path: "Teaching Effectiveness>Teaching Style"
Output:
{{
  "has_recommendations": true,
  "recommendations": [
    {{
      "recommendation_type": "stop",
      "excerpt": "should stop using PowerPoint slides",
      "reasoning": "Recommends discontinuing the use of PowerPoint slides in teaching.",
      "paraphrased_recommendation": "Discontinue the use of PowerPoint slides in teaching."
    }},
    {{
      "recommendation_type": "start",
      "excerpt": "start incorporating more interactive activities",
      "reasoning": "Suggests beginning to use interactive activities that aren't currently part of the teaching style.",
      "paraphrased_recommendation": "Begin incorporating interactive activities into the teaching approach."
    }},
    {{
      "recommendation_type": "change",
      "excerpt": "current lecture format needs to be more engaging",
      "reasoning": "Indicates the teaching approach should be modified to increase engagement.",
      "paraphrased_recommendation": "Modify the lecture format to make it more engaging."
    }}
  ]
}}

---

**Example 4:**
Comment: "Keep the hands-on exercises exactly as they are - they're perfect and really help with learning."
Topic Path: "Teaching Effectiveness>Practical Application"
Output:
{{
  "has_recommendations": true,
  "recommendations": [
    {{
      "recommendation_type": "continue",
      "excerpt": "Keep the hands-on exercises exactly as they are",
      "reasoning": "Recommends maintaining the current approach to hands-on exercises without changes.",
      "paraphrased_recommendation": "Maintain the current hands-on exercises without modification."
    }}
  ]
}}
"""
