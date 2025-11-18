def trends_detection_prompt(text: str) -> str:
    """
    Generate a prompt to detect, extract, and classify trends in feedback text.

    A trend represents a temporal comparison indicating change (or stability) over time,
    with both a direction component and a valence (positive/negative sentiment).
    """
    return f"""
You are an expert text analyzer. You are analyzing an employee comment from a post-training survey.
The goal is to identify whether the comment contains **trends** — statements that indicate change or stability over time — and to characterize both the **direction** and **valence** (positive/negative nature) of each trend.

**Comment**: {text}

---

# What Counts as a Trend
A trend is a statement that indicates:
- A temporal comparison between past and present states
- An observation of change (or consistency) over time
- An evolution or trajectory of some aspect

**Trends require temporal markers** such as:
- Explicit time references: "last year", "in 2023", "this quarter", "recently"
- Comparative language: "used to be", "no longer", "has become", "still"
- Change verbs: "improved", "declined", "increased", "decreased", "remains"
- Sequential language: "now", "anymore", "these days", "currently"

Examples of trends:
- "My manager used to be caring, but not anymore" (declining)
- "The quality has improved significantly since last year" (improving)
- "Training remains excellent as always" (stable/positive)
- "Support has been inconsistent lately" (fluctuating)

---

# What Does NOT Count as a Trend
- **Static descriptions** without temporal comparison: "The training is good"
- **Comparisons to expectations** (not temporal): "This was worse than I expected"
- **Recommendations for future**: "We should improve this" (use recommendations detection instead)
- **One-time events**: "The last session was cancelled" (no pattern/trajectory)

---

# Key Rules
- Use only the information **explicitly present** in the comment.
- A trend **requires** at least a moderate temporal marker (explicit or clearly implicit).
- DO NOT infer trends from purely static statements.
- Identify **all distinct** trends in the comment.
- Each trend must correspond to a specific identifiable excerpt.
- If multiple trends appear, extract them as separate items if they describe different aspects or directions.

---

# Direction Types

For each trend, assign one **direction** from the list below:

| Direction | Meaning | Example Temporal Markers | Example |
|-----------|---------|-------------------------|---------|
| **increasing** | Something is growing, rising, or expanding | "has increased", "more than before", "growing" | "Workload has increased significantly" |
| **decreasing** | Something is shrinking, falling, or contracting | "has decreased", "less than before", "declining" | "Support has declined over time" |
| **improving** | Something is getting better in quality/effectiveness | "has improved", "better than", "getting better" | "Communication has improved recently" |
| **deteriorating** | Something is getting worse in quality/effectiveness | "has worsened", "worse than", "getting worse" | "Morale has deteriorated since the changes" |
| **stable_positive** | Something good remains consistent | "still excellent", "remains strong", "consistently good" | "Quality remains excellent" |
| **stable_negative** | Something bad remains consistent | "still poor", "remains weak", "consistently bad" | "Response time is still slow" |
| **fluctuating** | Something varies or is inconsistent over time | "inconsistent", "varies", "sometimes...sometimes" | "Feedback has been hit or miss lately" |

---

# Valence Types

For each trend, assign one **valence** indicating whether the change/state is viewed positively or negatively:

| Valence | Meaning | When to Use |
|---------|---------|-------------|
| **positive** | The trend is favorable, beneficial, or desirable | Improvements, positive stability, decrease in negative things |
| **negative** | The trend is unfavorable, problematic, or undesirable | Deterioration, negative stability, increase in negative things |
| **neutral** | The trend is neither clearly good nor bad | Factual observations without evaluative content |
| **mixed** | The trend has both positive and negative aspects | Fluctuating quality, or context suggests ambivalence |

**Important**: Valence should be determined from the **commenter's perspective** as expressed in the text.

---

# Confidence Scoring

For each trend, assign a **confidence score** from 1 to 5:

| Score | Meaning | When to Use |
|-------|---------|-------------|
| **5** | Very certain | Explicit temporal markers, clear change, unambiguous direction |
| **4** | Quite certain | Strong temporal indicators, clear trend with minor ambiguity |
| **3** | Moderately certain | Temporal markers present but trend requires some interpretation |
| **2** | Somewhat uncertain | Weak temporal markers or ambiguous direction/valence |
| **1** | Very uncertain | Minimal temporal indicators, highly ambiguous interpretation |

**Confidence Guidelines**:
- Higher confidence for explicit time references ("since 2023", "compared to last year")
- Lower confidence for vague temporal markers ("recently", "these days")
- Higher confidence when direction and valence are both clear
- Lower confidence when trend is implied rather than stated

---

# Output Format
Return a JSON object with:
- **has_trends**: Boolean (true if at least one trend is present, false otherwise)
- **trends**: List of objects, each containing:
    * **excerpt**: The exact text span from the comment containing the trend.
    * **reasoning**: One sentence explaining why this qualifies as a trend and what is changing.
    * **subject**: What aspect is changing (e.g., "manager support", "training quality", "workload"). Keep this concise.
    * **direction**: One of: "increasing", "decreasing", "improving", "deteriorating", "stable_positive", "stable_negative", "fluctuating"
    * **valence**: One of: "positive", "negative", "neutral", "mixed"
    * **confidence**: Integer from 1 to 5

---

# Examples

**Example 1:**
Comment: "My manager used to be caring, but not anymore."
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "My manager used to be caring, but not anymore",
      "reasoning": "Indicates a decline in manager's caring behavior over time.",
      "subject": "manager caring behavior",
      "direction": "deteriorating",
      "valence": "negative",
      "confidence": 5
    }}
  ]
}}

---

**Example 2:**
Comment: "The training quality has improved significantly since last year."
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "The training quality has improved significantly since last year",
      "reasoning": "Shows an improvement in training quality with explicit temporal reference.",
      "subject": "training quality",
      "direction": "improving",
      "valence": "positive",
      "confidence": 5
    }}
  ]
}}

---

**Example 3:**
Comment: "The workload has increased dramatically, and bureaucracy has decreased."
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "The workload has increased dramatically",
      "reasoning": "Indicates an increase in workload over time.",
      "subject": "workload",
      "direction": "increasing",
      "valence": "negative",
      "confidence": 4
    }},
    {{
      "excerpt": "bureaucracy has decreased",
      "reasoning": "Shows a reduction in bureaucracy over time.",
      "subject": "bureaucracy",
      "direction": "decreasing",
      "valence": "positive",
      "confidence": 4
    }}
  ]
}}

---

**Example 4:**
Comment: "The instructor remains excellent, just like in previous sessions."
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "The instructor remains excellent, just like in previous sessions",
      "reasoning": "Indicates consistent high quality over time with explicit temporal comparison.",
      "subject": "instructor quality",
      "direction": "stable_positive",
      "valence": "positive",
      "confidence": 5
    }}
  ]
}}

---

**Example 5:**
Comment: "Support has been really inconsistent lately - sometimes responsive, sometimes not."
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "Support has been really inconsistent lately - sometimes responsive, sometimes not",
      "reasoning": "Describes varying quality of support over recent time period.",
      "subject": "support responsiveness",
      "direction": "fluctuating",
      "valence": "mixed",
      "confidence": 4
    }}
  ]
}}

---

**Example 6:**
Comment: "The training was excellent and very helpful."
Output:
{{
  "has_trends": false,
  "trends": []
}}

---

**Example 7:**
Comment: "Things seem to be getting better, I think."
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "Things seem to be getting better",
      "reasoning": "Suggests improvement over time, though the subject and certainty are vague.",
      "subject": "general situation",
      "direction": "improving",
      "valence": "positive",
      "confidence": 2
    }}
  ]
}}

---

Now analyze the comment and return the JSON output.
"""
