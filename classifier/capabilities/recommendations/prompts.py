def recommendations_detection_prompt(text: str) -> str:
    """
    Generate a prompt to detect, extract, paraphrase, and qualify recommendations from text.

    This version extends the prior version by adding a 'qualifier' field that
    categorizes the type or direction of the recommendation (e.g., add, reduce, modify, maintain, etc.).
    """
    return f"""
You are an expert text analyzer. You are analyzing an employee comment from a post-training survey.
The goal is to identify whether the comment contains **recommendations** — actionable suggestions, advice, proposals, or requests for change that the author is making — and to determine the **type of recommendation** being made.

**Comment**: {text}

---

# What Counts as a Recommendation
A recommendation is:
- An actionable suggestion for improvement, change, or future action
- Advice on what should or could be done differently
- A proposal for a specific approach, policy, or practice
- A request for something to be added, removed, or modified
- Constructive feedback that implies "we/you/they should do X"

Examples of recommendations:
- "The training should include more hands-on exercises"
- "I suggest we cover advanced topics in a separate session"
- "It would be helpful to provide reference materials afterward"
- "Consider adding more time for Q&A"

---

# What Does NOT Count as a Recommendation
- General positive/negative statements without actionable content ("It was great", "I didn't like it")
- Pure descriptions of current state without suggesting change
- Questions without implied suggestions
- Purely personal preferences without actionable proposals

---

# Key Rules
- Use only the information **explicitly present** in the comment.
- DO NOT infer unstated recommendations or read between the lines.
- DO NOT count vague dissatisfaction as a recommendation unless it includes a clear actionable suggestion.
- Identify **all distinct** recommendations in the comment.
- Each recommendation must correspond to a specific identifiable excerpt.
- If multiple recommendations appear in one sentence, extract them as separate items if they suggest different actions.
- If the comment is purely descriptive, observational, or expresses sentiment without actionable content, return has_recommendations: false.

---

# Qualifier Types (Direction of Change)

For each recommendation, assign one **qualifier** from the list below based on the intended action:

| Qualifier | Meaning | Example Phrases | Example |
|------------|----------|----------------|----------|
| **add_or_increase** | Suggests adding or increasing something | "Add", "Include", "More", "Increase" | "Add more examples" |
| **reduce_or_remove** | Suggests reducing or removing something | "Reduce", "Fewer", "Too much", "Cut down" | "Reduce lecture time" |
| **introduce_or_start** | Suggests starting something new | "Start", "Implement", "Offer", "Provide" | "Start offering online sessions" |
| **eliminate_or_stop** | Suggests stopping or discontinuing something | "Stop", "Eliminate", "Don't" | "Stop using group exercises" |
| **modify_or_improve** | Suggests changing or improving something existing | "Make it more", "Change", "Improve", "Adjust" | "Make the slides clearer" |
| **maintain_or_continue** | Suggests keeping something as is or reinforcing it | "Keep", "Continue", "Maintain" | "Keep the interactive activities" |
| **unspecified_or_general** | Suggests improvement without clear direction | "Should be better", "Needs improvement" | "The course should improve overall" |

---

# Output Format
Return a JSON object with:
- **has_recommendations**: Boolean (true if at least one recommendation is present, false otherwise)
- **recommendations**: List of objects, each containing:
    * **excerpt**: The exact text span from the comment containing the recommendation.
    * **reasoning**: One sentence explaining why this qualifies as a recommendation and what action is being suggested.
    * **paraphrased_recommendation**: A clear, concise restatement of the recommendation in natural language.
    * **qualifier**: One of the following literals:
      "add_or_increase", "reduce_or_remove", "introduce_or_start",
      "eliminate_or_stop", "modify_or_improve", "maintain_or_continue",
      or "unspecified_or_general".

---

# Examples

**Example 1:**
Comment: "The session was too short. We need more time to practice the concepts."
Output:
{{
  "has_recommendations": true,
  "recommendations": [
    {{
      "excerpt": "We need more time to practice the concepts",
      "reasoning": "Suggests extending the duration of training to allow for more practice.",
      "paraphrased_recommendation": "Extend the training sessions to provide additional practice time.",
      "qualifier": "add_or_increase"
    }}
  ]
}}

---

**Example 2:**
Comment: "I really enjoyed the training. The instructor was excellent."
Output:
{{
  "has_recommendations": false,
  "recommendations": []
}}

---

**Example 3:**
Comment: "The material was confusing and moved too quickly. It would help to slow down and add more examples, especially for the advanced topics."
Output:
{{
  "has_recommendations": true,
  "recommendations": [
    {{
      "excerpt": "It would help to slow down",
      "reasoning": "Recommends reducing the pace of content delivery.",
      "paraphrased_recommendation": "Deliver the material at a slower pace to improve comprehension.",
      "qualifier": "reduce_or_remove"
    }},
    {{
      "excerpt": "add more examples, especially for the advanced topics",
      "reasoning": "Suggests adding examples to enhance understanding, particularly for complex sections.",
      "paraphrased_recommendation": "Include more examples, especially for advanced topics.",
      "qualifier": "add_or_increase"
    }}
  ]
}}

---

**Example 4:**
Comment: "We should have more breakout sessions for group work. Also, it would be great to get the slides beforehand, and maybe include a follow-up session next month."
Output:
{{
  "has_recommendations": true,
  "recommendations": [
    {{
      "excerpt": "We should have more breakout sessions for group work",
      "reasoning": "Recommends increasing opportunities for collaborative learning.",
      "paraphrased_recommendation": "Add more breakout sessions to encourage group collaboration.",
      "qualifier": "add_or_increase"
    }},
    {{
      "excerpt": "it would be great to get the slides beforehand",
      "reasoning": "Suggests providing learning materials before the training begins.",
      "paraphrased_recommendation": "Distribute slides in advance of the session.",
      "qualifier": "introduce_or_start"
    }},
    {{
      "excerpt": "include a follow-up session next month",
      "reasoning": "Proposes scheduling a continuation session for reinforcement.",
      "paraphrased_recommendation": "Hold a follow-up session next month for continued learning.",
      "qualifier": "introduce_or_start"
    }}
  ]
}}
"""
