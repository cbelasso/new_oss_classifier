def alert_detection_prompt(text: str) -> str:
    """
    Generate a prompt to detect and categorize alerts from employee comments.

    Alerts are serious concerns, complaints, or reports of problematic situations
    that require attention from HR, management, or compliance teams.
    """
    return f"""
You are an expert text analyzer specializing in workplace safety and compliance. You are analyzing an employee comment from a post-training survey.
The goal is to identify whether the comment contains **alerts** — serious concerns, complaints, or reports of problematic workplace situations that require immediate attention or follow-up from HR, management, or compliance teams.

**Comment**: {text}

---

# What Counts as an Alert
An alert is:
- A report or complaint about discrimination, harassment, or hostile behavior
- A concern about safety, violence, or threatening situations
- A description of ethical violations, fraud, or policy breaches
- A disclosure of mental health crises or substance abuse issues
- A report of retaliation for protected activities
- Any serious workplace concern that could expose the organization to legal, safety, or reputational risk

Examples of alerts:
- "My manager makes inappropriate comments about my appearance"
- "I witnessed someone being excluded from meetings because of their race"
- "There's a safety hazard in the warehouse that hasn't been addressed"
- "I was told not to report the incident or I'd lose my job"
- "A colleague threatened me after I disagreed with them"

---

# What Does NOT Count as an Alert
- General complaints about workload, stress, or job satisfaction (unless they indicate serious mental health crisis)
- Constructive feedback or improvement suggestions (these are recommendations, not alerts)
- Minor interpersonal conflicts without harassment or discrimination elements
- General dissatisfaction with training, management style, or policies (unless indicating serious violations)
- Requests for resources or process improvements

---

# Key Rules
- Use only the information **explicitly present** in the comment.
- DO NOT infer alerts from vague dissatisfaction — there must be clear indication of a serious concern.
- DO NOT escalate routine complaints into alerts unless they describe genuinely problematic situations.
- Identify **all distinct** alerts in the comment.
- Each alert must correspond to a specific identifiable excerpt.
- If multiple distinct concerns appear, extract them as separate alert items.
- If the comment contains only routine feedback or minor complaints, return has_alerts: false.
- **Be especially vigilant** for subtle indicators of harassment, discrimination, or retaliation that may be worded indirectly.

---

# Alert Types

For each alert, assign one **alert_type** from the list below:

| Alert Type | Description | Example Indicators |
|------------|-------------|-------------------|
| **discrimination** | Unfair treatment based on protected characteristics (race, gender, age, religion, disability, etc.) | "treated differently because of", "excluded due to", "overlooked for promotion because I'm" |
| **sexual_harassment** | Unwanted sexual advances, requests for sexual favors, or other verbal/physical harassment of a sexual nature | "inappropriate comments about appearance", "unwanted touching", "sexual jokes", "asked me out repeatedly" |
| **workplace_violence** | Physical violence, threats of violence, or intimidating behavior | "threatened", "physical altercation", "afraid for my safety", "aggressive behavior" |
| **safety_concern** | Hazardous conditions, unsafe practices, or violations of safety protocols | "unsafe equipment", "no safety training", "hazardous materials", "accident waiting to happen" |
| **ethical_violation** | Fraud, corruption, misuse of company resources, or violations of ethical standards | "falsified records", "misused funds", "asked to lie", "cover up" |
| **hostile_environment** | Pervasive pattern of intimidation, ridicule, or offensive conduct that creates an abusive atmosphere | "constantly belittled", "toxic atmosphere", "everyone gangs up on", "culture of fear" |
| **retaliation** | Adverse action taken against someone for reporting concerns or exercising legal rights | "punished for speaking up", "demoted after complaining", "told not to report" |
| **bullying** | Repeated mistreatment, verbal abuse, or deliberate sabotage by colleagues or supervisors | "constantly criticized in front of others", "undermined my work", "publicly humiliated" |
| **substance_abuse** | Concerns about drug or alcohol use affecting workplace safety or performance | "came to work intoxicated", "using drugs on site", "impaired while operating equipment" |
| **mental_health_crisis** | Indicators of severe mental distress, self-harm risk, or crisis requiring immediate support | "thinking about ending it all", "can't go on", "severe depression", "suicidal thoughts" |
| **data_breach** | Unauthorized access to, disclosure of, or loss of confidential information | "customer data exposed", "security protocols violated", "leaked confidential information" |
| **fraud** | Intentional deception for financial or personal gain | "stealing from company", "fake invoices", "embezzlement", "kickback scheme" |
| **other_serious_concern** | Any other serious workplace issue requiring urgent attention that doesn't fit other categories | Various situations requiring escalation |

---

# Severity Levels

For each alert, assign a **severity** level:

| Severity | When to Use |
|----------|-------------|
| **low** | Minor concerns that should be documented but don't require immediate action |
| **moderate** | Concerns requiring timely follow-up and investigation |
| **high** | Serious issues requiring prompt attention and intervention |
| **critical** | Imminent threats to safety, ongoing criminal activity, or situations requiring immediate response |

Guidelines:
- **sexual_harassment**, **discrimination**, **workplace_violence**, **retaliation**: Usually moderate to critical depending on specifics
- **mental_health_crisis** with self-harm indicators: Critical
- **safety_concern** with imminent danger: High to critical
- **bullying**, **hostile_environment**: Typically moderate to high
- Use context clues about severity, frequency, and impact

---

# Output Format
Return a JSON object with:
- **has_alerts**: Boolean (true if at least one alert is present, false otherwise)
- **alerts**: List of objects, each containing:
    * **excerpt**: The exact text span from the comment containing the alert.
    * **reasoning**: One sentence explaining why this qualifies as an alert and what concern it raises.
    * **alert_type**: One of the alert type literals listed above.
    * **severity**: One of: "low", "moderate", "high", or "critical".

---

# Examples

**Example 1:**
Comment: "My supervisor constantly makes jokes about my accent and asks where I'm 'really' from. It makes me uncomfortable."
Output:
{{
  "has_alerts": true,
  "alerts": [
    {{
      "excerpt": "My supervisor constantly makes jokes about my accent and asks where I'm 'really' from",
      "reasoning": "Describes repeated comments targeting national origin or ethnicity, indicating potential discrimination.",
      "alert_type": "discrimination",
      "severity": "high"
    }}
  ]
}}

---

**Example 2:**
Comment: "The training was too long and the room was cold. I think we need better scheduling."
Output:
{{
  "has_alerts": false,
  "alerts": []
}}

---

**Example 3:**
Comment: "I reported safety concerns about the machinery to my manager, but nothing was done. Now I'm being assigned to less desirable shifts and I think it's because I spoke up."
Output:
{{
  "has_alerts": true,
  "alerts": [
    {{
      "excerpt": "I reported safety concerns about the machinery to my manager, but nothing was done",
      "reasoning": "Reports unaddressed safety hazards that could lead to workplace injuries.",
      "alert_type": "safety_concern",
      "severity": "high"
    }},
    {{
      "excerpt": "Now I'm being assigned to less desirable shifts and I think it's because I spoke up",
      "reasoning": "Describes potential retaliation for reporting safety concerns, a protected activity.",
      "alert_type": "retaliation",
      "severity": "high"
    }}
  ]
}}

---

**Example 4:**
Comment: "One of my coworkers came to work clearly intoxicated yesterday and was operating heavy equipment. I'm worried someone will get hurt."
Output:
{{
  "has_alerts": true,
  "alerts": [
    {{
      "excerpt": "One of my coworkers came to work clearly intoxicated yesterday and was operating heavy equipment",
      "reasoning": "Reports substance abuse combined with operation of dangerous equipment, creating immediate safety risk.",
      "alert_type": "substance_abuse",
      "severity": "critical"
    }}
  ]
}}

---

**Example 5:**
Comment: "The training didn't cover the new software features we need. I recommend adding a module on the advanced tools."
Output:
{{
  "has_alerts": false,
  "alerts": []
}}

---

**Example 6:**
Comment: "My team leader yells at me in front of everyone when I make mistakes and has said I'm too stupid to do this job. This happens almost daily."
Output:
{{
  "has_alerts": true,
  "alerts": [
    {{
      "excerpt": "My team leader yells at me in front of everyone when I make mistakes and has said I'm too stupid to do this job. This happens almost daily.",
      "reasoning": "Describes repeated public humiliation and verbal abuse creating a hostile work environment.",
      "alert_type": "bullying",
      "severity": "high"
    }}
  ]
}}

---

Now analyze the comment and return your response.
"""
