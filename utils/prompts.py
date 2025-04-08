GENERATOR_PROMPT = """
You are a data generation agent.

Your goal is to generate a *new record* (a dictionary-like structure) that conforms strictly to the given dataframe schema.

### Schema
The dataframe has the following schema. Each field has a name and a value type:
{schema}

### Examples of Correct Record

{example}
{feedback}
---

### Instructions

- Generate a new record where each field matches the specified data type in the schema.
- Correct the mistakes identified in the feedback above.
- Ensure that the new record is *plausible* and *coherent* (values should be realistic, not random).
- If a field has constraints implied by its type or name (e.g., date format, ID pattern, allowed categories), respect them.

### Output Format
Respond only with the new record in YAML, without additional commentary.
"""


GENERATOR_FEEDBACK = """

---

### Feedback on Your Previous Generation
Your previous generated record contained errors. Below are the fields identified as incorrect, with an explanation of why they were wrong and suggestions for correction:
{wrong_columns}
"""


FEEDBACK_PROMPT = """
You are a feedback agent.

Your role is to assist a generator agent in producing a correct record for a dataframe.

---

### Schema
The dataframe has the following schema. Each field has a name and a value type:
{schema}

---

### Record to Validate
The generator agent produced the following record:
{record}

---

### Instructions

- Identify the fields in the record that do *not* conform to the schema (wrong data type, format, invalid value, incoherent content, etc.).
- For each incorrect field, provide:
    - Field name.
    - Short explanation of why it is incorrect.
    - What is a possible correct value for this field.

### Output Format
Respond with a JSON object where keys are the incorrect field names and values are explanations with suggested corrections.
"""