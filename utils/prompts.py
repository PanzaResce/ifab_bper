GENERATOR_PROMPT="""You are an agent that has to generate a new record for a dataframe.
The dataframe has the following schema, with the column names alongside their value type.
Schema:
{schema}

{feedback}
The generated record must be coherent with the dataframe schema.
For each column, generate a corrisponding value considering its data type as described in the schema"""

GENERATOR_FEEDBACK = """The previous record you tried to generate gave an error.
The error is the following: {error}.
This is a suggestion on how to fix the error: {suggestion} 
Use this suggestion to fix the error when generating the new record."""

FEEDBACK_PROMPT="""You are an agent that helps another generator agent to fix its generated record.
The generator agent you have to help has the task to generate a new record for a dataframe following a specific schema.
The generated record is wrong, and it raises this error when it is inserted in the input dataframe:
{errors}
The schema of the input dataframe is the following.
Schema:
{schema}

Considering the dataframe schema, tell what field was wrongly generated and what should be the correct value for that column.
Be concise and clear in your explanations.
"""