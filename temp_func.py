import polars as pl

# This is what the function should look like
def func_generating_dict(row):
    """Generate a dictionary with id and job for Mistral batch API"""
    return {
        "role": "system",
        "content": "Extract the classification. Set response to 1 if ",
    }, {
        "role": "user",
        "content": f"ID: {row['id']}, Job: {row['job']}"
    }

# Test with sample data
df = pl.DataFrame({
    'id': [1, 2, 3],
    'job': ['developer', 'manager', 'designer']
})

# Apply the function to each row
messages_list = []
for row in df.iter_rows(named=True):
    system_msg, user_msg = func_generating_dict(row)
    messages_list.append([system_msg, user_msg])

print("Generated messages:")
for i, messages in enumerate(messages_list):
    print(f"Row {i}: {messages}")
