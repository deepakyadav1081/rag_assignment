import json
from docassist import agent_query  # Update if your agent_query is in another file

test_queries = [
    "What was NVIDIA's total revenue in fiscal year 2024?",
    "What percentage of Google's 2023 revenue came from advertising?",
    "How much did Microsoft's cloud revenue grow from 2022 to 2023?",
    "Which of the three companies had the highest gross margin in 2023?",
    "Compare the R&D spending as a percentage of revenue across all three companies in 2023",
    "How did each company's operating margin change from 2022 to 2024?",
    "What are the main AI risks mentioned by each company and how do they differ?"
]

results = []
for q in test_queries:
    result = agent_query(q, top_k=3)
    print(f"Query: {q}")
    print(json.dumps(result, indent=2))
    print("="*80)
    results.append(result)

with open("responses.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)