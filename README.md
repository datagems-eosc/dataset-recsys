
# DataGEMS RecSys 

Implementation of the DataGEMS recommendation services, designed to support dataset discovery across the DataGEMS platform. The goal of these services is to reduce the manual effort required to locate relevant datasets by identifying related resources. 

In addition to powering dataset discovery on the platform, the recommendation services will also be demonstrated within the four DataGEMS use cases (i.e., higher education, lifelong learning, language, and weather), where domain-specific recommenders are instantiated and showcased. In these use cases, the recommended items are not datasets: the type of item being recommended depends entirely on the application (e.g., educational materials for Higher Education, skills for Lifelong Learning, linguistic resources for Language, etc.). The recommender service adapts its input, representations, and retrieval strategy accordingly.

The item-to-item recommendation pipeline consists of four components:

1. **Metadata ingestion layer**: the textual information associated with each item.
2. **Representation module**: transforms this information into numerical embeddings.
3. **Candidate generator**: retrieves a pool of potentially related items.
4. **Re-ranking module**: orders the retrieved candidates according to their relevance.

