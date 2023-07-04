# Local Agent
English | [简体中文](./README.md)

## Implementation
1. This project explores using an Agent to process data from local files, ultimately generating complex content. For instance, it could input documents related to a specific industry and generate an analysis report on that industry based on this information.
2. Due to the limit on token count, we cannot pass large amounts of data directly to the language model for processing. The functions provided by traditional embedding databases are relatively basic and are mostly used for handling simple Q&A situations.

## Principles
1. I adopt the typical Tasken Driven Agent structure;
2. The OpenAI function call API and Langchain's language model are utilized in this project;
Considering the single-functionality of the embedding database, tasks must be refined sufficiently to allow the language model to locate corresponding data or information from the database;
3. Significant modifications and adjustments to the prompt are needed to create sufficiently detailed tasks. For instance, if no content is found in the embedding database, it might be due to the task not being finely segmented enough, necessitating further processing (not reflected in this repository);
4. he prompts follow the standard ReAct format.

## To Do
1. Implement the prompt's ToT (Tree of Thoughts) to handle complex data or logic reasoning problems;
2. Automate the creation of prompts. Currently, adjusting the prompts based on the task requires a lot of work;
3. Due to a high number of requests, the embedchain often experiences OpenAI request timeouts, requiring local implementation adjustments;
Add more tools; modify the Wikipedia tool.

## Code Implementation
1. Based on OpenAI and Langchain, we use Chromadb's local database;
2. To quickly achieve functionality, we have borrowed a large amount of code from aurelio-labs/funkagent and embedchain/embedchain.



