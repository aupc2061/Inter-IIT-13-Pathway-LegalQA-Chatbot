
<h1 align="center">PathLex - Agentic RAG Application for Legal QA</h1>
<h3 align = "center">Team 67</h3>

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
  - [1. Parallel Task Execution](#1-parallel-task-execution)
  - [2. Dynamic Replanning for Retrieval](#2-dynamic-replanning-for-retrieval)
  - [3. Enhanced Retrieval Precision with Task-Specific Tools](#3-enhanced-retrieval-precision-with-task-specific-tools)
  - [4. Scalability to Complex Queries](#4-scalability-to-complex-queries)
  - [5. Plan-and-Solve Alignment](#5-plan-and-solve-alignment)
  - [6. Reduced Token Usage and Cost](#6-reduced-token-usage-and-cost)
- [Installation and Setup](#installation-and-setup)
  - [1. Setting up the Pathway VectorStore](#1-setting-up-the-pathway-vectorstore)
  - [2. Setting up the Environment Variables](#2-setting-up-the-environment-variables)
  - [3. Scripts for Quick Testing](#3-scripts-for-quick-testing)
  - [4. How to Run the Frontend](#4-how-to-run-the-frontend)
- [Solution Pipeline & Usage Guide](#solution-pipeline--usage-guide)

## Introduction
We introduce PathLex, an advanced agentic Retrieval-Augmented Generation (RAG) system specifically tailored for the legal domain. Built on Pathway’s real-time data processing capabilities and leveraging LLMCompiler's dynamic task planning, PathLex addresses critical limitations in existing legal RAG systems, such as hallucinations, retrieval inaccuracies, and long-context handling. With innovations in chunking, multi-tier replanning, and robust fallback mechanisms, including a human-in-the-loop framework, PathLex ensures precise, context-aware answers with verifiable citations. This work lays the foundation for intelligent automation in high-stakes domains, demonstrating the potential for transformative improvements in legal information systems.

## Key Features

### 1. **Parallel Task Execution**
- **Challenge:** Traditional RAG systems process retrieval queries sequentially, introducing latency when handling multiple queries.
- **Solution:** LLMCompiler employs a **planner-executor architecture** to identify independent retrieval tasks and execute them in parallel, significantly reducing latency.

### 2. **Dynamic Replanning for Retrieval**
- **Challenge:** In multi-hop queries, intermediate retrieval results often necessitate changes in subsequent queries or reasoning.
- **Solution:** LLMCompiler adapts dynamically through a **dynamic execution graph**, recomputing task dependencies as results come in, ensuring actions remain contextually relevant.

### 3. **Enhanced Retrieval Precision with Task-Specific Tools**
- **Challenge:** Generic retrieval tools often lack precision for task-specific needs.
- **Solution:** LLMCompiler integrates **specialized retrieval tools**, dynamically assigning the most relevant tool for each task to improve precision.

### 4. **Scalability to Complex Queries**
- **Challenge:** Traditional RAG systems struggle with multi-step queries involving intricate reasoning and dependencies.
- **Solution:** LLMCompiler creates **directed acyclic graphs (DAGs)** for task execution, efficiently managing complex reasoning and retrieval dependencies.

### 5. **Plan-and-Solve Alignment**
- **Challenge:** Treating retrieval and generation as a monolithic process can lead to inefficiencies.
- **Solution:** LLMCompiler breaks tasks into **manageable sub-steps** (e.g., retrieval → analysis → generation), optimizing each independently for accuracy and efficiency.

### 6. **Reduced Token Usage and Cost**
- **Challenge:** Excessive token consumption increases costs in traditional RAG workflows.
- **Solution:** Inspired by **ReWOO** ([Xu et al., 2023](https://arxiv.org/abs/2305.18323)), LLMCompiler **decouples reasoning from execution**, minimizing unnecessary LLM invocations and reducing token usage.




## Installation and Setup

### Initial Steps 
Note: You can only run Pathway's Server in linux or MacOS, to run it in Windows, it is recommended you use WSL or use Docker to run the server.

1. Create a new virtual environment and activate it by running the following commands:
```bash
    python -m venv venv
    source venv/bin/activate
```

2. Install the requirements using:
```bash
    pip install -r requirements.txt
```

### 1. Setting up the Pathway VectorStore 
This is the initial step required to run Pathway's Vector Store such that it can be connected to our main pipeline for retrieval. Pathway offers real-time data processing facilities that allow one to add or remove documents from a vector store in real-time. 

### Prerequisites
- OpenAI API key or [VoyageAI's API key](https://www.voyageai.com/)  (depending on what embeddings you decided to use, we recommend using voyage-3 by default). Replace them inplace of the placeholders in ```run-server.py```.
- We use a custom parser in our solution, so you must have ```custom_parser.py``` in the root directory
- We modified the VectorStoreServer to suit our purpose and thus you must have ```server.py``` in your root directory.
- An active internet connection is required, disruption might lead to the server crashing.



### 1. Steps to run
1. Go to pathway-server using ```cd pathway-sever```
2. Create a directory named ```/data``` (or something else) and upload your documents to that folder. 
Note: You may use other data sources as well such as google drive, just replace the file path with the drive url for it to work.
3. You have to install tesseract-ocr and replace the TESSDATA path in ```run-server.py``` in place of TESSDATA_PREFIX
4. Replace your OpenAI/VoyageAI key in ```run-server.py```
5. Finally, simply run 
```bash
    python run-server.py
```

6. The server will be hosted on ```127.0.0.1``` on port ```8745``` by default, though you may change if you wish so.

7. You can test if the server is running and working by running:
```bash
    python test_server.py
```


Do note that embeddings may take a lot of time to be created due to which it might give a read timeout error if you try to retrieve from the vector store. This goes away once the embeddings have been created, so it is recommended that you put in a few documents at a time in the server at most.

### 2. Setting up the environment variables

We use a few models and services in our pipeline. You will have to put a .env file in the root directory with the following parameters filled in:

```
OPENAI_API_KEY= your_openai_api_key
ANTHROPIC_API_KEY= your_anthropic_api_key
COHERE_API_KEY=your_cohere_api_key
LANGCHAIN_HUB_API_KEY= your_langchain_hub_api_key
LANGFUSE_SECRET_KEY= your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY= your_langfuse_public_key
LANGFUSE_HOST= https://cloud.langfuse.com
PATHWAY_HOST = 127.0.0.1
PATHWAY_PORT = 8745
```



### 3. Scripts for Quick Testing
We have provided Python scripts of our pipeline without a UI to ease testing on our solution. You may use these scripts to run our pipeline for quick evaluation though **you also have the option to run the complete UI by following the steps mentioned in the section after this.**

### Steps to run:
1. Ensure you have followed the steps mentioned in sections 1 and 2 to run the Pathway VectorStore and set up all the environment variables.
2. You can test our pipeline by running:
```bash
    python main.py "<insert query here>"
```

### 4. How to run the front end
Here we are assuming that you are running the frontend on a Windows Device. For Linux/macOS, you can find similar steps to install the respective packages online.
#### Prerequisites 
1. NodeJS
2. npm
3. Download the [NodeJS MSI](https://nodejs.org/en/#download)
4. Download the [npm zip](https://github.com/npm/npm)

#### Steps to run:
1. Extract the MSI you downloaded in step 3 (with 7 Zip) in a directory "node".
2. Set the PATH environment variable to add the "node" directory
3. Extract the zip file from npm in a different directory (not under node directory)

4. Now you should have node + npm working, use these
commands to check:
```bash
 node --version
 npm --version
```

5. To use the UI, we need to create an Atlas MongoDB API. You can follow these [steps](https://www.mongodb.com/docs/drivers/node/current/quick-start/create-a-deployment/) to create one. 

6. Add your ```.env``` file to the ```legal-chatbot-frontend\src\final_scripts``` with API keys, server local host URL, server port number or the ngrok link.

7. Replace ```MONGO_URL``` in ```legal-chatbot-frontend\src\final_scripts\global_.py``` with your own hosted mongodb api_key with the name of the Database as ```legal_chatbot```. **This is very important.**

8. This mongodb url must also be replaced in ```legal-chatbot-backend\server.js``` also, in place of ```your_mongo_url```

9. Run this now: 
```
cd legal-chatbot-frontend\
npm i
```

10. Run this now:
```
cd legal-chatbot-backend\
npm i
```

11. To start the client, run:
```
cd legal-chatbot-frontend
npm run dev
```

12. To start the server, run:
```
cd legal-chatbot-backend
node server.js
```



## 5. Solution Pipeline & Usage Guide

1. The owner logs in via the login page.
2. The user can reaccess his/her previous chats, and create a new chat.
3. In a new chat, the user can put a legal query.
4. After entering the legal query, the user can see the dynamic thought process on the left. This happens in the order of
    - Plan and Schedule: This contains the plan made by the planner to answer the given query. It makes a plan with a list of tools to call and schedules them to be executed parallelly by the task-fetching unit. 
    - Joiner: The joiner agent decides whether to proceed with Generation or to Replan based on the outputs of the tool calls executed by the scheduler. It decides this by using an LLM to analyze the results of the tool calls and create a Thought and an Action along with feedback in case it decides to re-plan.
    - Rewrite: Rewrite Agent receives the Thought and Feedback from the joiner, based on which it decides to rewrite the query. It has grade documents
function that generates a score for each chunk retrieved
by the tool. This allows it to identify ”good” documents
among the retrieved documents that are relevant to the user
query. 
    - Generate:  This agent reviews
the outputs from previous tool calls and the query to generate
the final answer that is to be shown to the user.
    - HIL: Humans can give feedback to rewrite or generate according to the retrieved docs. 

## 6. Architecture
```
├── Experiments and miscellaneous
    ├── beam_retriever_train_and_exp.py
    ├── lumber chunking.py
    └── meta chunking.py
├── HIL.py
├── README.md
├── Reports
    ├── Pathway_MidEval_Report.pdf
    └── endterm_report.pdf
├── agents.py
├── anthropic_functions.py
├── beam_retriever.py
├── beam_tool.py
├── citations.py
├── demo_videos
    ├── demo.mp4
    └── summary_video.mkv
├── globals_.py
├── imports.py
├── joiner.py
├── legal-chatbot-backend
    ├── .gitignore
    ├── cache1.txt
    ├── models
    │   ├── Chat.js
    │   ├── Script_response.js
    │   └── User.js
    ├── package-lock.json
    ├── package.json
    ├── server.js
    ├── test.txt
    └── text_files
    │   ├── generate.txt
    │   ├── human_input_node.txt
    │   ├── join.txt
    │   ├── plan_and_schedule.txt
    │   └── rewrite.txt
├── legal-chatbot-frontend
    ├── .gitignore
    ├── README.md
    ├── eslint.config.js
    ├── index.html
    ├── package-lock.json
    ├── package.json
    ├── public
    │   ├── send_btn.svg
    │   └── vite.svg
    ├── src
    │   ├── App.css
    │   ├── App.jsx
    │   ├── Home.css
    │   ├── Home.jsx
    │   ├── SignIn.css
    │   ├── SignIn.jsx
    │   ├── SignUp.css
    │   ├── SignUp.jsx
    │   ├── assets
    │   │   └── react.svg
    │   ├── files.jsx
    │   ├── final_scripts
    │   │   ├── .gitignore
    │   │   ├── HIL.py
    │   │   ├── agents.py
    │   │   ├── anthropic_functions.py
    │   │   ├── beam_retriever.py
    │   │   ├── beam_tool.py
    │   │   ├── citations.py
    │   │   ├── get_all_files.py
    │   │   ├── globals_.py
    │   │   ├── imports.py
    │   │   ├── joiner.py
    │   │   ├── main.py
    │   │   ├── output.txt
    │   │   ├── output_parser.py
    │   │   ├── pathway_server
    │   │   │   ├── custom_parser.py
    │   │   │   ├── run-server.py
    │   │   │   ├── server.py
    │   │   │   └── test_server.py
    │   │   ├── planner.py
    │   │   ├── prompts.py
    │   │   ├── requirements.txt
    │   │   ├── task_fetching_unit.py
    │   │   ├── test.py
    │   │   ├── tools.py
    │   │   └── utils.py
    │   ├── index.css
    │   └── main.jsx
    └── vite.config.js
├── main.py
├── output.txt
├── output_parser.py
├── pathway_server
    ├── custom_parser.py
    ├── run-server.py
    ├── server.py
    └── test_server.py
├── planner.py
├── prompts.py
├── requirements.txt
├── task_fetching_unit.py
├── tools.py
└── utils.py
```

## Team Members
- [Himanshu Singhal](https://github.com/himanshu-skid19)  
- [Rishita Agarwal](https://github.com/rishita3003)  
- [Ayush Kumar](https://github.com/RedLuigi1)  
- [Mahua Singh]()
- [Anushka Gupta]()
- [Ashutosh Bala]()
- [Shayak Bhattacharya](https://github.com/aupc2061)

