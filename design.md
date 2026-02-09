start up
powershell
	- wsl: wsl --list --verbose      if ubuntu isn't running: wsl -d Ubuntu
	- docker: systemctl status docker    or    sudo service docker start (if it's stopped)
	- check docker can see my gpu: docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

ide
	- connect to wsl ubuntu distro, navigate to project
	- activate venv: source venv/bin/activate

---------------------------------------------------------------------------------

POWERSHELL
- Stop everything including wsl
    - wsl --shutdown

wsl --list --verbose
    - Says what's running and what I have installed

wsl --terminate Ubuntu

wsl -d Ubuntu
    - Start an instance and log in as default user

UBUNTU
- cd OneDrive/Desktop/ALL_CODE/Synthetic_Data_Generation

- this should be running
    - systemctl is-system-running

- verify wsl2 gpu access: nvidia-smi
    - should see a table with my gpu

- start docker: systemctl status docker    or    sudo service docker start (if it's stopped)
	- docker should be running
	- docker --version

- check nvidia toolkit
	- nvidia-ctk --version
		- if this fails then docker can't see my gpu / nvidia toolkit has a problem

- check docker config (checks nvidia runtime is registered)
	- cat /etc/docker/daemon.json

- verify docker can see my gpu
    - docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
        - should print a gpu table
        
-------------------------------------------------------------------------------

**Project prompt**
I'm getting started using nvidia nemo data designer to create synthetic data based on my human made stock trades. I'm using wsl2 ubuntu on windows 11, and docker inside ubuntu (no windows gui). I have my nvidia toolkit and nvidia ngc installed

tutorials: https://docs.nvidia.com/nemo/microservices/latest/manage-entities/tutorials/set-up-proj-entities.html
readme for the nemo data designer: https://github.com/NVIDIA-NeMo/DataDesigner/blob/main/README.md
documentation: https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/index.html
https://github.com/NVIDIA-NeMo
https://github.com/NVIDIA-NeMo/DataDesigner?tab=readme-ov-file

using a seed (existing data I want to expand)
1. separate data into dev, mini prod, and full prod using different models

2. organize my dataset (seed dataset). This is called "seed data", I can reference any column in promps using jinja templating
	- the dataset must be configured in the nemo data store store microservice (general entity). this is accessable by most microservices and is the default file storage and exposes api's compatible with hugging face hub client
	- how do I get my dataset configured?
		- For example, a dataset entity can be associated with a set of files such as train.json, validation.json, and test.json. NeMo Data Store microservice handles all that
		- json, jsonl, csv format, no missing fields (except category and source), all strings must be non-null
		- Upload your dataset to NeMo Data Store using the Hugging Face CLI or SDK
		- Register your dataset in NeMo Entity Store using the Dataset APIs
		- Reference in evaluation configs using the hf://datasets/{namespace}/{dataset-name} format (hugging face and jinja template)




3. organize my columns. 



vocab
- jinja templating
	- create 1 doc template with placeholders that get auto filled with info from a datasheet. so it's like an f-string in python but has variables and has "tags", the tags are like "if" and "for" statements. This means 1 prompt can generate 1 output per row in the datasheet
	- "Write a {{ tone }} email to {{ customer_name }} thanking them for their recent purchase of {{ product }}."
		- this is like a for loop over a csv file

- nemo entity store
	- storage for all entities (idk if datasets are duplicated here as well)

- nemo dataset store
	- just for datasets (seeds). it's a private copy of hugging face's implementation so it's weird

- nvidia ngc cli
	- access nvidias hub of gpu software. app store for ai. like optimized docker containers, pretrained models, sdk's, etc. use it to pull the exact environment I need



	• Install nemo microservice sdk data designer
	• I can access a privacy setting for my dataset if I want

	• Required: linux
		○ Get docker
		○ Get pip
		○ Get python running on it
		○ Pip install nemo-microservice[data-designer]
		○ I run this by connecting cursor to wsl2
		○ Test I have connection via the api
	• Once I'm connected
		○ Possible tutorial system
			§ https://docs.nvidia.com/nemo/microservices/latest/get-started/setup/minikube/index.html#nemo-ms-get-started-prerequisites
			§ Using this changes the NEMO_MICROSERVICES_BASE_URL that I use
		a. Configure the models
			i. https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/configure-models.html#gsd-configure-models
			ii. setting API endpoints and authentication keys in the deployment configuration.
			iii. define specific models, aliases, and inference parameters. This can be configured in your SDK code or API requests.
			iv. I should have access to nvidias models, or I can basically import models
				1) If I use nvidia build models, set the name as "nvidiabuild"
				2) Each provider (nvidia build) has a provider_type key defining the api format. Default is "openai" which is the openai api specification. If a provider doesn’t follow that I need to change it.
			v. Model provider config is in my docker compose yaml.
			vi. Provider registry config (deploy time)
				1) This defines which llm service endpoints are available to data designer. If you use multiple, set a default. They need a few fields in the docs
				2) After this is doen, I can config models, aliases, and inference params later via the sdk or api requests
			vii. Api keys should bea  referece to either an env var or a key-value in json
			viii. Applications layer (define params)
				1) They're in a modelconfig class to connect my columsn to ai models
			
		b. Configure the seed datasets and columns you want to use to diversify your dataset.
		
		c. Configure your LLM generated columns with prompts and structured outputs.
		
		d. Preview your dataset and iterate on your configuration.
		
		e. Generate data at scale.
		
		f. Evaluate the quality of your data.

**what models can I choose to use?**
available models
- any (openai, claude, nvidia...)

combining them
- this is a multi llm system. it can use 2+ llm's. I need to make a chain of dependencies to achieve this (assign different llm's to different steps) in the pursute of 1 goal
	- the personas are handled under the hood. but I can use a sampler column (samplerType.person or category) to generate thousands of unique profiles (age, job, location...)
	- next, I make a llmTextColumn that takes that data as a variable. this is a data generator
	- next, I make another llmTextColumn as a judge. these 2 models go in a loop
- for schema correction, that's under teh hood automated (by structured outputs / pydantic). I write a class defining the schema. the tool forces llm's to output data to match that schema. it will block wrong formatted data
- basically I need a DAG architecture.
	- [Sampler: Personas] -> [LLM 1: Generator (uses Persona)] -> [LLM 2: Judge (evaluates Generator)] -> [Filter: Keep only High Scores]

- if I were to take next steps and really maximize this as far as possible
	- workflow should adhear to a "data flywheel" concept which is cyclic loops, external knownledge injection, increasing complexity layers.
	- replace the generator judge setup with: generator, judge, refiner (fix the code in column 1 based on the crituqe in column 2), this fixes bad data instead of throwing it away
	- inject grounding seeds. synthetic data suffers from mode collapse. we load a small csv of gold examples, use few shot prompting to pull 2-3 random examples into 1 prompt context
	- basically now I ask an llm to the task harder. "write a bubble sort" llm1: "rewrite it to be more complex, add contraints" llm2: "solve the new problem"
	- if I'm working with data requiring facts then the llm will hallucinate. use nemo retriever. it generates a topic (gpu optimization), queries my vector db for actual documentation on the topic, uses that context to write a tutorial
	graph TD
		A[Seed Data (Real Examples)] --> B[Complicator LLM]
		B --> C{Complexity Check}
		C -- "Too Simple" --> B
		C -- "Good" --> D[Generator LLM]
		D --> E[Judge LLM]
		E -- "Score < 4/5" --> F[Refiner LLM]
		F --> G[Final Dataset]
		E -- "Score 5/5" --> G
	


Just follow the github
https://github.com/NVIDIA-NeMo/DataDesigner?tab=readme-ov-file
	1. Install it
	2. Use nvidia build api: https://build.nvidia.com/. Set the provider and keys
	3. Link my columns
	4. Use a 8b model for light testing since I'll for sure get it wrong
	5. Now just slam it and iterate



steps I took
- copy human data to wsl home: cp /mnt/c/Users/YourUsername/Documents/data.csv ~/
	- linux is actually in my file explorer now. you can do it like that instead



### video notes
notable actions
- need a nvidia ngc api key, then get the ngc cli. this is a pain to hook up, but once you get it right you have access to nvidias suite of ai tools and premade docker containers and stuff. we get the nemo microservices quickstart from it, which is isn't a docker container but rather a deployment kit (folder of config files) telling docker how to set things up. it has a yaml and env file in it. we get the docker images from nvidias registry (nvcr.io).
https://docs.nvidia.com/nemo/microservices/latest/guardrails/docker-compose.html

- you also need a hugging face write token in order to upload out datasets


this is the link for the docker quickstart you're downloading: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo-microservices/resources/nemo-microservices-quickstart?version=25.12. it downloads auditor, data designer, evaluator, guardrails (all nemo)
- docs for those 4 things
	- https://docs.nvidia.com/nemo/microservices/latest/audit/docker-compose.html
		- probes models for security vunerabilities. not totally sure about this, but I guess I don't need it since I'm not deployment public models
	- https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/docker-compose.html
		- the generator
	- https://docs.nvidia.com/nemo/microservices/latest/evaluate/docker-compose.html
		- tests quality of llms or datasets. either runs benchmarks or runs a judge llm
	- https://docs.nvidia.com/nemo/microservices/latest/guardrails/docker-compose.html
		- a bar bouncer. layer between user and llm that blocks topics, prevents hallucinations, and keeps the model on track. my understanding is this is mostly for like live things like a chat bot.
- what they are
	- nemo lifecycle (design -> guardrails -> evaluate -> audit)
	- to generate stock trades we only care about data designer right now. the others are different stages of llm deployment and safety

after you have containers running
- send dataset to the datastore (in the container)
- looking at Upload_seed_dataset()
	- you set the env variables yourself
	- in my case, the repo is in a local container so this is a local folder for me. it can be anything

Next, I think I have to register the dataset in nemo entity store using the dataset api
- https://docs.nvidia.com/nemo/microservices/latest/manage-entities/datasets/create-dataset.html#entities-datasets-create
- the files is should point to the repo id we made with the datastore
- note that theirs a 2nd entity store, I'm not sure if it should come into play

ogranization
- you should assess how structured everything is now before continuing.
- namespace diagram: https://docs.nvidia.com/nemo/microservices/latest/manage-entities/namespaces/index.html
- create a namespace in the data designer. see all the namespace functions
- see all the projects functions. projects are under a namespace (you likely want a projects namespace)

Next, we add columns
- 


TODO: do I make 1 namespace with 1 project, or do I separate them?
TODO: might need to limit ram useage soemhow. 90% of my ram is used when the docker containers run



misc notes
- when it says docker compose, that's fine, compose is a plugin you should already have



I'm here: https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/tutorials/index.html
 on this doc
https://github.com/NVIDIA/GenerativeAIExamples/blob/main/nemo/NeMo-Data-Designer/self-hosted-tutorials/getting-started/1-the-basics.ipynb


I need to tell the config_builder to use my hugging face datasets, and reference the data via the datadesigner client

**regarding adding columns to models**
example
    config_builder.add_column(LLMTextColumnConfig(
        name="trade_analysis",
        # The {{ ticker }} and {{ open_price }} must match your CSV headers exactly
        prompt="Analyze the trading volume for {{ ticker }} opening at ${{ open_price }}. Is this high volume?",
        model_alias="main-model"
    ))
question: I need it to generate synthetic data of all columns, like double my dataset size. I don't understand how your analysis_column comes into play in regard to generating what the Atr14 financial value should be for ticker x at 6:35am

I'm thinking of tabular data augmentation (making the csv longer by adding more rows). this is about data enrichment though (spreadsheet wider by adding new columns). to add more rows I'd have to trick nemo by treating my existing rows as templates. To make a new row, I don't create a separate config for each column -> that would lead to a disjoined mess where variables don't correlate. 

Instead I make 1 output column (synthetic_row_json) and ask teh llm to generate a json obj containing all data points at once
	so for each row, I do "Here is a real trade: {{ ticker }} at {{ time }} for {{ price }}. Create a NEW hypothetical trade that is similar but distinct. Output the result as JSON." then it returns: {"ticker": "NVDA", "time": "9:35 AM", "price": "105.50", "atr14": "1.2"}
	- this means the llm is guessing the other columns unless I add those columns to the prompt. 
	- let's say I want atr14 to be exscpecially accurate
		- option 1: pass atr14 to the prompt and say like "generate a new trade where atr14 is slightly higher", then the llm will give me a number that looks like an atr14 but it's still mathmatically perfect
		- option 2: I use the llm to generate the human or market elements (price, volumn, time) and then I recaluate teh technical indicators myself. llm's are bad at floating point math.
	- is this the correct approach? yes

what I need to do is genrate the core data via llm's, then do the technicals myself even. it's impossible to mimic tos's numbers but I can get close enough for this purpose.

**llm columns vs sampler columns**
- sampler
	- the one who provides a topic or direction for the data. no ai, it's algorithmic and picks values from a list I give
	- these force diversity by injecting instructions into the prompt before the ai sees it

- llm columns
	- use llm's to generate new content based on a prompt. like an improve actor acting off a topic or direction
	- ai tends to revert back to teh average, so you don't get outliers or edge cases

- how to combine them
	- if I have a column bear/bull/flat market conditions then the llm could pick bull every time. a sampler column forces distribution/diversity. I tell the sampler for every row pick 1 of 3 options, then tell the llm to use that new sample column variable
	market_condition_sampler = SamplerColumnConfig(
		name="market_condition",
		data=["Bull Market", "Bear Market", "High Volatility Chopping", "Low Volatility Flat"],
		# You can even weight them! (e.g., make Bear Markets rare)
		# weights=[0.3, 0.1, 0.2, 0.4] 
	)
	then the llm column includes this
	...
	If {{ market_condition }} is 'Bear Market', ensure the price drops significantly.
    If {{ market_condition }} is 'High Volatility', ensure the High/Low spread is wide.
    ...


**models and gpu handling**
- my output is a small json, so max_tokens can be 512. if it's too high it might print the result muiltiple times (maybe, that seems odd to me)
- the minstal nemo nintron 8b is low vram
- code should prevent an infiite loop bill scenario. use a max_job_time check that kills the job
- I'm set up to use nvidia nim (serverless api's via build.nvidia). brev nvidia is renting a linux server with dedicated gpu and pay per hour. I do this if I want to replace the api with my own self-hosted model to save tokens on massive runs. So I think this is if I can't host the system, but both cases use cloud gpu's
- using the really low power 8b model should just work if I run the code
- if I ran a 340b model, the same process applies, it'll just work. brev is just the infrastrurure, it's a full computer I rent by the hour. so you use it if you're running a 12 hour job and do'nt want your computer up, or if your computer can't handle the docker containers and stuff. also nim tokens are expensive at scale and thus it's cheaper to use brev



**MODELS**
Generator models role: use volaility_percent as the diversity column, apply a jinja2 prompt to all rows. this generates a new column which has a json for each row. then use a function to move those jsons to their own row. this is 2x data.

judge: each result from the generator llm be processed by a judge llm who passes or fails each data result. if it fails it then the generator has to try that data point again until it gets a pass.
	- do this by making a 2nd llmtextColumn
	- here's how to start/stop the nemo evaluator (evaluator: The service provides automated workflows for over 100 academic benchmarks, LLM-as-a-judge scoring, and specialized metrics for RAG and agent systems.) https://docs.nvidia.com/nemo/microservices/latest/evaluate/index.html


this system uses nvidia nemo data designer to generate synthetic data of a stock trading logs. currently it handles sending the dataset to the necessary places (entity store and hugging face hub), and then using volatility_percent to diversity the data and calling a generator llm. 

Now, I need a judge llm or some way to grade each result form the generator model.

we're going to use nemo evaluator as the judge. you're going to need to read the docs to undertand how to do this in python. most or all the docker containers are already launched via setup_and_cleaning.py. First, evaluate if the setup of the system is correct to be able to use evaluator

basic overview: https://docs.nvidia.com/nemo/microservices/latest/evaluate/index.html
use llm as a judge: https://docs.nvidia.com/nemo/microservices/latest/evaluate/flows/llm-as-a-judge.html

setup_and_cleaning.py should have the endpoint and docker commands necessary, but it's untested.

I don't care about the results quality however it should more complex than soemthing like "is this valid json", I just need it to work for now, and I want python code as opposed to curl commands
