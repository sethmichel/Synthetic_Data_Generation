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
I'm getting started using nvidia nemo data designer to create synthetic data based on my human made stock trades. I'm using wsl2 ubuntu on windows 11, and docker inside ubuntu (no windows gui). I also have my nvidia toolkit installed, and I've verified everything is correct so far (like docker can see my gpu).

readme for the nemo data designer: https://github.com/NVIDIA-NeMo/DataDesigner/blob/main/README.md
documentation: https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/index.html




Installations
- wsl2 with ubuntu
- on windows: latest nvidia game ready driver or latest stuidio driver (not in ubuntu)


https://github.com/NVIDIA-NeMo
https://github.com/NVIDIA-NeMo/DataDesigner
https://github.com/NVIDIA-NeMo/DataDesigner?tab=readme-ov-file


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