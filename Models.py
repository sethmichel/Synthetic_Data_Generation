from nemo_microservices.data_designer.essentials import (
    InferenceParameters,
    ModelConfig,
)

'''
PURPOSE: this file defines the models we're going to use
'''


# this is the external service hosting the model
MODEL_PROVIDER = "nvidiabuild"

# This sets reasoning to False for the model.
SYSTEM_PROMPT = "/no_think"

# temperature: chaos level -> controls how random the models choises are. 0.1-0.3 is focused, 0.8-1 is unpredicable
    # 0.5 is likely good me
# top_p: limits the pool of words (tokens) the model considers teh top x% probability. 1 = 100%, it considers every possible next word
    # if I use a 0.5 temperature, then I can use 0.9-1
# max_tokens: size of output. 
#             generator: makes the json of a row which is really small. 512 should be more than enough
#             judge: 

model_hash_map = {"development": 
    [
        ModelConfig(
            alias="generator-model-development",
            model="nvidia/mistral-nemo-minitron-8b-base",  # https://build.nvidia.com/nvidia/mistral-nemo-minitron-8b-base
            provider=MODEL_PROVIDER,   
            inference_parameters=InferenceParameters(
                temperature=0.5,
                top_p=0.9,
                max_tokens=512,
            ),
        ),
        ModelConfig(
            alias="judge-model",
            model="meta/llama-3.1-8b-instruct",  # https://build.nvidia.com/meta/llama-3_1-8b-instruct/modelcard
                                                 # we need an instruct variant
            provider=MODEL_PROVIDER,
            inference_parameters=InferenceParameters(
                temperature=0.5,
                top_p=0.9,
                max_tokens=512,
            ),
        ),
        ModelConfig(
            alias="refiner-model-development",
            model="nvidia/mistral-nemo-minitron-8b-base",  # https://build.nvidia.com/nvidia/mistral-nemo-minitron-8b-base
            provider=MODEL_PROVIDER,   
            inference_parameters=InferenceParameters(
                temperature=0.5,
                top_p=0.9,
                max_tokens=512,
            ),
        ),
    ],
    "mini_prod": [],
    "full_prod": 
    [
        ModelConfig(
            alias="generator-model-full-prod",
            model="nvidia/mistral-nemotron",  # https://build.nvidia.com/mistralai/mistral-nemotron/modelcard
            provider=MODEL_PROVIDER,   
            inference_parameters=InferenceParameters(
                temperature=0.5,
                top_p=0.9,
                max_tokens=512,
            ),
        ),
        ModelConfig(
            alias="judge-model-full-prod",
            model="nvidia/mistral-nemotron",  # https://build.nvidia.com/mistralai/mistral-nemotron/modelcard
            provider=MODEL_PROVIDER,
            inference_parameters=InferenceParameters(
                temperature=0.5,
                top_p=0.9,
                max_tokens=512,
            ),
        ),
        ModelConfig(
            alias="refiner-model-development",
            model="nvidia/mistral-nemo-minitron-8b-base",  # https://build.nvidia.com/nvidia/mistral-nemo-minitron-8b-base
            provider=MODEL_PROVIDER,   
            inference_parameters=InferenceParameters(
                temperature=0.5,
                top_p=0.9,
                max_tokens=512,
            ),
        ),
    ] 
}
