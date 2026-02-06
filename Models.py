from nemo_microservices.data_designer.essentials import (
    CategorySamplerParams,
    DataDesignerConfigBuilder,
    InferenceParameters,
    LLMTextColumnConfig,
    ModelConfig,
    NeMoDataDesignerClient,
    PersonSamplerParams,
    SamplerColumnConfig,
    SamplerType,
    SubcategorySamplerParams,
    UniformSamplerParams,
)

'''
PURPOSE: this file defines the models we're going to use
'''


# this is the external service hosting the model
MODEL_PROVIDER = "nvidiabuild"

# This sets reasoning to False for the model.
SYSTEM_PROMPT = "/no_think"

model_configs = [
    ModelConfig(
        alias="generator-model" ,
        model="nvidia/nemotron-3-nano-30b-a3b",
        provider=MODEL_PROVIDER,   
        inference_parameters=InferenceParameters(
            temperature=0.5,
            top_p=1.0,
            max_tokens=1024,
        ),
    ),
    ModelConfig(
        alias="judge-model",
        model="",
        provider=MODEL_PROVIDER,
        inference_parameters=InferenceParameters(
            temperature=0.5,
            top_p=1.0,
            max_tokens=1024,
        ),
    )
]
