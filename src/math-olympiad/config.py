class CFG:
    
    system_prompt = (
    )
    
    tool_prompt = (
    )
    
    preference_prompt = (
    )
    
    served_model_name = 'gpt-oss'
    model_path = '/kaggle/input/models/shelterw/qwen3.5/transformers/qwen3.5-27b-fp8/1'
    
    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'
    spec_config = {
        "method": "qwen3_next_mtp",
        "num_speculative_tokens": 2
    }
    use_spec_decode = True
    
    if "glm" in model_path.lower():
        re_pattern = r"<arg_value>(.*?)</arg_value>"
    elif "nemotron" in model_path.lower() or "qwen3.5" in model_path.lower():
        re_pattern = r"<parameter=[^>]+>(.*?)</parameter>"
    else:
        re_pattern = r"<tool_call>(.*?)</tool_call>"

    high_problem_timeout = 900
    base_problem_timeout = 300

    notebook_limit = 17400
    server_timeout = 180

    session_timeout = 960
    jupyter_timeout = 6
    sandbox_timeout = 3

    stream_interval = 200
    context_tokens = 65536
    buffer_tokens = 512
    search_tokens = 32
    top_logprobs = 5
    batch_size = 256
    early_stop = 4
    attempts = 8
    workers = 16
    turns = 128
    seed = 42

    gpu_memory_utilization = 0.96
    temperature = 1.0
    min_p = 0.02
    presence_penalty=1.5

    tools = [
        {
            "type": "function",
            "function": {
                "name": "python",
                "description": tool_prompt,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute."
                        }
                    },
                    "required": ["code"]
                },
            },
        },
    ]
