%pip uninstall --yes 'keras' 'matplotlib' 'scikit-learn' 'tensorflow'

import gc
import re
import math
import time
import queue
import threading
import contextlib
from typing import Optional
from jupyter_client import KernelManager
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
import json
import pandas as pd
import polars as pl
from .utils import set_env

from openai import OpenAI

from openai_harmony import (
    HarmonyEncodingName, 
    load_harmony_encoding, 
    SystemContent, 
    ReasoningEffort, 
    ToolNamespaceConfig, 
    Author, 
    Message, 
    Role, 
    TextContent, 
    Conversation
)

from transformers import set_seed, AutoTokenizer
import kaggle_evaluation.aimo_3_inference_server

set_seed(CFG.seed)

set_env(
    input_archive='/kaggle/input/notebooks/shelterw/aimo-3-vllm-v16/wheels.tar.gz', 
    temp_dir='/kaggle/tmp/setup'
)
