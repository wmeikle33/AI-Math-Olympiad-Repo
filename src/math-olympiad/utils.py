def set_env(input_archive, temp_dir):

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        
        subprocess.run(['tar', '-xzf', input_archive, '-C', temp_dir], check=True)
    
    subprocess.run([
        sys.executable, 
        '-m', 
        'pip', 
        'install', 
        '--no-index', 
        '--find-links', 
        f'{temp_dir}/wheels', 
        'unsloth', 
        'trl', 
        'vllm', 
        'openai_harmony'
    ], check=True)

class AIMO3Tool:
    """Python code execution tool using Jupyter kernel sandbox."""

    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox=None):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        self._owns_session = sandbox is None
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        """Lazily initialize the Jupyter session if not provided."""
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    def _ensure_last_print(self, code: str) -> str:
        """Wrap the last expression in print() if it's not already printing."""
        lines = code.strip().split('\n')
        
        if not lines:
            return code

        last_line = lines[-1].strip()

        # Skip if already has print, import, empty, or comment
        if not last_line or last_line.startswith('#'):
            return code
        if 'print' in last_line or 'import' in last_line:
            return code
        # Skip control flow statements
        if last_line.endswith(':') or last_line.startswith(('return', 'break', 'continue', 'pass', 'raise')):
            return code

        # Remove inline comment before wrapping
        if '#' in last_line:
            last_line = last_line.split('#')[0].strip()

        lines[-1] = f'print({last_line})'
        return '\n'.join(lines)

    @property
    def instruction(self) -> str:
        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name='python', 
            description=self.instruction, 
            tools=[]
        )

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')

        if channel:
            message = message.with_channel(channel)

        return message

    def process_sync_plus(self, message: Message, timeout: float | None = None) -> list[Message]:
        """Execute code from message using Jupyter kernel."""
        # Validate message content
        if not message.content or len(message.content) == 0:
            return [self._make_response('[ERROR] Message has no content', channel=message.channel)]
        
        try:
            script = message.content[0].text
            if not script or not script.strip():
                return [self._make_response('[ERROR] Empty script provided', channel=message.channel)]
        except (AttributeError, IndexError, TypeError) as e:
            return [self._make_response(f'[ERROR] Failed to extract script: {e}', channel=message.channel)]

        self._ensure_session()
        final_script = self._ensure_last_print(script)

        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script, timeout=timeout)
            except TimeoutError as exc:
                output = f'[ERROR] Execution timeout: {exc}'
            except RuntimeError as exc:
                output = f'[ERROR] Runtime error: {exc}'
            except Exception as exc:
                output = f'[ERROR] Unexpected error: {exc}'

        return [self._make_response(output, channel=message.channel)]

    def close(self):
        """Close the Python tool and cleanup resources."""
        with self._init_lock:
            if self._jupyter_session is not None and self._owns_session:
                try:
                    self._jupyter_session.close()
                except Exception:
                    pass
                finally:
                    self._jupyter_session = None

    def __del__(self):
        self.close()
