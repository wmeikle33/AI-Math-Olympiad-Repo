class Model:

    def __init__(self):
        self._model = None

    def load(self):
        """Simulate model loading."""
        print("Loading model...")
        # Just return a "model" that always answers with 0
        return lambda problem: 0

    def predict(self, problem: str):
        # Employ lazy loading: load model on the first model.predict call
        if self._model is None:
            self._model = self.load()
        return self._model(problem)

class AIMO3Sandbox:

    _port_lock = threading.Lock()
    _next_port = 50000
    _max_port = 65535

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        """Allocate unique ports for kernel communication."""
        with cls._port_lock:
            ports = []
            start_port = cls._next_port
            
            for i in range(count):
                port = start_port + i
                if port > cls._max_port:
                    start_port = 50000
                    port = start_port + i
                ports.append(port)
            
            cls._next_port = start_port + count
            if cls._next_port > cls._max_port:
                cls._next_port = 50000
            
            return ports
