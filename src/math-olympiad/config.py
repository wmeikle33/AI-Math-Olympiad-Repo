Class Config:
    notebook_limit = (4 * 60 + 55) * 60  # 4h 55m
    high_problem_timeout = 900
    base_problem_timeout = 300
    server_timeout = 180
    session_timeout = 960
    jupyter_timeout = 15
    sandbox_timeout = 5

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
