import abc
from typing import Any, Optional

class DatabaseConnector(abc.ABC):
    name: str

    def __init__(self, dsn: str, db_name: Optional[str]=None):
        self.dsn = dsn
        self.db_name = db_name

    @abc.abstractmethod
    async def connect(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def disconnect(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_db(self) -> Any:
        raise NotImplementedError