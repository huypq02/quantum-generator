from abc import ABC, abstractmethod


class IClock(ABC):
    @abstractmethod
    def current_utc_timestamp(self) -> float:
        """Return current timestamp in UTC"""
        pass
