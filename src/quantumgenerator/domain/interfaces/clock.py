from abc import ABC, abstractmethod


class IClock(ABC):
    """Interface for time operations."""

    @abstractmethod
    def current_utc_timestamp(self) -> float:
        """
        Return current timestamp in UTC.
        
        :return: Current UTC timestamp.
        :rtype: float
        """
        pass
