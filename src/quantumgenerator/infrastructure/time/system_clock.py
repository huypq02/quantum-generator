import datetime
from quantumgenerator.domain.interfaces.clock import IClock


class SystemClock(IClock):
    """System clock implementation for time operations."""

    def current_utc_timestamp(self):
        """
        Get the current timestamp in UTC.
        
        :return: Current UTC timestamp in seconds since epoch.
        :rtype: float
        """
        current_utc_datetime = datetime.datetime.now(datetime.timezone.utc)

        return current_utc_datetime.timestamp()
