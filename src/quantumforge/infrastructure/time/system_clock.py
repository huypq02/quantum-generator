import datetime
from src.quantumforge.domain.interfaces.clock import IClock


class SystemClock(IClock):
    def current_utc_timestamp(self):
        """Get the current timestamp in UTC"""
        current_utc_datetime = datetime.datetime.now(datetime.timezone.utc)

        return current_utc_datetime.timestamp()
